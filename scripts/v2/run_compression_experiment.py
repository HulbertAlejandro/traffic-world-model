"""Run one phase of the pre-registered compression experiment (docs/v2/ADDENDUM_AUTOENCODER.md).

    python scripts/v2/run_compression_experiment.py --phase selection
    python scripts/v2/run_compression_experiment.py --phase confirmation --latent-dim K

Selection:    latent sizes 16/32/48/64/80 x Autoencoder seeds 0-2 x LSTM seeds 0-2 (45 z models),
              raw branch LSTM seeds 0-2.
Confirmation: the selected size x Autoencoder seeds 100-104 x LSTM seeds 100-104 (25 z models),
              raw branch LSTM seeds 100-109 (fresh seeds: the selection runs are not reused).

Transformer repeat of the selection (addendum section 9): --architecture transformer reuses the
selection Autoencoders and encoded splits and trains Transformers in tf_* folders:

    python scripts/v2/run_compression_experiment.py --phase selection --architecture transformer         --sizes 16 32 48 64 80 96 --tag transformer

Weights go to models/checkpoints/v2/compression/<phase>/ (not committed; the .json files are).
Runs whose evaluation already exists are skipped, so an interrupted phase can be resumed.
Results: docs/results/v2/autoencoder/<phase>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED = ROOT_DIR / "datasets" / "v2" / "processed"
CKPT_ROOT = ROOT_DIR / "models" / "checkpoints" / "v2" / "compression"
RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "autoencoder"

CANDIDATES = (16, 32, 48, 64, 80)
PHASES = {
    "selection": {"ae_seeds": (0, 1, 2), "lstm_seeds": (0, 1, 2), "raw_seeds": (0, 1, 2)},
    "confirmation": {"ae_seeds": (100, 101, 102, 103, 104), "lstm_seeds": (100, 101, 102, 103, 104),
                     "raw_seeds": tuple(range(100, 110))},
}


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)


def _ae_job(k: int, seed: int, ae_dir: str) -> dict:
    from training import v2_compression_experiment as exp
    ae_dir = Path(ae_dir)
    if not (ae_dir / "autoencoder_best.json").exists():
        exp.train_autoencoder(PROCESSED / "train.npz", PROCESSED / "validation.npz", k, seed, ae_dir)
    ae = exp.load_autoencoder(ae_dir)
    for split in ("train", "validation", "test"):
        if not (ae_dir / f"{split}_seq.npz").exists():
            exp.encode_split(ae, PROCESSED / f"{split}.npz", ae_dir / f"{split}_seq.npz")
    meta = json.loads((ae_dir / "autoencoder_best.json").read_text(encoding="utf-8"))
    return meta | {"reconstruction": exp.reconstruction_report(ae, PROCESSED / "validation.npz")}


def _lstm_job(branch: str, seq_dir: str, seed: int, out_dir: str, latent_dim: int | None,
              architecture: str) -> dict:
    from training import v2_compression_experiment as exp
    seq_dir, out_dir = Path(seq_dir), Path(out_dir)
    result_path = out_dir / "evaluation.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    hp = exp.train_branch(branch, seq_dir / "train_seq.npz", seq_dir / "validation_seq.npz", seed, out_dir,
                          latent_dim=latent_dim, architecture=architecture)
    ev = exp.evaluate(out_dir, seq_dir / "test_seq.npz")
    result = {"architecture": architecture, "branch": branch, "latent_dim": latent_dim, "seed": seed, "training": hp, "evaluation": ev,
              "seconds": time.perf_counter() - t0}
    result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--latent-dim", type=int, help="confirmation only: the size selected in the selection phase")
    # 4 by default: 8 PyTorch processes ran this 7.7 GB machine out of memory (CLAUDE.md).
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+",
                        help="selection only: run only these latent sizes (post-hoc amendment, "
                             "ADDENDUM_AUTOENCODER.md section 8); results go to <phase>_<tag>.json")
    parser.add_argument("--tag", help="suffix of the results file, required with --sizes or a non-LSTM architecture")
    parser.add_argument("--architecture", choices=("lstm", "transformer"), default="lstm",
                        help="temporal model; non-LSTM runs go to <prefix>_* folders (addendum section 9)")
    args = parser.parse_args()
    if args.architecture != "lstm" and not args.tag:
        parser.error("a non-LSTM architecture needs --tag (do not overwrite the LSTM results)")
    if (args.phase == "confirmation") != (args.latent_dim is not None):
        parser.error("--latent-dim is required for confirmation and not allowed for selection")
    plan = PHASES[args.phase]
    if args.sizes and (args.phase != "selection" or not args.tag):
        parser.error("--sizes is for the selection phase and needs --tag")
    sizes = (tuple(args.sizes) if args.sizes else CANDIDATES) if args.phase == "selection" else (args.latent_dim,)
    root = CKPT_ROOT / args.phase
    from training import v2_compression_experiment as exp

    raw_dir = root / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        if not (raw_dir / f"{split}_seq.npz").exists():
            exp.encode_split(None, PROCESSED / f"{split}.npz", raw_dir / f"{split}_seq.npz")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        ae_keys = [(k, a) for k in sizes for a in plan["ae_seeds"]]
        ae_meta = dict(zip(ae_keys, pool.map(_ae_job, *zip(*[(k, a, str(root / f"ae_k{k}_s{a}")) for k, a in ae_keys]))))
        print(f"{len(ae_keys)} autoencoders ready ({time.perf_counter() - t0:.0f} s)", flush=True)
        arch, prefix = args.architecture, {"lstm": "", "transformer": "tf_"}[args.architecture]
        jobs = [("z", str(root / f"ae_k{k}_s{a}"), s, str(root / f"{prefix}z_k{k}_ae{a}_s{s}"), k, arch)
                for k in sizes for a in plan["ae_seeds"] for s in plan["lstm_seeds"]]
        jobs += [("raw", str(raw_dir), s, str(root / f"{prefix}raw_s{s}"), None, arch) for s in plan["raw_seeds"]]
        results = []
        for i, r in enumerate(pool.map(_lstm_job, *zip(*jobs)), 1):
            results.append(r)
            if i % 8 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {arch} models done ({time.perf_counter() - t0:.0f} s)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / (f"{args.phase}_{args.tag}.json" if args.tag else f"{args.phase}.json")
    out.write_text(json.dumps({
        "phase": args.phase, "architecture": args.architecture, "plan": plan, "sizes": list(sizes),
        "autoencoders": [{"latent_dim": k, "seed": a, **m} for (k, a), m in ae_meta.items()],
        "models": [{**r, "ae_seed": None if r["branch"] == "raw" else int(Path(j[1]).name.split("_s")[-1])}
                   for r, j in zip(results, jobs)],
        "wall_seconds": time.perf_counter() - t0,
    }, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
