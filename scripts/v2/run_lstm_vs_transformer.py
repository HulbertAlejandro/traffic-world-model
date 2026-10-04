"""Train the pre-registered, parameter-matched LSTM vs Transformer comparison
(docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md).

    python scripts/v2/run_lstm_vs_transformer.py [--workers 4]

Both architectures on the uncompressed 104-dim state, no Autoencoder, seeds 0-9 each, all trained
from scratch (the Phase 2 selection runs have other sizes and are not reused). Sizes matched to
~150k parameters: LSTM hidden 137 (152,038), Transformer d_model 80 (152,617). Training protocol:
Phase 2's LSTM_PROTOCOL object, through train_branch("raw", ...) -> train_temporal_model; only the
architecture hyperparameters are overridden. Same encoded splits as Phase 2 (selection/raw_data).
Runs whose evaluation.json already exists are skipped, so an interrupted run can be resumed.

Weights: models/checkpoints/v2/arch_comparison/<arch>_raw_s<seed>/ (not committed; the .json are).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RAW_DATA = ROOT_DIR / "models" / "checkpoints" / "v2" / "compression" / "selection" / "raw_data"
OUT_ROOT = ROOT_DIR / "models" / "checkpoints" / "v2" / "arch_comparison"
SEEDS = tuple(range(10))
# Parameter-matched sizes (addendum section 2). Not v1's 128/128, which Phase 2 keeps.
ARCH_HPARAMS = {
    "lstm": {"hidden_dim": 137},
    "transformer": {"d_model": 80, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.0},
}
EXPECTED_PARAMS = {"lstm": 152_038, "transformer": 152_617}


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)


def _job(arch: str, seed: int) -> dict:
    from training import v2_compression_experiment as exp
    out_dir = OUT_ROOT / f"{arch}_raw_s{seed}"
    result_path = out_dir / "evaluation.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    hp = exp.train_branch("raw", RAW_DATA / "train_seq.npz", RAW_DATA / "validation_seq.npz", seed, out_dir,
                          architecture=arch, architecture_hparams=ARCH_HPARAMS[arch])
    ev = exp.evaluate(out_dir, RAW_DATA / "test_seq.npz")
    result = {"architecture": arch, "branch": "raw", "seed": seed, "training": hp, "evaluation": ev,
              "seconds": time.perf_counter() - t0}
    result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 4 by default: 8 PyTorch processes ran this 7.7 GB machine out of memory (CLAUDE.md).
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    from training import v2_compression_experiment as exp
    for split in ("train", "validation", "test"):
        if not (RAW_DATA / f"{split}_seq.npz").exists():
            raise FileNotFoundError(f"{RAW_DATA / f'{split}_seq.npz'}: run the Phase 2 selection first")
    for arch, hp in ARCH_HPARAMS.items():  # refuse to train anything but the pre-registered sizes
        model = exp.build_temporal_model({"architecture": arch, "latent_dim": exp.STATE_DIM,
                                          "action_dim": exp.ACTION_DIM,
                                          "sequence_length": exp.LSTM_PROTOCOL.sequence_length, **hp})
        n = sum(p.numel() for p in model.parameters())
        if n != EXPECTED_PARAMS[arch]:
            raise RuntimeError(f"{arch}: {n} parameters, pre-registered {EXPECTED_PARAMS[arch]}")
    # Transformers first: the larger processes, better while memory is freshest.
    jobs = [(arch, s) for arch in ("transformer", "lstm") for s in SEEDS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.map(_job, *zip(*jobs)), 1):
            print(f"{i}/{len(jobs)} done: {r['architecture']} seed {r['seed']}, "
                  f"GM reward_mse {r['evaluation']['gm_reward_mse']:.1f} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
