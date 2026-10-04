"""Stage 2 of the v2 control phase: train the 20 dream controllers (docs/v2/ADDENDUM_CONTROL.md, section 11).

    python scripts/v2/run_control_stage2.py [--workers 4]

dream_lstm and dream_transformer, seeds 0-9, each through training/train_controller_v2.py with its
defaults (ControllerConfig, 50,000 imagined steps, selection on validation 21000-21004) and the
world model of its own seed. Interleaved by seed (lstm s0, transformer s0, lstm s1, ...) so both
arms advance together. Each run is a separate process writing its log to <output>/train.log.

Resumable: a run whose run_info.json exists is skipped; a folder left half-written by an
interrupted run is retrained from scratch with the same seed (--overwrite). Refuses to start with
2 GB or less of available memory (section 7). Child processes use one PyTorch / BLAS thread, so 4
of them do not oversubscribe the CPU (not a hyperparameter: it does not change what is trained).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parents[2]
CONTROL_DIR = ROOT_DIR / "models" / "checkpoints" / "v2" / "control"
ARMS = ("dream_lstm", "dream_transformer")
SEEDS = tuple(range(10))
MIN_AVAILABLE_GB = 2.0


def available_gb() -> float:
    return psutil.virtual_memory().available / 2**30


def run(arm: str, seed: int) -> tuple[str, int, int, float]:
    out_dir = CONTROL_DIR / f"{arm}_s{seed}"
    if (out_dir / "run_info.json").exists():
        return arm, seed, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT_DIR / "training" / "train_controller_v2.py"), "--arm", arm, "--seed", str(seed),
           "--output-dir", str(out_dir), "--overwrite"]
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"start {arm} s{seed} (available {available_gb():.2f} GB)", flush=True)
    t0 = time.perf_counter()
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return arm, seed, code, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)  # CLAUDE.md: 4 by default on this 7.7 GB machine
    args = parser.parse_args()
    if available_gb() <= MIN_AVAILABLE_GB:
        raise SystemExit(f"only {available_gb():.2f} GB available (needs more than {MIN_AVAILABLE_GB} GB): not launching")
    jobs = [(arm, seed) for seed in SEEDS for arm in ARMS]
    t0 = time.perf_counter()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (arm, seed, code, seconds) in enumerate(pool.map(lambda j: run(*j), jobs), 1):
            status = "skipped (done)" if seconds == 0.0 and code == 0 else f"exit {code}, {seconds:.0f} s"
            print(f"{i}/{len(jobs)} {arm} s{seed}: {status} (total {time.perf_counter() - t0:.0f} s, "
                  f"available {available_gb():.2f} GB)", flush=True)
            if code != 0:
                failed.append(f"{arm}_s{seed}")
    print(f"failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
