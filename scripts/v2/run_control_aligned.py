"""v2.1-0: retrain the 20 dream PPO in the corrected dream (docs/v2/ADDENDUM_SUENO_CORREGIDO.md).

    python scripts/v2/run_control_aligned.py [--workers 4]

dream_lstm and dream_transformer, seeds 0-9, with --window-alignment aligned and everything else as
in Phase 3 (train_controller_v2.py defaults: ControllerConfig, the same-seed world model, selection
on 21000-21004). Output: models/checkpoints/v2/control_aligned/dream_<arch>_s<i>/; the Phase 3
folder is never written. Same launcher as stage 2 (run_control_stage2.run): one process per run,
one PyTorch/BLAS thread, resumable. Refuses to start without AC power or with 2 GB or less available.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.v2.run_control_stage2 import ARMS, MIN_AVAILABLE_GB, SEEDS, available_gb, run  # noqa: E402

ALIGNED_DIR = ROOT_DIR / "models" / "checkpoints" / "v2" / "control_aligned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)  # CLAUDE.md: 4 by default on this 7.7 GB machine
    args = parser.parse_args()
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        raise SystemExit("not on AC power: not launching")
    if available_gb() <= MIN_AVAILABLE_GB:
        raise SystemExit(f"only {available_gb():.2f} GB available (needs more than {MIN_AVAILABLE_GB} GB): not launching")
    jobs = [(arm, seed, None, ("--window-alignment", "aligned"), ALIGNED_DIR) for seed in SEEDS for arm in ARMS]
    t0 = time.perf_counter()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, seed, code, seconds) in enumerate(pool.map(lambda j: run(*j), jobs), 1):
            status = "skipped (done)" if seconds == 0.0 and code == 0 else f"exit {code}, {seconds:.0f} s"
            print(f"{i}/{len(jobs)} {name} s{seed}: {status} (total {time.perf_counter() - t0:.0f} s, "
                  f"available {available_gb():.2f} GB)", flush=True)
            if code != 0:
                failed.append(f"{name}_s{seed}")
    print(f"failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
