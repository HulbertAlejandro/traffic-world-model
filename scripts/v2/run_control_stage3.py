"""Stage 3 of the v2 control phase: train the 20 direct-RL controllers
(docs/v2/ADDENDUM_CONTROL.md, sections 1-3 and 12).

    python scripts/v2/run_control_stage3.py [--workers 4]

training/train_controller_v2.py --arm direct, seeds 0-9, with 10,000 and 30,000 real steps
(folders direct_10k_s<i> and direct_30k_s<i> under models/checkpoints/v2/control/). Training
scenarios 30000, 30001, ... and selection on validation 21000-21004, as fixed in the script.
The 30k runs are queued first (the longest), so the 10k runs fill the workers at the end.
Same launcher as stage 2 (run_control_stage2.run): resumable, one process per run with one
PyTorch/BLAS thread, refuses to start with 2 GB or less of available memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.v2.run_control_stage2 import MIN_AVAILABLE_GB, SEEDS, available_gb, run  # noqa: E402

BUDGETS = (30_000, 10_000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)  # CLAUDE.md: 4 by default on this 7.7 GB machine
    args = parser.parse_args()
    if available_gb() <= MIN_AVAILABLE_GB:
        raise SystemExit(f"only {available_gb():.2f} GB available (needs more than {MIN_AVAILABLE_GB} GB): not launching")
    jobs = [("direct", seed, f"direct_{budget // 1000}k", ("--total-timesteps", str(budget)))
            for budget in BUDGETS for seed in SEEDS]
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
