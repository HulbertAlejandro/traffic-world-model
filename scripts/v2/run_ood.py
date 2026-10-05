"""The final confirmation on OOD 23000-23029, once (docs/v2/ADDENDUM_OOD.md).

    python scripts/v2/run_ood.py [--workers 4]

Exactly the policies of the addendum, section 1: plan_ppo_transformer and plan_ppo_lstm (H = 3,
replicas s0-s9, continuation with the Phase 3 dream PPO), the 20 Phase 3 dream PPO, the 20 direct-RL
controllers (10k and 30k) and the 5 rules. Each learned policy is split in two jobs of 5 seeds
(150 episodes); 13 jobs. Every job: evaluate_control_v2.py --split ood --confirm-held-out
--reference-agreement in its own process, one PyTorch/BLAS thread (run_planning_validation.run).
Output: docs/results/v2/ood/. Resumable: a job whose .json exists is skipped. Refuses to start
without AC power or with 2 GB or less available.
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

from scripts.v2.run_planning_test import CONTROL, H, HALVES  # noqa: E402
from scripts.v2.run_planning_validation import MIN_AVAILABLE_GB, REFERENCES, run  # noqa: E402

OUT_DIR = ROOT_DIR / "docs" / "results" / "v2" / "ood"


def jobs() -> list[tuple[str, list[str]]]:
    out = []
    for arch in ("lstm", "transformer"):
        for k, half in enumerate(HALVES):
            replicas = "+".join(str(s) for s in half)
            out.append((f"plan_ppo_{arch}_part{k}", [f"plan_ppo_{arch}=plan:{arch},ppo,{H},{replicas}"]))
    for arch in ("lstm", "transformer"):
        for k, half in enumerate(HALVES):
            out.append((f"sueno_{arch}_part{k}",
                        [f"sueno_{arch}=dream:" + ",".join(f"{CONTROL}/dream_{arch}_s{s}" for s in half)]))
    for budget in ("10k", "30k"):
        for k, half in enumerate(HALVES):
            out.append((f"directo_{budget}_part{k}",
                        [f"directo_{budget}=direct:" + ",".join(f"{CONTROL}/direct_{budget}_s{s}" for s in half)]))
    return out + [("references", [f"{r}=ref:{r}" for r in REFERENCES])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)  # CLAUDE.md: 4 by default on this 7.7 GB machine
    args = parser.parse_args()
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        raise SystemExit("not on AC power: not launching")
    if psutil.virtual_memory().available / 2**30 <= MIN_AVAILABLE_GB:
        raise SystemExit("2 GB or less available: not launching")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    t0 = time.perf_counter()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, code, seconds) in enumerate(
                pool.map(lambda j: run(*j, split="ood", out_dir=OUT_DIR, extra=("--confirm-held-out",)), todo), 1):
            status = "skipped (done)" if seconds == 0.0 and code == 0 else f"exit {code}, {seconds:.0f} s"
            print(f"{i}/{len(todo)} {name}: {status} (total {time.perf_counter() - t0:.0f} s)", flush=True)
            if code != 0:
                failed.append(name)
    print(f"failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
