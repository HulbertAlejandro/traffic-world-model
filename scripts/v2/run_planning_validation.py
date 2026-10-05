"""Paso 4 of the planner pre-registration: validation on 24000-24023
(docs/v2/ADDENDUM_PLANIFICACION.md, sections 3, 5, 13 and 14).

    python scripts/v2/run_planning_validation.py [--workers 4]

Jobs, each one call to scripts/v2/evaluate_control_v2.py --split validation_v21 --reference-agreement
in its own process (one PyTorch/BLAS thread):
- the 5 references (one job);
- 12 planner jobs: plan_solo and plan_ppo x LSTM and Transformer x H in {3, 5, 7}, each with the
  world-model replicas 0, 1 and 2 (section 13). plan_ppo continues with the Phase 3 dream PPO
  (section 13.1), which is what plan:ARCH,ppo loads.
Output: docs/results/v2/planning/validation/<job>.json/.csv. Resumable: a job whose .json exists is
skipped. The slowest jobs (LSTM, long H) are queued first. Refuses to start without AC power or with
2 GB or less of available memory. Never touches 25000-25047 or 23000-23029.
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
OUT_DIR = ROOT_DIR / "docs" / "results" / "v2" / "planning" / "validation"
REFERENCES = ("fijo_2_3", "min_verde_y_cambiar", "cola_mas_larga", "max_presion", "espera_mas_larga")
HORIZONS = (3, 5, 7)
REPLICAS = "0+1+2"
MIN_AVAILABLE_GB = 2.0


def jobs() -> list[tuple[str, list[str]]]:
    planner = [(f"plan_{cont}_{arch}_H{h}", [f"plan_{cont}_{arch}_H{h}=plan:{arch},{cont},{h},{REPLICAS}"])
               for arch in ("lstm", "transformer") for h in sorted(HORIZONS, reverse=True) for cont in ("ppo", "solo")]
    return planner + [("references", [f"{r}=ref:{r}" for r in REFERENCES])]


def run(name: str, policies: list[str]) -> tuple[str, int, float]:
    output = OUT_DIR / name
    if output.with_suffix(".json").exists():
        return name, 0, 0.0
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "v2" / "evaluate_control_v2.py"), "--split", "validation_v21",
           "--reference-agreement", "--output", str(output), "--overwrite"]
    for policy in policies:
        cmd += ["--policy", policy]
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"start {name} (available {psutil.virtual_memory().available / 2**30:.2f} GB)", flush=True)
    t0 = time.perf_counter()
    with (OUT_DIR / f"{name}.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return name, code, time.perf_counter() - t0


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
        for i, (name, code, seconds) in enumerate(pool.map(lambda j: run(*j), todo), 1):
            status = "skipped (done)" if seconds == 0.0 and code == 0 else f"exit {code}, {seconds:.0f} s"
            print(f"{i}/{len(todo)} {name}: {status} (total {time.perf_counter() - t0:.0f} s)", flush=True)
            if code != 0:
                failed.append(name)
    print(f"failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
