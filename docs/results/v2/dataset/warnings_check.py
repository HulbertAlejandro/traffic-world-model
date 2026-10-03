"""Re-runs every dataset episode (and the pilot) with SUMO's warnings written to a log file, to
count emergency-braking warnings, and checks that each episode reproduces the return recorded in
its manifest (docs/v2/ADDENDUM_DATASET.md, appendix). Nothing is written to datasets/.
"""
import json
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def run(seed: int, policy: str, log_dir: str) -> dict:
    from environments.corridor_environment import CorridorTrafficEnvironment
    from scripts.v2.collect_dataset_v2 import collect_episode

    log = Path(log_dir) / f"{seed}.log"
    env = CorridorTrafficEnvironment()
    env.env.additional_sumo_cmd = f"--no-step-log --error-log {log}"
    try:
        _, meta = collect_episode(env, seed, policy)
    finally:
        env.close()
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    braking = re.findall(r"Vehicle '([^']+)' performs emergency braking on lane '([^']+)'.*?time=(\d+(?:\.\d+)?)", text)
    other = [line for line in text.splitlines() if line.startswith("Warning") and "emergency braking" not in line]
    return {"seed": seed, "policy": policy, "return": meta["return"],
            "emergency_braking": [{"vehicle": v, "lane": l, "time": float(t)} for v, l, t in braking],
            "other_warnings": other}


def main() -> None:
    episodes = []
    for name in ("manifest.json", "pilot_manifest.json"):
        for ep in json.loads((HERE / name).read_text(encoding="utf-8"))["episodes"]:
            episodes.append((ep["seed"], ep["policy"], ep["return"], ep["split"]))
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(run, [e[0] for e in episodes], [e[1] for e in episodes], [tmp] * len(episodes)))
    mismatches = [(r["seed"], r["return"], e[2]) for r, e in zip(results, episodes) if r["return"] != e[2]]
    with_braking = [r for r in results if r["emergency_braking"]]
    summary = {
        "episodes": len(results), "return_mismatches": mismatches,
        "episodes_with_emergency_braking": len(with_braking),
        "emergency_braking_events": sum(len(r["emergency_braking"]) for r in results),
        "by_policy": dict(Counter(r["policy"] for r in with_braking)),
        "by_signal": dict(Counter(b["lane"].split("_")[0].lstrip(":") for r in results for b in r["emergency_braking"])),
        "other_warning_episodes": sum(1 for r in results if r["other_warnings"]),
        "details": [{"seed": r["seed"], "policy": r["policy"], "events": r["emergency_braking"]} for r in with_braking],
        "other_warnings_sample": [w for r in results for w in r["other_warnings"]][:10],
    }
    (HERE / "warnings_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=1))
    for d in summary["details"]:
        print(d)


if __name__ == "__main__":
    main()
