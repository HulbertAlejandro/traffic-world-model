"""Checks of the v2 dataset pre-registered in docs/v2/ADDENDUM_DATASET.md, section 5.

    python scripts/v2/check_dataset_v2.py --manifest docs/results/v2/dataset/pilot_manifest.json
    python scripts/v2/check_dataset_v2.py --manifest docs/results/v2/dataset/manifest.json

Checks 1-5 apply to every episode; 6 (reward scale against the Phase 0 validation with random
offset) and 7 (no invisible queue under cola_mas_larga) to the policy means. Also lists the
state columns that are constant across all the episodes, and verifies every file's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402

STATE_DIM, N_STEPS = 104, 60
PHASE0 = ROOT_DIR / "docs" / "results" / "v2" / "demand_calibration" / "it5_offset_300s.json"
FIELDS = ["vehicle_counts"] * 4 + ["queue_lengths"] * 4 + ["waiting_times"] * 4 + ["mean_speeds"] * 4 + \
         ["occupancies"] * 4 + ["phase_one_hot"] * 4 + ["elapsed_phase_time", "remaining_phase_time"]


def column_name(i: int) -> str:
    block, j = divmod(i, 26)
    name = FIELDS[j]
    if j < 24:
        name += f"[{j % 4}]"
    return f"{TRAFFIC_SIGNAL_IDS[block]}.{name}"


def check(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    failures, states_all = [], []
    for ep in episodes:
        path = ROOT_DIR / ep["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ep["sha256"]:
            failures.append(f"{ep['seed']}: SHA-256 does not match the manifest")
        with np.load(path) as d:
            s, a, r, ns = d["states"], d["actions"], d["rewards"], d["next_states"]
            if not (s.shape == ns.shape == (N_STEPS, STATE_DIM) and a.shape == (N_STEPS, 4) and r.shape == (N_STEPS,)):
                failures.append(f"{ep['seed']}: shapes {s.shape} {a.shape} {r.shape}")              # 1, 3
                continue
            if not (d["truncated"][-1] and not d["truncated"][:-1].any() and not d["terminated"].any()):
                failures.append(f"{ep['seed']}: terminated/truncated flags")                       # 1
            if not (np.isfinite(s).all() and np.isfinite(ns).all() and np.isfinite(r).all()):
                failures.append(f"{ep['seed']}: NaN or Inf")                                       # 2
            if not (s.dtype == ns.dtype == r.dtype == np.float32 and a.dtype == np.int64
                    and d["episode_id"].dtype == np.int64 and np.isin(a, (0, 1)).all()):
                failures.append(f"{ep['seed']}: dtypes or actions outside {{0, 1}}")              # 3
            if (r > 0).any():
                failures.append(f"{ep['seed']}: positive reward")                                 # 4
            if not np.array_equal(ns[:-1], s[1:]):
                failures.append(f"{ep['seed']}: next_states[t] != states[t+1]")                   # 5
            if not ((d["episode_id"] == ep["seed"]).all() and np.array_equal(d["time_step"], np.arange(N_STEPS))):
                failures.append(f"{ep['seed']}: episode_id / time_step")
            states_all.append(s)

    by_policy: dict[str, list[dict]] = {}
    for ep in episodes:
        by_policy.setdefault(ep["policy"], []).append(ep)
    phase0 = json.loads(PHASE0.read_text(encoding="utf-8"))["analysis"]["summary"]
    scale = {}
    for policy in ("fijo_2_3", "cola_mas_larga"):                                                  # 6
        mean = float(np.mean([e["return"] for e in by_policy.get(policy, [])]))
        ref = phase0[policy]["reward_v1_style"]
        ok = abs(mean - ref) <= 0.5 * abs(ref)
        scale[policy] = {"mean_return": mean, "phase0_mean": ref, "ok": ok}
        if not ok:
            failures.append(f"{policy}: mean return {mean:.0f} outside +-50% of Phase 0 {ref:.0f}")
    pending = max(e["max_pending_after_warmup"] for e in by_policy.get("cola_mas_larga", []))    # 7
    if pending > 5:
        failures.append(f"cola_mas_larga: {pending:.0f} vehicles pending insertion after 30 s")

    stacked = np.concatenate(states_all)
    constant = [column_name(i) for i in range(STATE_DIM) if np.ptp(stacked[:, i]) == 0]
    summary = {
        policy: {"episodes": len(eps), "return_mean": float(np.mean([e["return"] for e in eps])),
                 "return_min": float(min(e["return"] for e in eps)), "return_max": float(max(e["return"] for e in eps)),
                 "switches_per_signal_mean": np.mean([e["switches_per_signal"] for e in eps], axis=0).round(1).tolist(),
                 "max_pending_after_warmup": float(max(e["max_pending_after_warmup"] for e in eps)),
                 "seconds_mean": float(np.mean([e["seconds"] for e in eps]))}
        for policy, eps in by_policy.items()
    }
    return {"manifest": str(manifest_path.relative_to(ROOT_DIR).as_posix()), "episodes": len(episodes),
            "failures": failures, "passed": not failures, "reward_scale": scale,
            "max_pending_after_warmup_cola": pending, "constant_columns": constant, "per_policy": summary,
            "mean_episode_seconds": manifest["mean_episode_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.manifest.resolve())
    out = args.manifest.with_name(args.manifest.stem.replace("manifest", "checks") + ".json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_policy"}, indent=1))
    for policy, s in result["per_policy"].items():
        print(policy, s)
    print("->", out)


if __name__ == "__main__":
    main()
