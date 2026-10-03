"""Why does the state-dependent reference lose to "switch as soon as min_green allows"
under the v1-style reward once the pulse offset is random? (Phase 1, demand recalibration.)

Re-simulates every calibration seed of the given candidates with the pulse
offset drawn from the seed, under ``cola_mas_larga`` and ``min_verde_y_cambiar``
(the Phase 0 validator's own policies), and records per control step and per
signal: the v1-style reward terms (lane waiting time and halted vehicles on the
incoming lanes), the longest individual wait of a vehicle on those lanes, the
green phase and whether it switched.

The per-step reward gap (reference - trivial) is then aligned with PATTERN time,
(t + offset) mod period, so a problem caused by the pulse borders would pile up
right after pattern times 0 and 150 whatever the offset.

    python scripts/v2/analyze_reference_failures.py --candidates it5 it6 it7
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS, make_corridor_env  # noqa: E402
from scripts.v2.corridor_demand import CANDIDATES, pattern_period, pulse_offset, write_routes  # noqa: E402
from scripts.v2.validate_corridor_demand import CALIBRATION_SEEDS, _Topology, make_policy  # noqa: E402

POLICIES = ("cola_mas_larga", "min_verde_y_cambiar")
OUT_DIR = ROOT_DIR / "docs" / "results" / "v2" / "demand_calibration"


def trace_episode(route_file: str, policy: str, seed: int) -> dict:
    env = make_corridor_env(route_file=route_file, seed=seed, additional_sumo_cmd="--no-warnings --no-step-log")
    try:
        env.reset(seed=seed)
        sumo = env.sumo
        topo = _Topology(env)
        lanes = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
        act = make_policy(policy)
        rec = {k: [] for k in ("waiting", "halted", "max_wait", "phase", "switched")}
        k, done = 0, False
        while not done:
            before = {ts: env.traffic_signals[ts].green_phase for ts in TRAFFIC_SIGNAL_IDS}
            _, _, dones, _ = env.step(act(env, topo, k))
            done = dones["__all__"]
            row = {key: [] for key in rec}
            for ts in TRAFFIC_SIGNAL_IDS:
                row["waiting"].append(sum(sumo.lane.getWaitingTime(l) for l in lanes[ts]))
                row["halted"].append(sum(sumo.lane.getLastStepHaltingNumber(l) for l in lanes[ts]))
                waits = [sumo.vehicle.getWaitingTime(v) for l in lanes[ts] for v in sumo.lane.getLastStepVehicleIDs(l)]
                row["max_wait"].append(max(waits, default=0.0))
                row["phase"].append(env.traffic_signals[ts].green_phase)
                row["switched"].append(int(env.traffic_signals[ts].green_phase != before[ts]))
            for key in rec:
                rec[key].append(row[key])
            k += 1
    finally:
        env.close()
    return {"policy": policy, "seed": seed} | {key: np.asarray(v).tolist() for key, v in rec.items()}


def _green_runs(phases: np.ndarray, phase: int) -> list[int]:
    """Lengths, in control steps, of the uninterrupted runs of one green phase."""
    runs, n = [], 0
    for p in phases:
        if p == phase:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def analyze(candidate: str, traces: list[dict], offset_zero: bool = False) -> dict:
    spec = CANDIDATES[candidate]
    period = pattern_period(spec)
    by = {(t["policy"], t["seed"]): t for t in traces}
    seeds = sorted({t["seed"] for t in traces})
    steps = len(by[(POLICIES[0], seeds[0])]["waiting"])
    bins = 10  # 30 s bins of pattern time
    gap_by_pattern = np.zeros((bins, len(TRAFFIC_SIGNAL_IDS)))
    count_by_pattern = np.zeros(bins)
    per_seed = []
    for s in seeds:
        offset = 0 if offset_zero else pulse_offset(spec, s)
        ref, triv = by[("cola_mas_larga", s)], by[("min_verde_y_cambiar", s)]
        # reward per step and signal = -(waiting + halted); gap > 0 means the reference is better
        r_ref = -(np.array(ref["waiting"]) + np.array(ref["halted"]))
        r_triv = -(np.array(triv["waiting"]) + np.array(triv["halted"]))
        gap = r_ref - r_triv                                   # (steps, 4)
        pattern_t = (np.arange(1, steps + 1) * 5 + offset) % period  # time at the end of each step
        b = (pattern_t // (period // bins)).astype(int)
        np.add.at(gap_by_pattern, b, gap)
        np.add.at(count_by_pattern, b, 1)
        max_wait_ref = np.array(ref["max_wait"])
        max_wait_triv = np.array(triv["max_wait"])
        per_seed.append({
            "seed": s, "offset": offset, "gap_total": float(gap.sum()),
            "gap_per_signal": dict(zip(TRAFFIC_SIGNAL_IDS, gap.sum(axis=0).round(1).tolist())),
            "waiting_gap_per_signal": dict(zip(TRAFFIC_SIGNAL_IDS, (
                np.array(triv["waiting"]) - np.array(ref["waiting"])).sum(axis=0).round(1).tolist())),
            "halted_gap_per_signal": dict(zip(TRAFFIC_SIGNAL_IDS, (
                np.array(triv["halted"]) - np.array(ref["halted"])).sum(axis=0).round(1).tolist())),
            "max_single_wait_ref": dict(zip(TRAFFIC_SIGNAL_IDS, max_wait_ref.max(axis=0).tolist())),
            "max_single_wait_trivial": dict(zip(TRAFFIC_SIGNAL_IDS, max_wait_triv.max(axis=0).tolist())),
        })
    green_runs = {}
    for policy in POLICIES:
        green_runs[policy] = {}
        for i, ts in enumerate(TRAFFIC_SIGNAL_IDS):
            runs = {p: [] for p in (0, 1)}
            for s in seeds:
                phases = np.array(by[(policy, s)]["phase"])[:, i]
                for p in (0, 1):
                    runs[p] += _green_runs(phases, p)
            green_runs[policy][ts] = {f"phase{p}": {"mean_steps": float(np.mean(r)) if r else 0.0,
                                                   "max_steps": int(max(r, default=0))} for p, r in runs.items()}
    return {
        "candidate": candidate, "period": period, "seeds": seeds,
        "gap_by_pattern_time": {f"{int(i * period / bins)}-{int((i + 1) * period / bins)}":
                                dict(zip(TRAFFIC_SIGNAL_IDS, (gap_by_pattern[i] / max(count_by_pattern[i], 1)).round(2).tolist()))
                                for i in range(bins)},
        "per_seed": per_seed, "green_runs": green_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--offset-zero", action="store_true",
                        help="pulse offset 0 in every episode (the Phase 0 setting), for comparison")
    args = parser.parse_args()
    seeds = list(CALIBRATION_SEEDS)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tasks = []
        for c in args.candidates:
            for s in seeds:
                offset = 0 if args.offset_zero else pulse_offset(CANDIDATES[c], s)
                rf = str(write_routes(c, Path(tmp) / f"{c}_{s}.rou.xml", offset))
                tasks += [(c, rf, p, s) for p in POLICIES]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            traces = list(pool.map(trace_episode, *zip(*[t[1:] for t in tasks])))
    for c in args.candidates:
        results[c] = analyze(c, [tr for t, tr in zip(tasks, traces) if t[0] == c], args.offset_zero)
    tag = "_offset0" if args.offset_zero else ""
    out = OUT_DIR / f"analysis_reference_failures_{'_'.join(args.candidates)}{tag}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
