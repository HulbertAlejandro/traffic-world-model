"""Validate a v2 corridor demand candidate against trivial control policies.

Runs every policy below on the same calibration seeds (paired by seed) and
writes per-episode results to docs/results/v2/demand_calibration/<candidate>_<seconds>s.json.

Policies, applied to the four signals through sumo_rl (delta_time=5,
yellow_time=2, min_green=5, the v1 values):

Trivial (what the demand must NOT make optimal):
- ``min_verde_y_cambiar``: request the other green phase every step, i.e.
  switch as soon as min_green allows (the rule v1's symmetric demand made optimal).
- ``fijo_N0_N1``: fixed time, N0 steps requesting phase 0 (cross streets), then
  N1 steps requesting phase 1 (arterial), same program and offset 0 at the four
  signals. A grid of (N0, N1) is run; its best cell on these same seeds is the
  "best uniform fixed time" (selected in-sample, so the comparison favors it).
- ``fijo_v1``: v1's fixed-time baseline as is (request phase 1 once every 5 steps).
- ``siempre_arterial``: never leave the arterial green (sanity check; starves
  the cross streets).

State-dependent references (no training; they only show whether a policy that
looks at the traffic beats every trivial one):
- ``max_presion``: max-pressure, per signal: the green phase with the largest
  sum over its links of (vehicles on the incoming lane - vehicles on the outgoing lane).
- ``cola_mas_larga``: the green phase whose incoming lanes have more halted vehicles.

Metrics (none is the v2 reward, which is Phase 1):
- ``delay``: vehicle-seconds stopped, summed every simulated second over every
  lane of the network PLUS the vehicles waiting to be inserted (queued outside
  the network, invisible to the lanes -- the v1 700/150 problem).
- ``delay_per_signal``: the same, on each signal's incoming lanes only.
- ``reward_v1_style``: sum over control steps and signals of
  -(waiting time + halted vehicles) on the signal's incoming lanes, the
  alpha/beta terms of v1's reward (throughput and phase-penalty terms omitted).
- ``arrived``, ``inserted``, ``max_pending``, ``switches``.

    python scripts/v2/validate_corridor_demand.py --candidate it1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import TRAFFIC_SIGNAL_IDS, make_corridor_env  # noqa: E402
from scripts.evaluate_multiseed_statistical import paired_t  # noqa: E402
from scripts.v2.corridor_demand import CANDIDATES, total_demand, write_routes  # noqa: E402

# The project's paired Wilcoxon (exact without ties; validated against brute force there).
_spec = importlib.util.spec_from_file_location(
    "ppo_10_seeds_analyze", ROOT_DIR / "docs" / "results" / "ppo_10_seeds" / "analyze.py")
_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyze)
wilcoxon = _analyze.wilcoxon

# Calibration seeds. Disjoint from v1's 3000-3014 / 5000-5014 / 7000-7029, and
# must stay disjoint from any v2 evaluation seed: the demand is tuned on these.
CALIBRATION_SEEDS = list(range(11000, 11020))
FIXED_GRID = [2, 3, 4, 6, 8, 10]
RESULTS_DIR = ROOT_DIR / "docs" / "results" / "v2" / "demand_calibration"

# Acceptance criteria, fixed before running any candidate (see the design doc).
MIN_REDUCTION = 0.10   # best reference vs EVERY trivial policy, mean paired reduction of delay
MAX_P_VALUE = 0.01     # paired t-test and Wilcoxon, on delay and on reward_v1_style
MAX_PENDING_BEST = 5   # vehicles waiting outside the network under the best reference, after WARMUP_SECONDS
# Every flow inserts its first vehicle at t=0 (26 flows on 10 single-lane entries), so
# the first seconds always show a backlog that has nothing to do with control.
WARMUP_SECONDS = 30


def policy_names() -> list[str]:
    names = ["min_verde_y_cambiar", "fijo_v1", "siempre_arterial", "max_presion", "cola_mas_larga"]
    names += [f"fijo_{a}_{b}" for a in FIXED_GRID for b in FIXED_GRID]
    return names


def is_trivial(name: str) -> bool:
    return name not in ("max_presion", "cola_mas_larga")


class _Topology:
    """Per-signal link structure, read once per episode."""

    def __init__(self, env):
        self.links = {}
        self.green_in_lanes = {}
        for ts_id in TRAFFIC_SIGNAL_IDS:
            ts = env.traffic_signals[ts_id]
            controlled = env.sumo.trafficlight.getControlledLinks(ts_id)
            per_phase_links, per_phase_lanes = [], []
            for phase in ts.green_phases:
                links, lanes = [], []
                for i, link in enumerate(controlled):
                    if link and phase.state[i] in "Gg":
                        links.append((link[0][0], link[0][1]))
                        if link[0][0] not in lanes:
                            lanes.append(link[0][0])
                per_phase_links.append(links)
                per_phase_lanes.append(lanes)
            self.links[ts_id] = per_phase_links
            self.green_in_lanes[ts_id] = per_phase_lanes


def _choose(scores: list[float], current: int) -> int:
    best = int(np.argmax(scores))
    return current if scores[best] == scores[current] else best


def make_policy(name: str):
    if name == "min_verde_y_cambiar":
        return lambda env, topo, k: {ts: 1 - env.traffic_signals[ts].green_phase for ts in TRAFFIC_SIGNAL_IDS}
    if name == "fijo_v1":
        return lambda env, topo, k: {ts: 1 if k % 5 == 0 else 0 for ts in TRAFFIC_SIGNAL_IDS}
    if name == "siempre_arterial":
        return lambda env, topo, k: {ts: 1 for ts in TRAFFIC_SIGNAL_IDS}
    if name.startswith("fijo_"):
        n0, n1 = (int(x) for x in name.split("_")[1:])
        return lambda env, topo, k: {ts: 0 if k % (n0 + n1) < n0 else 1 for ts in TRAFFIC_SIGNAL_IDS}
    if name == "max_presion":
        def max_pressure(env, topo, k):
            count = env.sumo.lane.getLastStepVehicleNumber
            actions = {}
            for ts in TRAFFIC_SIGNAL_IDS:
                scores = [sum(count(i) - count(o) for i, o in links) for links in topo.links[ts]]
                actions[ts] = _choose(scores, env.traffic_signals[ts].green_phase)
            return actions
        return max_pressure
    if name == "cola_mas_larga":
        def longest_queue(env, topo, k):
            halted = env.sumo.lane.getLastStepHaltingNumber
            actions = {}
            for ts in TRAFFIC_SIGNAL_IDS:
                scores = [sum(halted(lane) for lane in lanes) for lanes in topo.green_in_lanes[ts]]
                actions[ts] = _choose(scores, env.traffic_signals[ts].green_phase)
            return actions
        return longest_queue
    raise ValueError(f"unknown policy {name!r}")


def run_episode(route_file: str, policy: str, seed: int, seconds: int) -> dict:
    env = make_corridor_env(route_file=route_file, seed=seed, simulation_seconds=seconds,
                            additional_sumo_cmd="--no-warnings --no-step-log")
    t0 = time.perf_counter()
    try:
        env.reset(seed=seed)
        topo = _Topology(env)
        sumo = env.sumo
        lanes = [lane for lane in sumo.lane.getIDList() if not lane.startswith(":")]
        signal_lanes = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
        acc = {"delay": 0.0, "max_pending": 0, "max_pending_after_warmup": 0,
               "per_signal": dict.fromkeys(TRAFFIC_SIGNAL_IDS, 0.0)}
        original_step = env._sumo_step

        def metered_step():
            original_step()
            halted = {lane: sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes}
            pending = len(sumo.simulation.getPendingVehicles())
            acc["delay"] += sum(halted.values()) + pending
            acc["max_pending"] = max(acc["max_pending"], pending)
            if sumo.simulation.getTime() > WARMUP_SECONDS:
                acc["max_pending_after_warmup"] = max(acc["max_pending_after_warmup"], pending)
            for ts, ts_lanes in signal_lanes.items():
                acc["per_signal"][ts] += sum(halted[lane] for lane in ts_lanes)

        env._sumo_step = metered_step
        act = make_policy(policy)
        reward_v1, switches, k, done = 0.0, 0, 0, False
        while not done:
            before = {ts: env.traffic_signals[ts].green_phase for ts in TRAFFIC_SIGNAL_IDS}
            _, _, dones, _ = env.step(act(env, topo, k))
            done = dones["__all__"]
            switches += sum(env.traffic_signals[ts].green_phase != before[ts] for ts in TRAFFIC_SIGNAL_IDS)
            for ts_lanes in signal_lanes.values():
                reward_v1 -= sum(sumo.lane.getWaitingTime(lane) + sumo.lane.getLastStepHaltingNumber(lane)
                                 for lane in ts_lanes)
            k += 1
        inserted = int(sumo.simulation.getParameter("", "stats.vehicles.inserted"))
        running = int(sumo.simulation.getParameter("", "stats.vehicles.running"))
    finally:
        env.close()
    return {
        "policy": policy, "seed": seed, "delay": acc["delay"], "delay_per_signal": acc["per_signal"],
        "reward_v1_style": reward_v1, "arrived": inserted - running, "inserted": inserted,
        "max_pending": acc["max_pending"], "max_pending_after_warmup": acc["max_pending_after_warmup"],
        "switches": switches, "steps": k,
        "wall_seconds": time.perf_counter() - t0,
    }


def _paired(ref: np.ndarray, other: np.ndarray, lower_is_better: bool) -> dict:
    """Mean paired relative improvement of ref over other, with t and Wilcoxon p-values."""
    diff = other - ref if lower_is_better else ref - other
    rel = float(diff.mean() / abs(other.mean()))
    t_p = paired_t(ref, other)["p"] if np.any(diff) else 1.0
    t_p = 0.0 if t_p is None else float(t_p)  # None only when every paired difference is identical and nonzero
    w_p = float(wilcoxon(ref, other)["p"]) if np.any(diff) else 1.0
    return {"reduction": rel, "t_p": t_p, "wilcoxon_p": w_p, "ref_wins": int((diff > 0).sum())}


def analyze(episodes: list[dict], seeds: list[int]) -> dict:
    by = {}
    for ep in episodes:
        by.setdefault(ep["policy"], {})[ep["seed"]] = ep
    arr = {p: {m: np.array([by[p][s][m] for s in seeds], dtype=float)
               for m in ("delay", "reward_v1_style", "arrived", "max_pending", "max_pending_after_warmup",
                         "switches", "wall_seconds")}
           for p in by}
    summary = {p: {m: float(v.mean()) for m, v in d.items()}
               | {"max_pending_after_warmup_max": float(d["max_pending_after_warmup"].max())}
               for p, d in arr.items()}
    trivial = [p for p in by if is_trivial(p)]
    refs = [p for p in by if not is_trivial(p)]
    best_trivial = min(trivial, key=lambda p: summary[p]["delay"])
    best_fixed = min((p for p in trivial if p.startswith("fijo_") and p != "fijo_v1"),
                     key=lambda p: summary[p]["delay"])
    best_ref = min(refs, key=lambda p: summary[p]["delay"])
    best_trivial_reward = max(trivial, key=lambda p: summary[p]["reward_v1_style"])

    comparisons = {p: {"delay": _paired(arr[best_ref]["delay"], arr[p]["delay"], True),
                       "reward_v1_style": _paired(arr[best_ref]["reward_v1_style"], arr[p]["reward_v1_style"], False)}
                   for p in trivial}
    hardest = comparisons[best_trivial]
    checks = {
        "no_invisible_queue": summary[best_ref]["max_pending_after_warmup_max"] <= MAX_PENDING_BEST,
        "beats_every_trivial_delay": all(
            c["delay"]["reduction"] >= MIN_REDUCTION and c["delay"]["t_p"] < MAX_P_VALUE
            and c["delay"]["wilcoxon_p"] < MAX_P_VALUE for c in comparisons.values()),
        "beats_every_trivial_reward_v1_style": all(
            c["reward_v1_style"]["reduction"] > 0 and c["reward_v1_style"]["t_p"] < MAX_P_VALUE
            for c in comparisons.values()),
    }
    return {
        "summary": summary, "best_trivial_by_delay": best_trivial, "best_uniform_fixed_time": best_fixed,
        "best_trivial_by_reward_v1_style": best_trivial_reward, "best_reference": best_ref,
        "best_reference_vs_best_trivial": hardest, "comparisons_vs_best_reference": comparisons,
        "checks": checks, "accepted": all(checks.values()),
    }


def _print_report(name: str, seconds: int, analysis: dict, seeds: list[int]) -> None:
    s = analysis["summary"]
    print(f"\n=== {name}, {seconds} s, {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}) ===")
    ranked = sorted(s, key=lambda p: s[p]["delay"])
    shown = [p for p in ranked if not p.startswith("fijo_") or p == "fijo_v1"
             or p == analysis["best_uniform_fixed_time"]] + [p for p in ranked[:5] if p.startswith("fijo_")]
    seen = set()
    print(f"{'policy':22s} {'delay veh-s':>12s} {'reward_v1':>11s} {'arrived':>8s} {'pend>30s':>9s} {'switches':>9s}")
    for p in ranked:
        if p in shown and p not in seen:
            seen.add(p)
            print(f"{p:22s} {s[p]['delay']:12.0f} {s[p]['reward_v1_style']:11.0f} {s[p]['arrived']:8.1f} "
                  f"{s[p]['max_pending_after_warmup_max']:9.0f} {s[p]['switches']:9.1f}")
    c = analysis["best_reference_vs_best_trivial"]
    print(f"best reference: {analysis['best_reference']}; best trivial: {analysis['best_trivial_by_delay']} "
          f"(by reward_v1: {analysis['best_trivial_by_reward_v1_style']}); best uniform fixed: "
          f"{analysis['best_uniform_fixed_time']}")
    print(f"  delay reduction {c['delay']['reduction']:+.1%} (t p={c['delay']['t_p']:.2g}, "
          f"Wilcoxon p={c['delay']['wilcoxon_p']:.2g}, wins {c['delay']['ref_wins']}/{len(seeds)}); "
          f"reward_v1 improvement {c['reward_v1_style']['reduction']:+.1%} (t p={c['reward_v1_style']['t_p']:.2g})")
    print(f"  checks: {analysis['checks']} -> accepted={analysis['accepted']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=len(CALIBRATION_SEEDS))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    seeds = CALIBRATION_SEEDS[: args.num_seeds]
    with tempfile.TemporaryDirectory() as tmp:
        route_file = str(write_routes(args.candidate, Path(tmp) / f"{args.candidate}.rou.xml"))
        tasks = [(route_file, p, s, args.seconds) for p in policy_names() for s in seeds]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            episodes = list(pool.map(run_episode, *zip(*tasks)))
        elapsed = time.perf_counter() - t0

    analysis = analyze(episodes, seeds)
    _print_report(args.candidate, args.seconds, analysis, seeds)
    print(f"{len(tasks)} episodes in {elapsed:.0f} s with {args.workers} workers")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.candidate}_{args.seconds}s.json"
    out.write_text(json.dumps({
        "candidate": args.candidate, "spec": CANDIDATES[args.candidate],
        "total_demand_veh_h": total_demand(CANDIDATES[args.candidate]), "seconds": args.seconds,
        "seeds": seeds, "criteria": {"min_reduction": MIN_REDUCTION, "max_p_value": MAX_P_VALUE,
                                     "max_pending_best": MAX_PENDING_BEST},
        "analysis": analysis, "episodes": episodes,
    }, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
