"""Step-by-step trace of ONE episode at ONE signal, for a mechanism check (Phase 1, design doc 7.9).

Re-simulates a single calibration episode (candidate, seed, pulse offset drawn from
the seed, as in the --random-offset validation) under the given Phase 0 validator
policies, and records at the given signal, at every control step:
- what the policy sees when it decides: halted vehicles, vehicle count and lane
  waiting time per incoming lane, and the phase scores of cola_mas_larga
  (halted) and espera_mas_larga (alpha * waiting + beta * halted);
- the green phase before and after the step and the requested target phase;
- the longest current stop of a vehicle on each incoming lane, and which vehicle.

As a control, the episode's v1-style reward summed over the four signals must
equal the validator's reward_v1_style for the same policy and seed.

    python scripts/v2/trace_episode_signal.py --candidate it5 --seed 11013 --signal B0 \
        --policies cola_mas_larga espera_mas_larga
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.corridor_reward import CorridorRewardConfig  # noqa: E402
from environments.four_intersections import TRAFFIC_SIGNAL_IDS, make_corridor_env  # noqa: E402
from scripts.v2.corridor_demand import CANDIDATES, pulse_offset, write_routes  # noqa: E402
from scripts.v2.validate_corridor_demand import _Topology, make_policy  # noqa: E402

OUT_DIR = ROOT_DIR / "docs" / "results" / "v2" / "demand_calibration"


def trace(route_file: str, policy: str, seed: int, signal: str) -> dict:
    config = CorridorRewardConfig()
    env = make_corridor_env(route_file=route_file, seed=seed, additional_sumo_cmd="--no-warnings --no-step-log")
    try:
        env.reset(seed=seed)
        sumo = env.sumo
        topo = _Topology(env)
        phase_lanes = topo.green_in_lanes[signal]
        lanes = [lane for p in phase_lanes for lane in p]
        all_lanes = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
        act = make_policy(policy)
        steps, reward_total, k, done = [], 0.0, 0, False
        while not done:
            seen = {lane: {"halted": sumo.lane.getLastStepHaltingNumber(lane),
                           "vehicles": sumo.lane.getLastStepVehicleNumber(lane),
                           "waiting": sumo.lane.getWaitingTime(lane)} for lane in lanes}
            scores_queue = [sum(seen[l]["halted"] for l in pl) for pl in phase_lanes]
            scores_wait = [sum(config.alpha * seen[l]["waiting"] + config.beta * seen[l]["halted"] for l in pl)
                           for pl in phase_lanes]
            phase_before = int(env.traffic_signals[signal].green_phase)
            actions = act(env, topo, k)
            target = actions[signal]
            _, _, dones, _ = env.step(actions)
            done = dones["__all__"]
            longest = {}
            for lane in lanes:
                waits = [(sumo.vehicle.getWaitingTime(v), v) for v in sumo.lane.getLastStepVehicleIDs(lane)]
                w, v = max(waits, default=(0.0, None))
                longest[lane] = {"stop_s": w, "vehicle": v}
            for ts_lanes in all_lanes.values():
                reward_total -= sum(sumo.lane.getWaitingTime(l) + sumo.lane.getLastStepHaltingNumber(l)
                                    for l in ts_lanes)
            steps.append({
                "step": k, "t_end": float(sumo.simulation.getTime()), "seen": seen,
                "score_cola": scores_queue, "score_espera": scores_wait,
                "phase_before": phase_before, "target": int(target),
                "phase_after": int(env.traffic_signals[signal].green_phase),
                "signal_reward": -sum(sumo.lane.getWaitingTime(l) + sumo.lane.getLastStepHaltingNumber(l)
                                      for l in lanes),
                "longest_stop": longest,
            })
            k += 1
    finally:
        env.close()
    return {"policy": policy, "seed": seed, "signal": signal, "phase_lanes": phase_lanes,
            "reward_v1_style_total": reward_total, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--signal", required=True, choices=TRAFFIC_SIGNAL_IDS)
    parser.add_argument("--policies", nargs="+", required=True)
    args = parser.parse_args()
    offset = pulse_offset(CANDIDATES[args.candidate], args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        rf = str(write_routes(args.candidate, Path(tmp) / "episode.rou.xml", offset))
        traces = [trace(rf, p, args.seed, args.signal) for p in args.policies]
    out = OUT_DIR / f"trace_{args.candidate}_seed{args.seed}_{args.signal}.json"
    out.write_text(json.dumps({"candidate": args.candidate, "seed": args.seed, "pulse_offset": offset,
                               "signal": args.signal, "traces": traces}, indent=1), encoding="utf-8")
    for t in traces:
        print(f"{t['policy']}: reward_v1_style total {t['reward_v1_style_total']:.1f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
