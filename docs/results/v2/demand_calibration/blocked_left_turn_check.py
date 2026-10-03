"""Does the B0 mechanism of design doc 7.10 also appear at C0? (informative check, 7.11)

Re-simulates the three it5 episodes (random pulse offset) where cola_mas_larga loses most at C0
(seeds 11013, 11011, 11016; analysis_reference_failures_it5_it6_it7.json), under cola_mas_larga
and min_verde_y_cambiar, and measures second by second at every signal:

- internal_stop_veh_s: vehicle-seconds of vehicles stopped (speed < 0.1 m/s) on the signal's
  internal (junction) lanes, i.e. turning vehicles waiting inside the intersection;
- blocked_green_veh_s: vehicle-seconds of vehicles halted on an approach lane whose head vehicle's
  movement currently shows green (G or g) while the approach's leader is on an internal lane
  stopped: vehicles that have green and cannot move because a turning vehicle ahead is waiting
  inside the junction (the B0 event of 7.10).
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from environments.four_intersections import TRAFFIC_SIGNAL_IDS, make_corridor_env  # noqa: E402
from scripts.v2.corridor_demand import CANDIDATES, pulse_offset, write_routes  # noqa: E402
from scripts.v2.validate_corridor_demand import _Topology, make_policy  # noqa: E402

SEEDS = (11013, 11011, 11016)
POLICIES = ("cola_mas_larga", "min_verde_y_cambiar")


def run(seed: int, policy: str, route_file: str) -> dict:
    env = make_corridor_env(route_file=route_file, seed=seed, additional_sumo_cmd="--no-warnings --no-step-log")
    env.reset(seed=seed)
    sumo = env.sumo
    topo = _Topology(env)
    internal = {ts: [l for l in sumo.lane.getIDList() if l.startswith(f":{ts}_")] for ts in TRAFFIC_SIGNAL_IDS}
    approaches = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
    acc = {ts: {"internal_stop_veh_s": 0, "blocked_green_veh_s": 0, "longest_internal_stop_s": 0}
           for ts in TRAFFIC_SIGNAL_IDS}
    stopped_since: dict[str, float] = {}
    original = env._sumo_step

    def step():
        original()
        t = sumo.simulation.getTime()
        for ts in TRAFFIC_SIGNAL_IDS:
            stopped_inside = set()
            for lane in internal[ts]:
                for v in sumo.lane.getLastStepVehicleIDs(lane):
                    if sumo.vehicle.getSpeed(v) < 0.1:
                        stopped_inside.add(v)
                        stopped_since.setdefault(v, t)
                        acc[ts]["longest_internal_stop_s"] = max(acc[ts]["longest_internal_stop_s"],
                                                                 t - stopped_since[v] + 1)
                    else:
                        stopped_since.pop(v, None)
            acc[ts]["internal_stop_veh_s"] += len(stopped_inside)
            for lane in approaches[ts]:
                vehs = sumo.lane.getLastStepVehicleIDs(lane)
                if not vehs:
                    continue
                head = vehs[-1]
                tls = sumo.vehicle.getNextTLS(head)
                leader = sumo.vehicle.getLeader(head, 30)
                if tls and tls[0][3] in "Gg" and leader and leader[0] in stopped_inside:
                    acc[ts]["blocked_green_veh_s"] += sumo.lane.getLastStepHaltingNumber(lane)

    env._sumo_step = step
    act = make_policy(policy)
    k, done, reward = 0, False, 0.0
    while not done:
        _, _, dones, _ = env.step(act(env, topo, k))
        done = dones["__all__"]
        for ts in TRAFFIC_SIGNAL_IDS:
            reward -= sum(sumo.lane.getWaitingTime(l) + sumo.lane.getLastStepHaltingNumber(l) for l in approaches[ts])
        k += 1
    env.close()
    return {"seed": seed, "policy": policy, "reward_v1_style": reward, "per_signal": acc}


results = []
with tempfile.TemporaryDirectory() as tmp:
    for seed in SEEDS:
        rf = str(write_routes("it5", Path(tmp) / f"{seed}.rou.xml", pulse_offset(CANDIDATES["it5"], seed)))
        for policy in POLICIES:
            results.append(run(seed, policy, rf))
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(results, indent=1), encoding="utf-8")
for r in results:
    print(r["seed"], f"{r['policy']:20s}", round(r["reward_v1_style"]),
          {ts: (v["internal_stop_veh_s"], v["blocked_green_veh_s"], v["longest_internal_stop_s"])
           for ts, v in r["per_signal"].items()})
print("->", out)
