"""Why is the head vehicle of C0B0_0 stopped with green at B0 (it5, seed 11013, offset 29,
cola_mas_larga, t = 160-200 s)? Design doc section 7.10. Same single episode as
trace_it5_seed11013_B0*.json, re-simulated with SECOND-by-second logging between
t = 150 and t = 210 of:
- B0's internal (junction) lanes: every vehicle on them, with position and speed;
- every vehicle on C0B0_0 and A0B0_0 (the two arterial approaches) and on the cross approaches;
- for the vehicles at the head of C0B0_0: next traffic light link state, leader,
  junction foes (vehicle.getJunctionFoes) and speed;
- the static link table of each B0 approach lane (lane.getLinks): which movements
  the single lane carries, whether each has priority and which internal lane it uses.
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

SEED, T0, T1 = 11013, 150, 210
APPROACHES = ["C0B0_0", "A0B0_0", "top1B0_0", "bottom1B0_0"]
offset = pulse_offset(CANDIDATES["it5"], SEED)

with tempfile.TemporaryDirectory() as tmp:
    rf = str(write_routes("it5", Path(tmp) / "ep.rou.xml", offset))
    env = make_corridor_env(route_file=rf, seed=SEED, additional_sumo_cmd="--no-warnings --no-step-log")
    env.reset(seed=SEED)
    sumo = env.sumo
    topo = _Topology(env)
    act = make_policy("cola_mas_larga")
    internal = sorted(l for l in sumo.lane.getIDList() if l.startswith(":B0_"))
    links = {lane: [{"to": l[0], "via": l[4], "has_prio": bool(l[1]), "is_open": bool(l[2]), "has_foe": bool(l[3]),
                     "state": l[5], "direction": l[6], "length": l[7]} for l in sumo.lane.getLinks(lane, True)]
             for lane in APPROACHES}
    tls_links = sumo.trafficlight.getControlledLinks("B0")
    seconds = []
    original_step = env._sumo_step

    def vehicle_info(v):
        info = {"id": v, "lane": sumo.vehicle.getLaneID(v), "pos": round(sumo.vehicle.getLanePosition(v), 2),
                "speed": round(sumo.vehicle.getSpeed(v), 2), "wait": sumo.vehicle.getWaitingTime(v),
                "route": sumo.vehicle.getRoute(v)}
        return info

    def logged_step():
        original_step()
        t = sumo.simulation.getTime()
        if not (T0 <= t <= T1):
            return
        row = {"t": t, "B0_state": sumo.trafficlight.getRedYellowGreenState("B0"),
               "internal": {l: [vehicle_info(v) for v in sumo.lane.getLastStepVehicleIDs(l)] for l in internal
                            if sumo.lane.getLastStepVehicleNumber(l)},
               "approaches": {l: [vehicle_info(v) for v in sumo.lane.getLastStepVehicleIDs(l)] for l in APPROACHES}}
        head_ids = list(sumo.lane.getLastStepVehicleIDs("C0B0_0"))[-2:]  # last = closest to B0
        heads = []
        for v in head_ids:
            h = vehicle_info(v)
            h["next_tls"] = [list(x) for x in sumo.vehicle.getNextTLS(v)][:1]
            leader = sumo.vehicle.getLeader(v, 50)
            h["leader"] = None if leader is None else [leader[0], round(leader[1], 2)]
            try:
                h["junction_foes"] = [list(f) for f in sumo.vehicle.getJunctionFoes(v, 20)]
            except Exception as exc:  # noqa: BLE001 - record whatever TraCI says
                h["junction_foes"] = f"unavailable: {exc}"
            heads.append(h)
        row["C0B0_heads"] = heads
        seconds.append(row)

    env._sumo_step = logged_step
    k, done, reward = 0, False, 0.0
    all_lanes = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
    while not done:
        _, _, dones, _ = env.step(act(env, topo, k))
        done = dones["__all__"]
        for ts_lanes in all_lanes.values():
            reward -= sum(sumo.lane.getWaitingTime(l) + sumo.lane.getLastStepHaltingNumber(l) for l in ts_lanes)
        k += 1
    env.close()

out = Path(__file__).with_name("junction_check_seed11013.json")
out.write_text(json.dumps({"seed": SEED, "pulse_offset": offset, "policy": "cola_mas_larga",
                           "reward_v1_style": reward, "approach_links": links,
                           "tls_controlled_links": [[list(x) for x in l] for l in tls_links],
                           "internal_lanes": internal, "seconds": seconds}, indent=1), encoding="utf-8")
print("control reward_v1_style:", reward, "(validator -3107.0)")
print("->", out)
