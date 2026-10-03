"""Same episode as trace_it5_seed11013_B0 (cola_mas_larga), with B0's outgoing lanes,
A0's signal and the head vehicle of C0B0_0 logged, to check for spillback from A0."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from environments.four_intersections import make_corridor_env, TRAFFIC_SIGNAL_IDS  # noqa: E402
from scripts.v2.corridor_demand import CANDIDATES, pulse_offset, write_routes  # noqa: E402
from scripts.v2.validate_corridor_demand import _Topology, make_policy  # noqa: E402

SEED = 11013
offset = pulse_offset(CANDIDATES["it5"], SEED)
with tempfile.TemporaryDirectory() as tmp:
    rf = str(write_routes("it5", Path(tmp) / "ep.rou.xml", offset))
    env = make_corridor_env(route_file=rf, seed=SEED, additional_sumo_cmd="--no-warnings --no-step-log")
    env.reset(seed=SEED)
    sumo = env.sumo
    topo = _Topology(env)
    act = make_policy("cola_mas_larga")
    out_lanes = ["B0A0_0", "B0C0_0", "B0top1_0", "B0bottom1_0"]
    cap = {l: sumo.lane.getLength(l) / 7.5 for l in out_lanes}
    rows, reward, k, done = [], 0.0, 0, False
    all_lanes = {ts: list(env.traffic_signals[ts].lanes) for ts in TRAFFIC_SIGNAL_IDS}
    while not done:
        _, _, dones, _ = env.step(act(env, topo, k))
        done = dones["__all__"]
        for ts_lanes in all_lanes.values():
            reward -= sum(sumo.lane.getWaitingTime(l) + sumo.lane.getLastStepHaltingNumber(l) for l in ts_lanes)
        head = sumo.lane.getLastStepVehicleIDs("C0B0_0")
        lead = head[-1] if head else None  # last in list = closest to the junction
        rows.append({
            "step": k, "t": sumo.simulation.getTime(),
            "B0_phase": env.traffic_signals["B0"].green_phase, "A0_phase": env.traffic_signals["A0"].green_phase,
            "C0B0_halted": sumo.lane.getLastStepHaltingNumber("C0B0_0"),
            "out": {l: (sumo.lane.getLastStepVehicleNumber(l), sumo.lane.getLastStepHaltingNumber(l)) for l in out_lanes},
            "A0_from_B0_halted": sumo.lane.getLastStepHaltingNumber("B0A0_0"),
            "lead_C0B0": None if lead is None else {
                "id": lead, "speed": round(sumo.vehicle.getSpeed(lead), 2),
                "dist_to_B0": round(sumo.lane.getLength("C0B0_0") - sumo.vehicle.getLanePosition(lead), 1),
                "route_next": sumo.vehicle.getRoute(lead)[sumo.vehicle.getRouteIndex(lead) + 1]
                if sumo.vehicle.getRouteIndex(lead) + 1 < len(sumo.vehicle.getRoute(lead)) else None},
        })
        k += 1
    env.close()
print("control reward_v1_style:", reward, "(validator -3107.0)")
print("B0A0 capacity ~", round(cap["B0A0_0"], 1), "vehicles")
for r in rows[30:42]:
    print(r["step"], r["t"], "B0", r["B0_phase"], "A0", r["A0_phase"], "C0B0 halted", r["C0B0_halted"],
          "| B0A0 (veh, halted)", r["out"]["B0A0_0"], "| other outs", {k: v for k, v in r["out"].items() if k != "B0A0_0"},
          "| lead", r["lead_C0B0"])
(ROOT / "docs/results/v2/demand_calibration/trace_it5_seed11013_B0_spillback.json").write_text(
    json.dumps({"seed": SEED, "pulse_offset": offset, "policy": "cola_mas_larga", "reward_v1_style": reward,
                "rows": rows}, indent=1), encoding="utf-8")
