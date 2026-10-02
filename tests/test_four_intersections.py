"""Smoke test of the v2 four-intersection corridor (SUMO in the loop).

Only the simulator side: the network loads with its four signals, a full
episode runs under a fixed alternating program, and the environment closes.
The v2 state, joint action and reward do not exist yet (Phase 1).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.four_intersections import (
    DELTA_TIME,
    NET_FILE,
    ROUTE_FILE,
    SIMULATION_SECONDS,
    TRAFFIC_SIGNAL_IDS,
    make_corridor_env,
)


def test_corridor_files_exist():
    assert NET_FILE.is_file()
    assert ROUTE_FILE.is_file()


def test_corridor_runs_full_episode_and_closes():
    env = make_corridor_env(seed=0, additional_sumo_cmd="--no-step-log")
    try:
        env.reset(seed=0)
        assert tuple(env.ts_ids) == TRAFFIC_SIGNAL_IDS
        for ts_id in TRAFFIC_SIGNAL_IDS:
            ts = env.traffic_signals[ts_id]
            # Two green phases per signal, as in v1: 0 = cross streets, 1 = arterial.
            assert ts.num_green_phases == 2
            assert len(ts.lanes) == 4

        steps, switches, done = 0, 0, False
        while not done:
            before = {ts: env.traffic_signals[ts].green_phase for ts in TRAFFIC_SIGNAL_IDS}
            actions = {ts: (steps // 4) % 2 for ts in TRAFFIC_SIGNAL_IDS}
            _, _, dones, _ = env.step(actions)
            switches += sum(env.traffic_signals[ts].green_phase != before[ts] for ts in TRAFFIC_SIGNAL_IDS)
            done = dones["__all__"]
            steps += 1

        assert steps == SIMULATION_SECONDS // DELTA_TIME
        assert env.sim_step == SIMULATION_SECONDS
        assert switches > 0  # the actions reach the four signals
        simulation = env.sumo.simulation
        assert int(simulation.getParameter("", "stats.vehicles.inserted")) > 0
        arrived = int(simulation.getParameter("", "stats.vehicles.inserted")) - int(
            simulation.getParameter("", "stats.vehicles.running"))
        assert arrived > 0  # vehicles cross the corridor and leave
    finally:
        env.close()
