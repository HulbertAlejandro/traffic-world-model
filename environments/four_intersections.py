"""
SUMO infrastructure for v2: a corridor of four signalized intersections.

Only the simulator side lives here (network/demand paths and the sumo_rl
parameters, kept identical to v1). The v2 state vector, joint action and reward
are NOT defined yet: that is Phase 1, after the network and demand are
validated (docs/v2/DISENO_RED_4_INTERSECCIONES.md). Until then, callers drive
the returned ``sumo_rl.SumoEnvironment`` directly in multi-agent mode
(``step({ts_id: green_phase_index, ...})``).
"""

from __future__ import annotations

from pathlib import Path

import sumo_rl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRIDOR_DIR = PROJECT_ROOT / "environments" / "four-intersection-corridor"
NET_FILE = CORRIDOR_DIR / "four-intersection-corridor.net.xml"
ROUTE_FILE = CORRIDOR_DIR / "four-intersection-corridor.rou.xml"

# West to east. Green phase 0 serves the cross streets (North/South), green
# phase 1 the arterial (East/West) -- same phase order as v1, where phase 0 is
# North/South.
TRAFFIC_SIGNAL_IDS = ("A0", "B0", "C0", "D0")

# Same values v1 runs with (the sumo_rl defaults TrafficEnvironment relies on),
# written out so a sumo_rl upgrade cannot change them silently.
DELTA_TIME = 5
YELLOW_TIME = 2
MIN_GREEN = 5
MAX_GREEN = 50
SIMULATION_SECONDS = 300


def make_corridor_env(
    route_file: str | Path = ROUTE_FILE,
    seed: int = 42,
    simulation_seconds: int = SIMULATION_SECONDS,
    use_gui: bool = False,
    additional_sumo_cmd: str | None = None,
) -> sumo_rl.SumoEnvironment:
    """Multi-agent sumo_rl environment over the four-intersection corridor."""
    return sumo_rl.SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(route_file),
        use_gui=use_gui,
        single_agent=False,
        num_seconds=simulation_seconds,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        sumo_seed=seed,
        additional_sumo_cmd=additional_sumo_cmd,
    )
