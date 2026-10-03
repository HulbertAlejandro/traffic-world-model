"""Non-learned policies for the v2 corridor, in the environment's keep/switch encoding.

Used to collect the v2 dataset (docs/v2/ADDENDUM_DATASET.md). ``fijo_2_3`` and
``cola_mas_larga`` are NOT reimplemented: they call the Phase 0 validator's
policies (scripts/v2/validate_corridor_demand.py, written in v1's target-phase
convention) and translate the result to keep/switch, so they are exactly the
policies the demand was validated with.

Each factory returns ``policy(env, step) -> np.ndarray of 4 keep/switch actions``
for a ``CorridorTrafficEnvironment`` already reset for the episode.
"""

from __future__ import annotations

import numpy as np

from environments.corridor_environment import CorridorActionSpace, CorridorTrafficEnvironment
from environments.four_intersections import TRAFFIC_SIGNAL_IDS
from scripts.v2.validate_corridor_demand import _Topology, make_policy

COLLECTION_POLICIES = ("aleatoria", "fijo_2_3", "cola_mas_larga")


def _from_validator(name: str):
    target_policy = make_policy(name)
    topo_cache: dict[int, _Topology] = {}

    def policy(env: CorridorTrafficEnvironment, step: int) -> np.ndarray:
        sumo_env = env.env
        key = id(sumo_env.sumo)  # one topology per simulation run (a reset opens a new connection)
        if key not in topo_cache:
            topo_cache.clear()
            topo_cache[key] = _Topology(sumo_env)
        targets = target_policy(sumo_env, topo_cache[key], step)
        current = env.green_phases()
        return CorridorActionSpace.from_target_phases([targets[ts] for ts in TRAFFIC_SIGNAL_IDS],
                                                      [current[ts] for ts in TRAFFIC_SIGNAL_IDS])

    return policy


def make_collection_policy(name: str, episode_seed: int):
    """Policy for one episode. The random policy draws from its own generator seeded
    with the episode seed, so every episode is reproducible (v1's collection used the
    unseeded global numpy generator, and a re-collection gave a different dataset)."""
    if name == "aleatoria":
        rng = np.random.default_rng(episode_seed)
        return lambda env, step: rng.integers(0, 2, size=len(TRAFFIC_SIGNAL_IDS)).astype(np.int64)
    if name in ("fijo_2_3", "cola_mas_larga"):
        return _from_validator(name)
    raise ValueError(f"unknown collection policy {name!r}; expected one of {COLLECTION_POLICIES}")
