"""
v2 environment: state, joint action and reward over the four-intersection corridor.

Same design as v1, applied per signal and joined:

- State (``CorridorStateBuilder``): one v1 ``CustomStateBuilder`` per signal
  (A0, B0, C0, D0, west to east), each giving the v1 26-dim ``TrafficState``
  of that signal, concatenated into a 104-dim vector. Columns of signal k sit
  at ``26*k : 26*(k+1)``, in v1's order.
- Joint action (``CorridorActionSpace``): 4 binary decisions, one per signal,
  0 = keep the current green phase, 1 = switch to the other one, applied to the
  four signals at once. NOTE: this is not v1's encoding. v1's action is the
  index of the target green phase (see ProjectActionSpace). With 2 green phases
  the two are a bijection given the current phase, which is in the state:
  target = current if keep else 1 - current. As in v1, a switch requested
  before ``yellow_time + min_green`` seconds have passed is ignored by sumo_rl.
- Reward (``CorridorRewardFunction``): the "reward_v1_style" of the Phase 0
  validator, -(waiting + queue) on each signal's incoming lanes, SUMMED over the
  four signals (configs/corridor_reward.py and docs/v2/ADDENDUM_DATASET.md).

Demand: candidate it5 with the pulse offset drawn from each episode's seed
(scripts/v2/corridor_demand.py), written to a route file private to each
environment instance, so two instances never share one.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

from configs.corridor_reward import CorridorRewardConfig
from environments.custom_state_builder import CustomStateBuilder
from environments.four_intersections import (
    ROUTE_FILE,
    SIMULATION_SECONDS,
    TRAFFIC_SIGNAL_IDS,
    make_corridor_env,
)
from environments.traffic_state import TrafficState

KEEP, SWITCH = 0, 1


class CorridorStateBuilder:
    """Joint state: v1's per-signal state for each of the four signals, concatenated."""

    def __init__(self, signal_ids=TRAFFIC_SIGNAL_IDS) -> None:
        self.signal_ids = tuple(signal_ids)
        self.builders = {ts: CustomStateBuilder(traffic_light_id=ts) for ts in self.signal_ids}

    def build_states(self, sumo_rl_env) -> dict[str, TrafficState]:
        return {ts: builder.build_state(sumo_rl_env=sumo_rl_env) for ts, builder in self.builders.items()}

    @staticmethod
    def to_vector(states: dict[str, TrafficState]) -> np.ndarray:
        return np.concatenate([state.to_vector() for state in states.values()]).astype(np.float32)

    def state_size(self) -> int:
        """Joint vector size once connected; 0 before the first build (as in v1)."""
        sizes = [builder.state_size() for builder in self.builders.values()]
        return 0 if 0 in sizes else sum(sizes)

    def signal_slice(self, signal_id: str) -> slice:
        """Columns of one signal inside the joint vector."""
        size = TrafficState().size
        k = self.signal_ids.index(signal_id)
        return slice(size * k, size * (k + 1))


class CorridorActionSpace:
    """Joint action: one keep (0) / switch (1) decision per signal."""

    def __init__(self, signal_ids=TRAFFIC_SIGNAL_IDS) -> None:
        self.signal_ids = tuple(signal_ids)
        self.n_signals = len(self.signal_ids)

    def gym_space(self) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete([2] * self.n_signals)

    def contains(self, action) -> bool:
        arr = np.asarray(action)
        return arr.shape == (self.n_signals,) and bool(np.isin(arr, (KEEP, SWITCH)).all())

    def to_target_phases(self, action, current_phases: dict[str, int]) -> dict[str, int]:
        """sumo_rl's per-signal target green phase for a keep/switch action."""
        return {ts: int(current_phases[ts]) if int(a) == KEEP else 1 - int(current_phases[ts])
                for ts, a in zip(self.signal_ids, np.asarray(action))}

    @staticmethod
    def from_target_phases(target_phases, current_phases) -> np.ndarray:
        """Inverse map, for policies written in v1's target-phase convention."""
        return np.array([int(t != c) for t, c in zip(target_phases, current_phases)], dtype=np.int64)


class CorridorRewardFunction:
    """R = -sum over signals of (alpha * waiting + beta * queue) on the incoming lanes."""

    def __init__(self, config: CorridorRewardConfig | None = None) -> None:
        self.config = config or CorridorRewardConfig()

    def per_signal(self, states: dict[str, TrafficState]) -> dict[str, float]:
        return {ts: -(self.config.alpha * float(s.waiting_times.sum()) + self.config.beta * float(s.queue_lengths.sum()))
                for ts, s in states.items()}

    def compute(self, states: dict[str, TrafficState]) -> float:
        return float(sum(self.per_signal(states).values()))


class CorridorTrafficEnvironment(gym.Env):
    """Gym environment over the v2 corridor: 104-dim state, 4 keep/switch decisions, summed reward.

    ``reset(seed=s)`` starts the episode with SUMO seed ``s`` and pulse offset
    ``pulse_offset(it5, s)``. ``reset()`` without a seed repeats the last seed,
    as sumo_rl does. ``random_pulse_offset=False`` uses the static official route
    file (offset 0), the demand exactly as validated in Phase 0.
    """

    def __init__(self, seed: int = 42, simulation_seconds: int = SIMULATION_SECONDS, use_gui: bool = False,
                 random_pulse_offset: bool = True, reward_config: CorridorRewardConfig | None = None) -> None:
        self.random_pulse_offset = random_pulse_offset
        self._seed = int(seed)
        self._route_dir = Path(tempfile.mkdtemp(prefix="corridor_routes_"))
        self._env = make_corridor_env(route_file=ROUTE_FILE, seed=self._seed, simulation_seconds=simulation_seconds,
                                      use_gui=use_gui, additional_sumo_cmd="--no-step-log")
        self.state_builder = CorridorStateBuilder()
        self._action_space = CorridorActionSpace()
        self.reward_function = CorridorRewardFunction(reward_config)
        self.pulse_offset = 0
        self._state: np.ndarray | None = None
        self._ended_count = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(TrafficState().size * len(TRAFFIC_SIGNAL_IDS),),
                                                dtype=np.float32)
        self.action_space = self._action_space.gym_space()

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = int(seed)
        if self.random_pulse_offset:
            # Imported here: the generator lives in scripts/ and imports this package's sibling module.
            from scripts.v2.corridor_demand import write_episode_routes

            route_file, self.pulse_offset = write_episode_routes(self._seed, self._route_dir / "episode.rou.xml")
            self._env._route = str(route_file)
        self._env.reset(seed=self._seed)
        self._ended_count = self._ended_vehicles()
        states = self.state_builder.build_states(self._env)
        self._state = self.state_builder.to_vector(states)
        return self._state, {"seed": self._seed, "pulse_offset": self.pulse_offset}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid joint action: {action!r}")
        before = self.green_phases()
        _, _, dones, _ = self._env.step(self._action_space.to_target_phases(action, before))
        after = self.green_phases()

        states = self.state_builder.build_states(self._env)
        next_state = self.state_builder.to_vector(states)
        per_signal = self.reward_function.per_signal(states)
        reward = float(sum(per_signal.values()))

        ended = self._ended_vehicles()
        info = {
            "waiting_total": float(sum(s.waiting_times.sum() for s in states.values())),
            "queue_total": float(sum(s.queue_lengths.sum() for s in states.values())),
            "reward_per_signal": per_signal,
            "switch_requested": action.astype(np.float32),
            "phase_switched": np.array([float(after[ts] != before[ts]) for ts in TRAFFIC_SIGNAL_IDS],
                                       dtype=np.float32),
            # Same definition as v1's info["arrivals_total"] (not the broken "throughput").
            "arrivals_total": float(ended - self._ended_count),
            "pending_vehicles": float(len(self._env.sumo.simulation.getPendingVehicles())),
            "pulse_offset": self.pulse_offset,
        }
        self._ended_count = ended
        self._state = next_state
        return next_state, reward, False, bool(dones["__all__"]), info

    def green_phases(self) -> dict[str, int]:
        return {ts: int(self._env.traffic_signals[ts].green_phase) for ts in TRAFFIC_SIGNAL_IDS}

    def _ended_vehicles(self) -> int:
        simulation = self._env.sumo.simulation
        return int(simulation.getParameter("", "stats.vehicles.inserted")) - int(
            simulation.getParameter("", "stats.vehicles.running"))

    @property
    def env(self):
        """The underlying sumo_rl environment (infrastructure use only, as in v1)."""
        return self._env

    def close(self):
        try:
            self._env.close()
        finally:
            shutil.rmtree(self._route_dir, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
