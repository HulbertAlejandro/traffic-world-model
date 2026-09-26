"""
Traffic environment wrapper over sumo-rl.

This module is the single integration point between the project and SUMO.
Everything outside this class should ignore how the simulator works.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import sumo_rl

from configs.environment import EnvironmentConfig
from configs.reward import RewardConfig
from environments.custom_state_builder import CustomStateBuilder
from environments.project_action_space import ProjectActionSpace
from environments.project_reward_function import ProjectRewardFunction
from environments.traffic_state import TrafficState


class TrafficEnvironment(gym.Env):
    """Project wrapper around SUMO-RL. The rest of the project should
    interact only with this class."""

    def __init__(self, config: EnvironmentConfig | None = None,
                 reward_config: RewardConfig | None = None):
        self.config = config or EnvironmentConfig()

        self.state_builder = CustomStateBuilder()
        self._action_space = ProjectActionSpace()
        self.reward_function = ProjectRewardFunction(reward_config)

        self._env = sumo_rl.SumoEnvironment(
            net_file=str(self.config.net_file),
            route_file=str(self.config.route_file),
            use_gui=self.config.use_gui,
            single_agent=self.config.single_agent,
            num_seconds=self.config.simulation_seconds,
            sumo_seed=self.config.seed,
        )

        self._current_state = None
        self._current_state_obj = None
        self._last_reward = 0.0
        self._last_info: dict = {}
        self._last_arrived_count = 0
        self._ended_count = 0
        self._observation_space: gym.spaces.Box | None = None

    ####################################################################
    # Gym API
    ####################################################################

    def reset(self, **kwargs):
        _, info = self._env.reset(**kwargs)
        info = dict(info)
        self._last_info = info

        if self._env.sumo is not None:
            self._last_arrived_count = int(self._env.sumo.simulation.getArrivedNumber())
            self._ended_count = self._ended_vehicles()

        state_obj = self.state_builder.build_state(sumo_rl_env=self._env)
        state = state_obj.to_vector()

        self._current_state_obj = state_obj
        self._current_state = state

        return state, info

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action!r}")

        traffic_signal = self._env.traffic_signals[self._env.ts_ids[0]]
        green_before = traffic_signal.green_phase
        _, simulator_reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)

        self._last_reward = simulator_reward
        self._last_info = info

        state = self._current_state
        next_state_obj = self.state_builder.build_state(sumo_rl_env=self._env)
        next_state = next_state_obj.to_vector()

        info["raw_reward"] = float(simulator_reward)
        # Two phase measures, both always recorded; RewardConfig.phase_penalty
        # picks which one the reward penalizes:
        # - "phase_change": whether phase 1 was REQUESTED (action == 1; the action
        #   is the target green phase index, see ProjectActionSpace). Despite its
        #   name, it is NOT an actual change. Kept unchanged, and still the default
        #   penalty, because all collected data and every model trained so far
        #   (Autoencoder, LSTM, Transformer, TSMixer, both PPO controllers) use it.
        # - "phase_switched": whether the green phase ACTUALLY changed during this
        #   step, in either direction. sumo_rl applies the action once, through
        #   TrafficSignal.set_next_phase, which updates green_phase immediately when
        #   the switch is allowed (different phase and yellow_time + min_green
        #   elapsed), so comparing green_phase before and after the step is exact.
        # See PROJECT_STATUS.md, "Baseline de RL directo", point 7.
        info["phase_change"] = float(int(action == 1))
        info["phase_switched"] = float(traffic_signal.green_phase != green_before)

        # info["throughput"] -- LEGACY, kept unchanged because the reward's gamma term
        # uses it and every dataset and trained model was produced with it. It was
        # meant to be the arrivals of the control interval, but getArrivedNumber()
        # returns only the arrivals of the LAST simulated second (a step runs
        # delta_time=5 of them), so this is the difference between the arrivals of
        # two single seconds, clipped at 0: ~13% of the real arrivals, not a flow
        # measure. Do not report it as throughput.
        #
        # info["arrivals_total"] -- the correct measure: vehicles that reached their
        # destination during the WHOLE control interval of this step. SUMO counts
        # inserted and still-running vehicles cumulatively, so inserted - running is
        # the number of vehicles that have ended their trip; teleports are disabled
        # (sumo_rl's time_to_teleport=-1) and there are no collisions, so every
        # ended vehicle arrived. Not used by the reward.
        if self._env.sumo is not None:
            arrived_now = int(self._env.sumo.simulation.getArrivedNumber())
            info["throughput"] = float(max(0, arrived_now - self._last_arrived_count))
            self._last_arrived_count = arrived_now
            ended_now = self._ended_vehicles()
            info["arrivals_total"] = float(ended_now - self._ended_count)
            self._ended_count = ended_now
        else:
            info["throughput"] = 0.0
            info["arrivals_total"] = 0.0

        # Read aggregate metrics from the structured state object instead of
        # slicing the flat vector by hardcoded index ranges. This keeps the
        # reward decoupled from TrafficState's internal field order.
        info["waiting_total"] = float(next_state_obj.waiting_times.sum())
        info["queue_total"] = float(next_state_obj.queue_lengths.sum())

        reward = self.reward_function.compute(
            state=state, action=action, next_state=next_state, info=info,
        )

        self._current_state_obj = next_state_obj
        self._current_state = next_state

        return next_state, reward, terminated, truncated, info

    ####################################################################
    # API del proyecto
    ####################################################################

    def _ended_vehicles(self) -> int:
        """Vehicles that have ended their trip since the simulation started."""
        simulation = self._env.sumo.simulation
        inserted = int(simulation.getParameter("", "stats.vehicles.inserted"))
        running = int(simulation.getParameter("", "stats.vehicles.running"))
        return inserted - running

    def get_state(self):
        """Devuelve el estado actual (vector) construido mediante TraCI."""
        return self._current_state

    def get_state_object(self):
        """Devuelve el estado actual como objeto de dominio ``TrafficState``."""
        return self._current_state_obj

    def compute_project_reward(self, state, action, next_state, info):
        return self.reward_function.compute(
            state=state, action=action, next_state=next_state, info=info,
        )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def env(self):
        """Acceso explícito al entorno interno. Solo debe usarse desde
        componentes de infraestructura (por ejemplo CustomStateBuilder)."""
        return self._env

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Espacio de acciones expuesto como ``gymnasium.spaces.Discrete``,
        para compatibilidad con Stable-Baselines3 (Sección 17)."""
        return gym.spaces.Discrete(self._action_space.n)

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Espacio de observación derivado del tamaño real del estado
        propio, no del observation_space original de sumo-rl (que ya no
        corresponde a lo que este entorno devuelve)."""
        if self._observation_space is None:
            size = self.state_builder.state_size()
            if size == 0:
                # Gymnasium/SB3 inspect spaces before reset opens TraCI.
                size = TrafficState().size
            self._observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32,
            )
        return self._observation_space

    ####################################################################
    # Utils
    ####################################################################

    def close(self):
        self._env.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass