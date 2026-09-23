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
from environments.custom_state_builder import CustomStateBuilder
from environments.project_action_space import ProjectActionSpace
from environments.project_reward_function import ProjectRewardFunction
from environments.traffic_state import TrafficState


class TrafficEnvironment(gym.Env):
    """Project wrapper around SUMO-RL. The rest of the project should
    interact only with this class."""

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()

        self.state_builder = CustomStateBuilder()
        self._action_space = ProjectActionSpace()
        self.reward_function = ProjectRewardFunction()

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

        state_obj = self.state_builder.build_state(sumo_rl_env=self._env)
        state = state_obj.to_vector()

        self._current_state_obj = state_obj
        self._current_state = state

        return state, info

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action!r}")

        _, simulator_reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)

        self._last_reward = simulator_reward
        self._last_info = info

        state = self._current_state
        next_state_obj = self.state_builder.build_state(sumo_rl_env=self._env)
        next_state = next_state_obj.to_vector()

        info["raw_reward"] = float(simulator_reward)
        info["phase_change"] = float(int(action == 1))

        # Throughput = vehicles that actually left the network in this control
        # interval (arrivals), not vehicles currently sitting in a lane. The
        # latter is a congestion signal and would reward accumulation.
        if self._env.sumo is not None:
            arrived_now = int(self._env.sumo.simulation.getArrivedNumber())
            info["throughput"] = float(max(0, arrived_now - self._last_arrived_count))
            self._last_arrived_count = arrived_now
        else:
            info["throughput"] = 0.0

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