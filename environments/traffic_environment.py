"""
Traffic environment wrapper over sumo-rl.

This module is the single integration point between the project and SUMO.
Everything outside this class should ignore how the simulator works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sumo_rl

from configs.environment import EnvironmentConfig
from environments.custom_state_builder import CustomStateBuilder
from environments.project_action_space import ProjectActionSpace
from environments.project_reward_function import ProjectRewardFunction


class TrafficEnvironment:
    """
    Project wrapper around SUMO-RL.

    The rest of the project should interact only with this class.
    """

    def __init__(self, config: EnvironmentConfig | None = None):

        self.config = config or EnvironmentConfig()

        # Nuestro estado propio
        self.state_builder = CustomStateBuilder()
        self._action_space = ProjectActionSpace()

        # Recompensa del proyecto
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

        self._last_reward = 0.0
        self._last_info = {}
        self._last_arrived_count = 0

    ####################################################################
    # Gym API
    ####################################################################

    def reset(self, **kwargs):

        _, info = self._env.reset(**kwargs)

        info = dict(info)

        self._last_info = info

        if self._env.sumo is not None:
            self._last_arrived_count = int(self._env.sumo.simulation.getArrivedNumber())

        state = self.state_builder.build(self)

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
        next_state = self.state_builder.build(self)

        info["raw_reward"] = float(simulator_reward)
        info["phase_change"] = float(int(action == 1))

        # Throughput must reflect vehicles that actually left the network in the
        # last control interval. Counting how many vehicles are currently visible in
        # the lanes is a congestion metric, not a completion metric, and would
        # reward the model for accumulating backlog.
        if self._env.sumo is not None:
            arrived_now = int(self._env.sumo.simulation.getArrivedNumber())
            info["throughput"] = float(max(0, arrived_now - self._last_arrived_count))
            self._last_arrived_count = arrived_now
        else:
            info["throughput"] = 0.0

        if isinstance(next_state, np.ndarray):
            vector = np.asarray(next_state, dtype=np.float32)
            if vector.size >= 20:
                info["waiting_total"] = float(np.sum(vector[8:12]))
                info["queue_total"] = float(np.sum(vector[4:8]))

        reward = self.reward_function.compute(
            state=state,
            action=action,
            next_state=next_state,
            info=info,
        )

        self._current_state = next_state

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    ####################################################################
    # API del proyecto
    ####################################################################

    def get_state(self):
        """
        Devuelve el estado actual construido mediante TraCI.
        """
        return self._current_state

    def compute_reward(
        self,
        state,
        action,
        next_state,
        info,
    ):
        """
        Calcula la recompensa del proyecto.
        """

        return self.reward_function.compute(
            state=state,
            action=action,
            next_state=next_state,
            info=info,
        )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def env(self):
        """
        Acceso explícito al entorno interno.

        Solo debe usarse desde componentes de infraestructura
        (por ejemplo StateBuilder).
        """
        return self._env

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._env.observation_space

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