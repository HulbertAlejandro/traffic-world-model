"""Traffic environment wrapper over sumo-rl.

This module is the single integration point between the project and SUMO. The
rest of the codebase should depend only on this abstraction.
"""

from __future__ import annotations

from typing import Any, Optional

import sumo_rl

from configs.environment import EnvironmentConfig
from environments.default_reward_function import DefaultRewardFunction
from environments.default_state_builder import DefaultStateBuilder


class TrafficEnvironment:
    """Project-level simulation environment built on top of sumo-rl."""

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()

        self.state_builder = DefaultStateBuilder()
        self.reward_function = DefaultRewardFunction()

        self._env = sumo_rl.SumoEnvironment(
            net_file=str(self.config.net_file),
            route_file=str(self.config.route_file),
            use_gui=self.config.use_gui,
            single_agent=self.config.single_agent,
            num_seconds=self.config.simulation_seconds,
            sumo_seed=self.config.seed,
        )

        self._current_state = None
        self._last_observation = None
        self._last_reward = 0.0
        self._last_info: dict[str, Any] = {}

    def reset(self, **kwargs):
        """Reset the simulator and return the project state."""
        observation, info = self._env.reset(**kwargs)
        info = dict(info)

        self._last_observation = observation
        self._last_info = info

        state = self.state_builder.build(observation, self)
        self._current_state = state
        return state, info

    def step(self, action):
        """Run a simulation step and compute the project reward from state data."""
        observation, simulator_reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)

        self._last_observation = observation
        self._last_reward = simulator_reward
        self._last_info = info

        state = self._current_state
        next_state = self.state_builder.build(observation, self)
        info["raw_reward"] = float(simulator_reward)

        reward = self.reward_function.compute(
            state=state,
            action=action,
            next_state=next_state,
            info=info,
        )

        self._current_state = next_state
        return next_state, reward, terminated, truncated, info

    def get_state(self, observation=None):
        """Return the current project state.

        This method is the extension point for the future custom traffic state.
        """
        if observation is None:
            return self._current_state
        return self.state_builder.build(observation, self)

    def compute_reward(self, state, action, next_state, info):
        """Compute the project reward using the project-level transition API."""
        return self.reward_function.compute(
            state=state,
            action=action,
            next_state=next_state,
            info=info,
        )

    @property
    def env(self):
        return self._env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def close(self):
        """Close the simulator."""
        self._env.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._env, item)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass