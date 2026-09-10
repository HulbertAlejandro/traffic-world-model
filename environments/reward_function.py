"""Reward function contract and compatibility exports."""

from __future__ import annotations

from typing import Protocol

from environments.default_reward_function import DefaultRewardFunction


class RewardFunction(Protocol):
    """Contract that any project reward function must implement."""

    def compute(self, state, action, next_state, info) -> float:
        """Compute the project reward for a transition."""
        ...


__all__ = ["RewardFunction", "DefaultRewardFunction"]