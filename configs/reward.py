"""Reward-related configuration values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardConfig:
    """Project reward coefficients.

    These values are placeholders until the custom traffic-state reward is
    implemented. They preserve the existing configuration surface without adding
    functional behavior.
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.1
