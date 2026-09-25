"""Reward-related configuration values."""

from __future__ import annotations

from dataclasses import dataclass

PHASE_PENALTY_MODES = ("requested_phase_1", "actual_switch")


@dataclass(slots=True)
class RewardConfig:
    """Project reward coefficients:

        R_t = -alpha * waiting - beta * queue + gamma * throughput - delta * phase_penalty

    ``phase_penalty`` selects what the delta term penalizes:

    - ``"requested_phase_1"`` (default): ``info["phase_change"]``, 1.0 whenever the
      action requests phase 1, whether or not the signal actually changes. This is
      the definition every collected dataset and trained model so far used, so it
      stays the default to keep all existing results reproducible.
    - ``"actual_switch"``: ``info["phase_switched"]``, 1.0 only when the green phase
      really changed during the step (in either direction). Requests blocked by
      min_green, or requests for the phase already active, are not penalized.
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.1
    phase_penalty: str = "requested_phase_1"

    def __post_init__(self) -> None:
        if self.phase_penalty not in PHASE_PENALTY_MODES:
            raise ValueError(
                f"phase_penalty must be one of {PHASE_PENALTY_MODES}, got {self.phase_penalty!r}"
            )
