"""Reward configuration for the v2 four-intersection corridor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CorridorRewardConfig:
    """Coefficients of the v2 reward, summed over the four signals:

        R_t = - sum over signals of (alpha * waiting + beta * queue)

    ``waiting`` and ``queue`` are the v1 state's ``waiting_times`` and
    ``queue_lengths`` (TraCI lane waiting time and halted vehicles) summed over
    each signal's incoming lanes, read after the step. It is exactly the
    "reward_v1_style" metric the Phase 0 validator used to calibrate the demand
    (scripts/v2/validate_corridor_demand.py), so the demand was validated
    against this reward.

    v1's other two terms are left out on purpose:
    - throughput (gamma): v1's measure is broken (counts only the last simulated
      second of each step, ~13% of arrivals; see TrafficEnvironment.step) and was
      kept in v1 only because everything was already trained with it.
    - phase penalty (delta): v1 penalized REQUESTING phase 1, a side effect of its
      target-phase action. Not used in Phase 0, so adding it would change the
      reward the demand was validated with.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError(f"alpha and beta must be non-negative, got {self.alpha}, {self.beta}")
