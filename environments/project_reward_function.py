"""Project-defined reward function for the traffic signal controller."""

from __future__ import annotations

from configs.reward import RewardConfig


class ProjectRewardFunction:
    """Custom reward aligned with the project's traffic-control objective.

    The reward combines congestion penalties, throughput gains, and phase-change penalties:

        R_t = -alpha * waiting - beta * queue + gamma * throughput - delta * changes

    The throughput term is intentionally based on arrivals completed in the last
    control interval, not on the instantaneous number of vehicles present in the
    lanes. The latter is a congestion signal and would reward accumulation rather
    than throughput.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(self, state, action, next_state, info) -> float:
        """Compute the project reward from the transition state and metadata."""
        waiting = float(info.get("waiting_total", 0.0))
        queue = float(info.get("queue_total", 0.0))
        throughput = float(info.get("throughput", 0.0))
        phase_change = float(info.get("phase_change", 0.0))

        reward = (
            -self.config.alpha * waiting
            - self.config.beta * queue
            + self.config.gamma * throughput
            - self.config.delta * phase_change
        )

        return float(reward)
