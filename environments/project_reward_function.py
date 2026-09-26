"""Project-defined reward function for the traffic signal controller."""

from __future__ import annotations

from configs.reward import RewardConfig

# info key read by the delta term, per RewardConfig.phase_penalty mode.
PHASE_PENALTY_INFO_KEY = {
    "requested_phase_1": "phase_change",
    "actual_switch": "phase_switched",
}


class ProjectRewardFunction:
    """Custom reward aligned with the project's traffic-control objective.

    The reward combines congestion penalties, throughput gains, and a phase
    penalty:

        R_t = -alpha * waiting - beta * queue + gamma * throughput - delta * phase_penalty

    What the phase penalty measures depends on RewardConfig.phase_penalty:
    ``"requested_phase_1"`` (default, the definition every trained model used)
    reads ``info["phase_change"]``, which despite its name is 1.0 when phase 1
    was REQUESTED (action == 1); ``"actual_switch"`` reads
    ``info["phase_switched"]``, 1.0 only when the green phase really changed.
    Both keys are set by TrafficEnvironment.step().

    The throughput term reads ``info["throughput"]``, which was meant to count the
    arrivals of the control interval but, because of how it is measured (see
    TrafficEnvironment.step), is only ~13% of them and behaves as noise of about
    0.2 per step against rewards averaging about -43. It is kept unchanged because
    every dataset and trained model used it; with gamma=1 its weight in the reward
    is negligible. The correct arrivals count is ``info["arrivals_total"]``,
    reported but not rewarded.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(self, state, action, next_state, info) -> float:
        """Compute the project reward from the transition state and metadata."""
        waiting = float(info.get("waiting_total", 0.0))
        queue = float(info.get("queue_total", 0.0))
        throughput = float(info.get("throughput", 0.0))

        phase_key = PHASE_PENALTY_INFO_KEY[self.config.phase_penalty]
        if self.config.phase_penalty == "actual_switch" and phase_key not in info:
            # No silent 0.0 default for the new mode: a missing key would mean
            # the caller never measured the switch, not that none happened.
            raise KeyError(f"phase_penalty='actual_switch' requires info[{phase_key!r}]")
        phase_penalty = float(info.get(phase_key, 0.0))

        reward = (
            -self.config.alpha * waiting
            - self.config.beta * queue
            + self.config.gamma * throughput
            - self.config.delta * phase_penalty
        )

        return float(reward)
