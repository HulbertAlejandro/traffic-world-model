"""Default reward function.

The reward interface is intentionally designed around project-level transitions
(state, action, next_state, info). The current implementation delegates to the
simulator reward as a temporary compatibility measure until the custom traffic
reward is implemented.
"""

from __future__ import annotations


class DefaultRewardFunction:
    """Compatibility reward implementation for the current project stage."""

    def compute(self, state, action, next_state, info) -> float:
        """Compute the transition reward.

        The raw SUMO reward is kept inside ``info['raw_reward']`` temporarily so
        the rest of the project can migrate to the custom reward API without
        contract changes.
        """
        raw_reward = info.get("raw_reward", 0.0) if info is not None else 0.0
        return float(raw_reward)
