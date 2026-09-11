"""Project-defined action space for the traffic signal controller."""

from __future__ import annotations

import numpy as np


class ProjectActionSpace:
    """Discrete action space for the signal controller.

    Actions are defined by the project specification:
    - 0: keep the current phase
    - 1: switch to the next phase
    """

    def __init__(self, num_actions: int = 2) -> None:
        self.n = int(num_actions)

    def sample(self) -> int:
        """Sample a valid control action."""
        return int(np.random.randint(self.n))

    def contains(self, action) -> bool:
        """Return whether an action is valid for this project."""
        try:
            return int(action) in (0, 1)
        except (TypeError, ValueError):
            return False

    def __contains__(self, action) -> bool:
        return self.contains(action)

    def __len__(self) -> int:
        return self.n
