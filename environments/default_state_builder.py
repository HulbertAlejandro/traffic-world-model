"""Default state builder.

This class is the initial implementation of the project-level state builder. It
encapsulates the raw SUMO observation but keeps the API ready to be replaced by
TraCI-based custom state extraction without modifying the rest of the project.
"""

from __future__ import annotations

import numpy as np


class DefaultStateBuilder:
    """Temporary state representation based on the simulator observation."""

    def build(self, observation, env) -> np.ndarray:
        """Return the state vector for the current transition.

        Args:
            observation: raw observation returned by the simulator
            env: environment instance to allow future custom state builders
        """
        return np.asarray(observation, dtype=np.float32)
