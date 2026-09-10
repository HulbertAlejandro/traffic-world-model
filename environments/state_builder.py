"""State builder contract and compatibility exports."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from environments.default_state_builder import DefaultStateBuilder


class StateBuilder(Protocol):
    """Contract that any project state builder must implement."""

    def build(self, observation, env) -> np.ndarray:
        """Build the project state from the simulator observation."""
        ...


__all__ = ["StateBuilder", "DefaultStateBuilder"]