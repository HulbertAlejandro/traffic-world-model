"""Project-defined action space for the traffic signal controller."""

from __future__ import annotations

import numpy as np


class ProjectActionSpace:
    """Discrete action space for the signal controller.

    An action is the INDEX OF THE TARGET GREEN PHASE (0 or 1), not a
    keep/switch toggle. The project specification originally described it as
    "0 = keep the current phase, 1 = switch to the next phase", but sumo-rl
    passes the value straight to
    ``sumo_rl.environment.traffic_signal.TrafficSignal.set_next_phase``, which
    treats it as the green phase to go to. The signal only changes when the
    requested phase differs from the current green phase AND at least
    ``yellow_time + min_green`` seconds have passed since the last change;
    otherwise the current phase is kept, whatever the action.

    With the 2 green phases of this intersection this means:
    - current phase 0: action 1 switches, action 0 keeps;
    - current phase 1: action 0 switches, action 1 keeps.

    So "action 1" matches the old "switch" meaning only while phase 0 is
    active -- about half of the steps in the recorded trajectories, not almost
    always. The whole pipeline (dataset, Autoencoder, temporal models, both
    PPO controllers) uses this target-phase convention consistently. See
    PROJECT_STATUS.md, "Controlador PPO contra SUMO real", "Notas tecnicas".
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
