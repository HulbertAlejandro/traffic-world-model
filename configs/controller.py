"""Configuration for the PPO controller trained inside the Dream Environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ControllerConfig:
    """Hyperparameters for the PPO controller.

    These are a reasonable starting point, not empirically tuned -- the
    Dream Environment's imagined rollouts are cheap (no SUMO calls), so
    total_timesteps can be generous without a real time cost.
    """

    seed: int = 0
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
