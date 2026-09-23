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

    # Imagined horizon for THIS training run; matches DreamEnvironment's own
    # default (7). A value of 20 was tried (commit 8d84d52) to cover long action
    # streaks that seemed to cause catastrophic SUMO episodes, but that diagnosis
    # turned out to be an artifact of the EncodedTrafficEnvironment normalization
    # bug (commit 990c6e5): with the corrected bridge, the policy trained with 7
    # (v1) beats the one trained with 20 (v2) in all 30 evaluated episodes.
    # Reverted to 7; the field is kept so the horizon stays an explicit knob.
    dream_max_steps: int = 7

    # Reward normalization (VecNormalize) during PPO training, in both the
    # Dream-trained and the direct-RL scripts. Recorded here so every
    # checkpoint's sidecar .json says whether it was trained with it.
    # reward_clip bounds the NORMALIZED reward (in units of the running std of
    # the discounted return), not the raw one.
    normalize_reward: bool = True
    reward_clip: float = 10.0

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
        if self.dream_max_steps <= 0:
            raise ValueError(f"dream_max_steps must be positive, got {self.dream_max_steps}")
        if self.reward_clip <= 0:
            raise ValueError(f"reward_clip must be positive, got {self.reward_clip}")
