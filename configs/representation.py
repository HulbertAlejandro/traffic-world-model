"""Configuration for the traffic-state representation model (Autoencoder).

This config is intentionally agnostic to whether the representation model
ends up being a plain Autoencoder or a Variational Autoencoder later: it only
describes shape/training hyperparameters that both variants share. Neither
``Encoder`` nor ``Decoder`` (Commit 2) read anything from this file that is
specific to one architecture over the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class RepresentationConfig:
    """Hyperparameters for training the state representation model.

    ``input_dim`` has NO default on purpose. The state vector's dimension
    has already changed once in this project (22 -> 26, when the signal
    phase moved to one-hot encoding), and hardcoding it here would repeat
    the same class of bug we already fixed elsewhere. Callers must obtain it
    from the live source of truth -- ``CustomStateBuilder.state_size()`` --
    the same pattern ``TrafficEnvironment.observation_space`` already uses.
    """

    input_dim: int
    hidden_dim: int = 16
    latent_dim: int = 16
    activation: str = "relu"

    learning_rate: float = 1e-3
    optimizer: str = "adam"
    batch_size: int = 32
    epochs: int = 100

    checkpoint_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints"
    )

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.latent_dim >= self.input_dim:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) should be smaller than "
                f"input_dim ({self.input_dim}) for the bottleneck to compress "
                "anything; if you want no compression, that's Experimento 0's "
                "raw-vector baseline, not this config."
            )
