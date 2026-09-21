"""World-model configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass

from .representation import RepresentationConfig


@dataclass(slots=True)
class WorldModelConfig:
    """Future world-model settings.

    This class keeps the configuration structure ready for latent dynamics and
    imagination-based planning without introducing any training logic yet.
    The representation config remains the single source of truth for the latent
    size used by the encoder.
    """

    representation: RepresentationConfig | None = None
    latent_dim: int | None = None
    action_dim: int = 2
    sequence_length: int = 16
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if self.representation is not None:
            if self.latent_dim is not None and self.latent_dim != self.representation.latent_dim:
                raise ValueError(
                    "WorldModelConfig.latent_dim must match "
                    "RepresentationConfig.latent_dim when both are provided."
                )
            self.latent_dim = self.representation.latent_dim
        elif self.latent_dim is None:
            self.latent_dim = 16

        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
