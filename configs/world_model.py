"""World-model configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorldModelConfig:
    """Future world-model settings.

    This class keeps the configuration structure ready for latent dynamics and
    imagination-based planning without introducing any training logic yet.
    """

    latent_dim: int = 16
    sequence_length: int = 16
    hidden_dim: int = 128
