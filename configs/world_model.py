"""Configuration of the temporal model (LatentDynamicsLSTM and its Experimento 3 alternatives)."""

from __future__ import annotations

from dataclasses import dataclass

from .representation import RepresentationConfig


@dataclass(slots=True)
class WorldModelConfig:
    """Shape of the temporal model: latent_dim, action_dim, sequence_length, hidden_dim.

    Used by training/train_world_model.py (and by the Transformer, TSMixer and
    raw-state trainings) to build the model; the values actually used are saved in
    the .json next to each checkpoint. latent_dim is taken from the
    RepresentationConfig when one is given, so the representation config remains
    the single source of truth for the latent size produced by the encoder.
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
