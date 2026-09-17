from __future__ import annotations

import torch
from torch import nn


class LatentDynamicsLSTM(nn.Module):
    """Single-step latent dynamics model for predicting the next latent state.

    The world model consumes a sequence of latent vectors, such as
    ``(z_t, z_{t+1}, ..., z_{t+sequence_length-1})``, and returns the predicted
    next latent state ``z_{t+sequence_length}``.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        sequence_length: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        if latent_sequence.ndim != 3:
            raise ValueError(
                "latent_sequence must have shape (batch, sequence_length, latent_dim), "
                f"got {tuple(latent_sequence.shape)}"
            )
        if latent_sequence.shape[-1] != self.latent_dim:
            raise ValueError(
                "latent_sequence last dimension must equal latent_dim; "
                f"expected {self.latent_dim}, got {latent_sequence.shape[-1]}"
            )
        if self.sequence_length is not None and latent_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                "latent_sequence length does not match configured sequence_length; "
                f"expected {self.sequence_length}, got {latent_sequence.shape[1]}"
            )

        outputs, _ = self.lstm(latent_sequence)
        return self.output_head(outputs[:, -1, :])
