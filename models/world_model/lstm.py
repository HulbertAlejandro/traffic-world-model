from __future__ import annotations

import torch
from torch import nn

from .base import TemporalModel


class LatentDynamicsLSTM(nn.Module, TemporalModel):
    """Latent dynamics model for the traffic World Model.

    The model consumes a temporal window of latent states paired with the
    traffic-light actions taken at each step:

        [(z_t, a_t), (z_{t+1}, a_{t+1}), ..., (z_{t+L-1}, a_{t+L-1})]

    and predicts both the next latent state and its associated reward:

        (z_hat_{t+L}, r_hat_{t+L})

    This mirrors the LSTM -> Dense -> prediction pattern used for the
    Autoencoder, extended with a second output head for the reward, as
    specified in Section 15 of the project proposal. No Mixture Density
    Network is used: both outputs are deterministic point predictions
    trained with MSE loss.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        sequence_length: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.latent_dim + self.action_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.latent_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        latent_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_sequence.ndim != 3:
            raise ValueError(
                "latent_sequence must have shape (batch, sequence_length, latent_dim), "
                f"got {tuple(latent_sequence.shape)}"
            )
        if action_sequence.ndim != 3:
            raise ValueError(
                "action_sequence must have shape (batch, sequence_length, action_dim), "
                f"got {tuple(action_sequence.shape)}"
            )
        if latent_sequence.shape[0] != action_sequence.shape[0]:
            raise ValueError(
                "latent_sequence and action_sequence must have the same batch size; "
                f"got {latent_sequence.shape[0]} and {action_sequence.shape[0]}"
            )
        if latent_sequence.shape[1] != action_sequence.shape[1]:
            raise ValueError(
                "latent_sequence and action_sequence must have the same sequence_length; "
                f"got {latent_sequence.shape[1]} and {action_sequence.shape[1]}"
            )
        if latent_sequence.shape[-1] != self.latent_dim:
            raise ValueError(
                "latent_sequence last dimension must equal latent_dim; "
                f"expected {self.latent_dim}, got {latent_sequence.shape[-1]}"
            )
        if action_sequence.shape[-1] != self.action_dim:
            raise ValueError(
                "action_sequence last dimension must equal action_dim; "
                f"expected {self.action_dim}, got {action_sequence.shape[-1]}"
            )
        if self.sequence_length is not None and latent_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                "latent_sequence length does not match configured sequence_length; "
                f"expected {self.sequence_length}, got {latent_sequence.shape[1]}"
            )

        combined = torch.cat([latent_sequence, action_sequence], dim=-1)
        outputs, _ = self.lstm(combined)
        last_hidden = outputs[:, -1, :]

        next_latent = self.latent_head(last_hidden)
        next_reward = self.reward_head(last_hidden).squeeze(-1)

        return next_latent, next_reward
