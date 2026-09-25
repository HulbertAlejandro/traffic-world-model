from __future__ import annotations

import torch
from torch import nn

from .base import TemporalModel


class MixerBlock(nn.Module):
    """One TSMixer block: time-mixing then feature-mixing, each residual.

    Operates on a batch-first sequence of shape
    ``(batch, sequence_length, feature_dim)`` and returns the same shape.

    - Time-mixing: LayerNorm over features, then a dense layer shared across
      features that mixes the ``sequence_length`` steps of each feature
      (applied on the transposed sequence), then ReLU.
    - Feature-mixing: LayerNorm over features, then a two-layer MLP
      ``feature_dim -> hidden_dim -> feature_dim`` shared across steps.
    """

    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.time_norm = nn.LayerNorm(feature_dim)
        self.time_mixing = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mixing = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, L, F) -> (batch, F, L): the Linear acts on the time axis.
        time_mixed = self.time_mixing(self.time_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + time_mixed
        x = x + self.feature_mixing(self.feature_norm(x))
        return x


class LatentDynamicsTSMixer(nn.Module, TemporalModel):
    """All-MLP latent dynamics model for the traffic World Model (TSMixer).

    Drop-in alternative to ``LatentDynamicsLSTM`` (Experimento 3), following
    Chen et al. (2023). It consumes the same temporal window of latent states
    paired with actions:

        [(z_t, a_t), (z_{t+1}, a_{t+1}), ..., (z_{t+L-1}, a_{t+L-1})]

    and predicts the same two outputs:

        (z_hat_{t+L}, r_hat_{t+L})

    Uses neither recurrence nor attention: only dense layers, normalization and
    residual connections. Unlike the LSTM and the Transformer, the time-mixing
    layers are ``Linear(sequence_length, sequence_length)``, so the window
    length must be fixed at construction and every input must match it.

    The feature dimension stays ``latent_dim + action_dim`` through all
    blocks (as in the original TSMixer); only the output of the last step
    feeds the two deterministic heads, for consistency with the other two
    architectures.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        sequence_length: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if sequence_length is None or sequence_length <= 0:
            raise ValueError(
                "TSMixer requires a fixed, positive sequence_length (its time-mixing "
                f"layers operate on that dimension directly), got {sequence_length}"
            )
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.sequence_length = int(sequence_length)

        feature_dim = self.latent_dim + self.action_dim
        self.blocks = nn.Sequential(
            *[
                MixerBlock(self.sequence_length, feature_dim, self.hidden_dim, dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.latent_head = nn.Linear(feature_dim, self.latent_dim)
        self.reward_head = nn.Linear(feature_dim, 1)

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
        if latent_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                "latent_sequence length does not match configured sequence_length; "
                f"expected {self.sequence_length}, got {latent_sequence.shape[1]}"
            )

        combined = torch.cat([latent_sequence, action_sequence], dim=-1)
        outputs = self.blocks(combined)
        last_hidden = outputs[:, -1, :]

        next_latent = self.latent_head(last_hidden)
        next_reward = self.reward_head(last_hidden).squeeze(-1)

        return next_latent, next_reward
