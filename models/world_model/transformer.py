from __future__ import annotations

import math

import torch
from torch import nn

from .base import TemporalModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Adds a fixed, non-trainable position signal to a batch-first sequence of
    shape ``(batch, sequence_length, d_model)``. Without it, self-attention is
    permutation-invariant and the model could not tell the order of the steps
    in the window.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.max_len = int(max_len)

        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(self.max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        # Non-persistent buffer: moves with .to(device), is never updated by the
        # optimizer, and is left out of the state_dict (it is recomputed here).
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        if sequence_length > self.max_len:
            raise ValueError(
                "sequence is longer than the positional encoding supports; "
                f"max_len is {self.max_len}, got {sequence_length}"
            )
        return x + self.encoding[:, :sequence_length, :]


class LatentDynamicsTransformer(nn.Module, TemporalModel):
    """Transformer-based latent dynamics model for the traffic World Model.

    Drop-in alternative to ``LatentDynamicsLSTM`` (Experimento 3). It consumes
    the same temporal window of latent states paired with actions:

        [(z_t, a_t), (z_{t+1}, a_{t+1}), ..., (z_{t+L-1}, a_{t+L-1})]

    and predicts the same two outputs:

        (z_hat_{t+L}, r_hat_{t+L})

    Each (z, a) pair is projected to ``d_model``, a sinusoidal positional
    encoding is added, and the sequence goes through a stack of standard
    ``nn.TransformerEncoderLayer`` blocks. As in the LSTM, only the output of
    the last step feeds the two deterministic heads. No causal mask is used:
    the target lies outside the window, so letting every step attend to every
    other step leaks no future information.
    """

    DEFAULT_MAX_LEN = 512

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        sequence_length: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if nhead <= 0:
            raise ValueError(f"nhead must be positive, got {nhead}")
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, got d_model={d_model}, nhead={nhead}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {dim_feedforward}")
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.sequence_length = sequence_length

        self.input_projection = nn.Linear(self.latent_dim + self.action_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            max_len=sequence_length if sequence_length is not None else self.DEFAULT_MAX_LEN,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.latent_head = nn.Linear(self.d_model, self.latent_dim)
        self.reward_head = nn.Linear(self.d_model, 1)

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
        projected = self.positional_encoding(self.input_projection(combined))
        outputs = self.encoder(projected)
        last_hidden = outputs[:, -1, :]

        next_latent = self.latent_head(last_hidden)
        next_reward = self.reward_head(last_hidden).squeeze(-1)

        return next_latent, next_reward
