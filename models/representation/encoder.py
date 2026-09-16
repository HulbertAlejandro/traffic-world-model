from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Compresses the traffic state into a latent representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
