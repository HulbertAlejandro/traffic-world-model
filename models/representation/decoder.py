from __future__ import annotations

import torch
from torch import nn


class Decoder(nn.Module):
    """Reconstructs the original traffic state from the latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)
