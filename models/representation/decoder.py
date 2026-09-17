from __future__ import annotations

import torch
from torch import nn


def _get_activation(activation: str) -> nn.Module:
    """Return the configured activation function from its string name."""
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
    }
    if activation not in activations:
        raise ValueError(
            f"Unsupported activation '{activation}'. "
            f"Expected one of: {sorted(activations)}."
        )
    return activations[activation]


class Decoder(nn.Module):
    """Reconstructs the original traffic state from the latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            _get_activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)
