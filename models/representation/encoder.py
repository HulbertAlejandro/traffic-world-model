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


class Encoder(nn.Module):
    """Compresses the traffic state into a latent representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _get_activation(activation),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
