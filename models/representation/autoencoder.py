from __future__ import annotations

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder


class Autoencoder(nn.Module):
    """Autoencoder for learning a compact representation of traffic states."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latent representation and reconstructed input."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction
