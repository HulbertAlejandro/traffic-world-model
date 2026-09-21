from __future__ import annotations

from typing import Protocol

import torch


class TemporalModel(Protocol):
    """Common interface for future temporal models."""

    def forward(
        self,
        latent_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        ...
