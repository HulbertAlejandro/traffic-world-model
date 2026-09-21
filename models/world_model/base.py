from __future__ import annotations

from typing import Protocol

import torch


class TemporalModel(Protocol):
    """Common interface for future temporal models (LSTM, Transformer, TSMixer).

    Every implementation consumes a temporal window of latent states paired
    with the actions taken at each step, and predicts both the next latent
    state and the reward associated with that transition:

        [(z_t, a_t), (z_{t+1}, a_{t+1}), ..., (z_{t+L-1}, a_{t+L-1})]
        -> (z_hat_{t+L}, r_hat_{t+L})
    """

    def forward(
        self,
        latent_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
