"""Training configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    """Future training settings for representation and controller training."""

    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-4
    device: str = "cpu"
