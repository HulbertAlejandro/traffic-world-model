"""PyTorch-friendly dataset wrapper for traffic transition samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Dataset for (state, action, reward, next_state) tuples."""

    def __init__(self, path: str | Path):
        dataset_path = Path(path)
        data = np.load(dataset_path)

        self.states = torch.from_numpy(np.asarray(data["states"], dtype=np.float32))
        self.actions = torch.from_numpy(np.asarray(data["actions"], dtype=np.float32))
        self.rewards = torch.from_numpy(np.asarray(data["rewards"], dtype=np.float32))
        self.next_states = torch.from_numpy(np.asarray(data["next_states"], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
        )
