"""PyTorch-friendly dataset wrapper for traffic transition samples.

Each sample is interpreted as a transition from the state observed before the
agent applies an action to the next_state observed after the environment step.

``actions`` are kept as ``torch.long`` (discrete action ids), matching how
they are stored throughout the rest of the pipeline. Cast to float only at
the point a specific model architecture requires it, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Dataset for (state, action, reward, next_state, episode_id, time_step, terminated, truncated)."""

    def __init__(self, path: str | Path):
        data = np.load(Path(path))
        n = len(data["states"])

        self.states = torch.from_numpy(np.asarray(data["states"], dtype=np.float32))
        self.actions = torch.from_numpy(np.asarray(data["actions"], dtype=np.int64))
        self.rewards = torch.from_numpy(np.asarray(data["rewards"], dtype=np.float32))
        self.next_states = torch.from_numpy(np.asarray(data["next_states"], dtype=np.float32))
        self.episode_ids = torch.from_numpy(
            np.asarray(data.get("episode_id", np.zeros(n, dtype=np.int64)), dtype=np.int64)
        )
        self.time_steps = torch.from_numpy(
            np.asarray(data.get("time_step", np.arange(n, dtype=np.int64)), dtype=np.int64)
        )
        # dtype passed here must be a NumPy dtype (np.bool_), not torch.bool --
        # torch.from_numpy infers the torch.bool tensor automatically.
        self.terminated = torch.from_numpy(
            np.asarray(data.get("terminated", np.zeros(n, dtype=bool)), dtype=np.bool_)
        )
        self.truncated = torch.from_numpy(
            np.asarray(data.get("truncated", np.zeros(n, dtype=bool)), dtype=np.bool_)
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.episode_ids[idx],
            self.time_steps[idx],
            self.terminated[idx],
            self.truncated[idx],
        )