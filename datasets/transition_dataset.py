"""PyTorch-friendly dataset wrapper for traffic transition samples.

Each sample is interpreted as a transition from the state observed before the
agent applies an action to the next_state observed after the environment step.
The dataset also carries episode-level metadata so trajectories can be rebuilt
and terminal conditions can be respected during dynamic-model training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Dataset for (state, action, reward, next_state, episode_id, time_step, terminated, truncated)."""

    def __init__(self, path: str | Path):
        dataset_path = Path(path)
        data = np.load(dataset_path)

        self.states = torch.from_numpy(np.asarray(data["states"], dtype=np.float32))
        self.actions = torch.from_numpy(np.asarray(data["actions"], dtype=np.float32))
        self.rewards = torch.from_numpy(np.asarray(data["rewards"], dtype=np.float32))
        self.next_states = torch.from_numpy(np.asarray(data["next_states"], dtype=np.float32))
        self.episode_ids = torch.from_numpy(np.asarray(data.get("episode_id", np.zeros(len(self.states), dtype=np.int64)), dtype=np.int64))
        self.time_steps = torch.from_numpy(np.asarray(data.get("time_step", np.arange(len(self.states), dtype=np.int64)), dtype=np.int64))
        self.terminated = torch.from_numpy(np.asarray(data.get("terminated", np.zeros(len(self.states), dtype=bool)), dtype=np.bool_))
        self.truncated = torch.from_numpy(np.asarray(data.get("truncated", np.zeros(len(self.states), dtype=bool)), dtype=np.bool_))

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
