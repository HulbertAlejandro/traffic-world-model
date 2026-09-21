"""Windowed latent-sequence dataset for training the temporal model.

Builds fixed-length windows of (z_t, a_t) pairs, entirely within a single
episode, together with the target next latent state and reward. Episodes
shorter than ``sequence_length`` are discarded entirely rather than padded,
by explicit project decision: it keeps the training signal free of any
artificial padding value that the model could learn to exploit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    """Yields (latent_window, action_window, target_latent, target_reward)."""

    def __init__(self, path: str | Path, sequence_length: int, action_dim: int = 2):
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.sequence_length = sequence_length
        self.action_dim = action_dim

        with np.load(Path(path)) as data:
            z = data["z"].astype(np.float32)
            next_z = data["next_z"].astype(np.float32)
            actions = data["actions"].astype(np.int64)
            rewards = data["rewards"].astype(np.float32)
            episode_id = data["episode_id"].astype(np.int64)
            time_step = data["time_step"].astype(np.int64)

        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]] = []
        self._build_windows(z, next_z, actions, rewards, episode_id, time_step)

    def _build_windows(self, z, next_z, actions, rewards, episode_id, time_step) -> None:
        for current_episode_id in np.unique(episode_id):
            mask = episode_id == current_episode_id
            order = np.argsort(time_step[mask])

            episode_z = z[mask][order]
            episode_next_z = next_z[mask][order]
            episode_actions = actions[mask][order]
            episode_rewards = rewards[mask][order]
            episode_length = len(episode_z)

            if episode_length < self.sequence_length:
                continue  # discard short episodes, per project decision

            last_start = episode_length - self.sequence_length
            for start in range(last_start + 1):
                end = start + self.sequence_length
                self._windows.append(
                    (
                        episode_z[start:end],
                        episode_actions[start:end],
                        episode_next_z[end - 1],
                        float(episode_rewards[end - 1]),
                        int(current_episode_id),
                        int(start),
                    )
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        latent_window, action_window, target_latent, target_reward, episode_id, start = self._windows[idx]

        latent_window = torch.from_numpy(latent_window)
        action_window = F.one_hot(
            torch.from_numpy(action_window), num_classes=self.action_dim
        ).float()
        target_latent = torch.from_numpy(target_latent)
        target_reward = torch.tensor(target_reward, dtype=torch.float32)

        return (
            latent_window,
            action_window,
            target_latent,
            target_reward,
            episode_id,
            start,
        )
