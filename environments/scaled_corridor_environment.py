"""Bridge between the real corridor and a PPO controller trained in the v2 Dream Environment.

v2 has no Encoder: the world models were trained on the 104-dim corridor state normalized with
``datasets/v2/processed/scaler.pkl`` (train-split mean and std, written by
scripts/normalize_dataset.py). A policy trained in the dream therefore sees normalized states, and
in real SUMO it must see them normalized with that SAME file -- v1's EncodedTrafficEnvironment once
used a different normalization than training, and the policy acted on states it had never seen.
This wrapper is v2's whole bridge: it normalizes every observation and changes nothing else.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np

from environments.corridor_environment import CorridorTrafficEnvironment

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCALER_PATH = ROOT_DIR / "datasets" / "v2" / "processed" / "scaler.pkl"


def load_state_scaler(scaler_path: str | Path = DEFAULT_SCALER_PATH) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) as flat float32 vectors, from the scaler.pkl of scripts/normalize_dataset.py."""
    with Path(scaler_path).open("rb") as handle:
        scaler = pickle.load(handle)
    mean = np.asarray(scaler["state_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(scaler["state_std"], dtype=np.float32).reshape(-1)
    return mean, std


class ScaledCorridorEnvironment(gym.ObservationWrapper):
    """CorridorTrafficEnvironment whose observations are normalized with the dataset's scaler."""

    def __init__(self, env: CorridorTrafficEnvironment | None = None,
                 scaler_path: str | Path = DEFAULT_SCALER_PATH) -> None:
        super().__init__(env if env is not None else CorridorTrafficEnvironment())
        self.state_mean, self.state_std = load_state_scaler(scaler_path)
        size = self.env.observation_space.shape[0]
        if self.state_mean.shape != (size,) or self.state_std.shape != (size,):
            raise ValueError(f"Scaler shape {self.state_mean.shape} does not match state size {size}.")
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return ((np.asarray(observation, dtype=np.float32) - self.state_mean) / self.state_std).astype(np.float32)
