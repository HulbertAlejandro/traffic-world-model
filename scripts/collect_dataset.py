"""Collect a discrete transition dataset for the traffic world-model pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import ProjectActionSpace, TrafficEnvironment


DATASET_PATH = ROOT_DIR / "data" / "processed" / "dataset.npz"


def collect_dataset(
    num_episodes: int = 5,
    steps_per_episode: int = 60,
    output_path: str | Path = DATASET_PATH,
) -> dict[str, np.ndarray]:
    """Collect a dataset of transitions using the project-defined state and actions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = TrafficEnvironment()
    action_space = ProjectActionSpace()

    states = []
    actions = []
    rewards = []
    next_states = []

    try:
        for _ in range(num_episodes):
            state, _ = env.reset()
            for _ in range(steps_per_episode):
                action = action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(np.asarray(state, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.int64))
                rewards.append(np.asarray(float(reward), dtype=np.float32))
                next_states.append(np.asarray(next_state, dtype=np.float32))

                state = next_state
                if terminated or truncated:
                    break
    finally:
        env.close()

    dataset = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
    }

    np.savez(output_path, **dataset)
    return dataset


if __name__ == "__main__":
    dataset = collect_dataset()
    print(f"Dataset saved to: {output_path if False else DATASET_PATH}")
    print(f"States shape: {dataset['states'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
    print(f"Rewards shape: {dataset['rewards'].shape}")
    print(f"Next states shape: {dataset['next_states'].shape}")
