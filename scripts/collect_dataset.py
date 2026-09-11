"""Collect raw episode datasets for the traffic world-model pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import ProjectActionSpace, TrafficEnvironment


DATASET_ROOT = ROOT_DIR / "datasets"
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def collect_dataset(
    num_episodes: int = 3,
    steps_per_episode: int = 60,
    output_dir: str | Path = RAW_DIR,
) -> list[Path]:
    """Collect one `.npz` file per episode in the raw datasets folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = TrafficEnvironment()
    action_space = ProjectActionSpace()
    saved_paths: list[Path] = []

    try:
        for episode_idx in range(num_episodes):
            states = []
            actions = []
            rewards = []
            next_states = []

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

            dataset = {
                "states": np.asarray(states, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "next_states": np.asarray(next_states, dtype=np.float32),
            }

            output_path = output_dir / f"episode_{episode_idx:03d}.npz"
            np.savez(output_path, **dataset)
            saved_paths.append(output_path)
    finally:
        env.close()

    return saved_paths


if __name__ == "__main__":
    files = collect_dataset()
    print(f"Saved raw episodes to: {RAW_DIR}")
    for item in files:
        print(f"- {item.name}")
