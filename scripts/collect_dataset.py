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

DEFAULT_NUM_EPISODES = 40


def collect_dataset(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    steps_per_episode: int = 60,
    output_dir: str | Path = RAW_DIR,
) -> list[Path]:
    """Collect one `.npz` file per episode, with full transition metadata.

    Each saved episode carries ``episode_id`` (constant per file) and
    ``time_step`` (0-indexed within the episode), plus ``terminated`` and
    ``truncated`` flags, so later stages can split by episode and reconstruct
    trajectories without guessing episode boundaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = TrafficEnvironment()
    action_space = ProjectActionSpace()
    saved_paths: list[Path] = []

    try:
        for episode_idx in range(num_episodes):
            states, actions, rewards, next_states = [], [], [], []
            terminated_flags, truncated_flags = [], []

            state, _ = env.reset()
            for time_step in range(steps_per_episode):
                action = action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(np.asarray(state, dtype=np.float32))
                actions.append(np.int64(action))
                rewards.append(np.float32(reward))
                next_states.append(np.asarray(next_state, dtype=np.float32))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))

                state = next_state
                if terminated or truncated:
                    break

            num_transitions = len(states)
            dataset = {
                "states": np.asarray(states, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "next_states": np.asarray(next_states, dtype=np.float32),
                "terminated": np.asarray(terminated_flags, dtype=bool),
                "truncated": np.asarray(truncated_flags, dtype=bool),
                "episode_id": np.full(num_transitions, episode_idx, dtype=np.int64),
                "time_step": np.arange(num_transitions, dtype=np.int64),
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