"""Inspect and visualize a processed dataset split."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"


def summarize_split(path: str | Path) -> None:
    """Print basic statistics for one processed split file."""
    with np.load(path) as data:
        states = data["states"]
        rewards = data["rewards"]
        actions = data["actions"]
        episode_id = data["episode_id"]

        print(f"--- {Path(path).name} ---")
        print(f"Transitions: {states.shape[0]} | State dim: {states.shape[1]}")
        print(f"Episodes: {len(np.unique(episode_id))}")
        print(f"Reward: mean={rewards.mean():.3f} std={rewards.std():.3f} "
              f"min={rewards.min():.3f} max={rewards.max():.3f}")
        print(f"Action distribution: {dict(zip(*np.unique(actions, return_counts=True)))}")
        print()


def plot_reward_histogram(path: str | Path, output_path: str | Path) -> None:
    """Save a histogram of rewards for visual inspection of sparsity/skew."""
    with np.load(path) as data:
        rewards = data["rewards"]

    plt.figure(figsize=(6, 4))
    plt.hist(rewards, bins=40)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title(f"Reward distribution — {Path(path).stem}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    for split_name in ("train", "validation", "test"):
        split_path = PROCESSED_DIR / f"{split_name}.npz"
        if not split_path.exists():
            print(f"Skipping {split_name}: {split_path} not found.")
            continue
        summarize_split(split_path)
        plot_reward_histogram(split_path, PROCESSED_DIR / f"{split_name}_reward_hist.png")
        print(f"Saved histogram: {PROCESSED_DIR / f'{split_name}_reward_hist.png'}")