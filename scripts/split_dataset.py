"""Split a dataset into train, validation, and test partitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def split_dataset(
    input_path: str | Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str | Path | None = None,
):
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    with np.load(input_path) as data:
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]

    total = len(states)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    splits = {
        "train": (states[:train_end], actions[:train_end], rewards[:train_end], next_states[:train_end]),
        "validation": (
            states[train_end:validation_end],
            actions[train_end:validation_end],
            rewards[train_end:validation_end],
            next_states[train_end:validation_end],
        ),
        "test": (
            states[validation_end:],
            actions[validation_end:],
            rewards[validation_end:],
            next_states[validation_end:],
        ),
    }

    saved_paths = {}
    for name, arrays in splits.items():
        dataset = {
            "states": arrays[0].astype(np.float32),
            "actions": arrays[1].astype(np.float32),
            "rewards": arrays[2].astype(np.float32),
            "next_states": arrays[3].astype(np.float32),
        }
        output_path = output_dir / f"{name}.npz"
        np.savez(output_path, **dataset)
        saved_paths[name] = output_path

    return saved_paths


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "normalized_dataset.npz"
    if not input_path.exists():
        raise FileNotFoundError(f"Normalized dataset not found: {input_path}")

    saved = split_dataset(input_path)
    print("Saved split datasets:")
    for name, path in saved.items():
        print(f"- {name}: {path}")
