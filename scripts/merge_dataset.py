"""Merge many per-episode `.npz` files into a single dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
PROCESSED_DIR = ROOT / "datasets" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def merge_dataset(raw_dir: str | Path = RAW_DIR, output_path: str | Path | None = None) -> dict[str, np.ndarray]:
    raw_dir = Path(raw_dir)
    if output_path is None:
        output_path = PROCESSED_DIR / "merged_dataset.npz"

    files = sorted(raw_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {raw_dir!s}")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []

    for path in files:
        with np.load(path) as data:
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            all_rewards.append(data["rewards"])
            all_next_states.append(data["next_states"])

    dataset = {
        "states": np.concatenate(all_states, axis=0).astype(np.float32),
        "actions": np.concatenate(all_actions, axis=0).astype(np.float32),
        "rewards": np.concatenate(all_rewards, axis=0).astype(np.float32),
        "next_states": np.concatenate(all_next_states, axis=0).astype(np.float32),
    }

    np.savez(output_path, **dataset)
    return dataset


if __name__ == "__main__":
    dataset = merge_dataset()
    metadata = {
        "num_samples": int(dataset["states"].shape[0]),
        "state_dim": int(dataset["states"].shape[1]),
        "output": str(PROCESSED_DIR / "merged_dataset.npz"),
    }
    (PROCESSED_DIR / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Merged dataset saved to: {PROCESSED_DIR / 'merged_dataset.npz'}")
    print(f"Total transitions: {dataset['states'].shape[0]}")
