"""Prepare raw-state sequence datasets for Experimento 0.

Renames states/next_states -> z/next_z in each processed split, so that
LatentSequenceDataset (built for the Autoencoder's latent output) can be
reused UNCHANGED for training an LSTM directly on the raw normalized state.
This avoids maintaining two separate windowing implementations that could
silently drift apart -- the episode-boundary logic stays a single source of
truth for both experiments.

This does NOT touch the Autoencoder pipeline at all: train.npz/validation.npz/
test.npz are read, never modified, and a new *_raw_seq.npz is written next to
them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"


def prepare_raw_sequence_split(split_path: Path) -> Path:
    """Rename states/next_states -> z/next_z for one processed split."""
    with np.load(split_path) as data:
        payload = {
            "z": data["states"].astype(np.float32),
            "next_z": data["next_states"].astype(np.float32),
            "actions": data["actions"].astype(np.int64),
            "rewards": data["rewards"].astype(np.float32),
            "episode_id": data["episode_id"].astype(np.int64),
            "time_step": data["time_step"].astype(np.int64),
        }

    output_path = split_path.with_name(split_path.stem + "_raw_seq.npz")
    np.savez(output_path, **payload)
    return output_path


if __name__ == "__main__":
    for split_name in ("train", "validation", "test"):
        split_path = PROCESSED_DIR / f"{split_name}.npz"
        if not split_path.exists():
            print(f"Skipping {split_name}: {split_path} not found.")
            continue
        output_path = prepare_raw_sequence_split(split_path)
        print(f"Prepared {split_name} raw-state sequence -> {output_path}")
