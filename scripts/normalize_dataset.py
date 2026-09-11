"""Normalize a dataset with StandardScaler-like statistics."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    input_path = Path(input_path)
    if output_path is None:
        output_path = PROCESSED_DIR / "normalized_dataset.npz"
    if scaler_path is None:
        scaler_path = PROCESSED_DIR / "scaler.pkl"

    with np.load(input_path) as data:
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        next_states = data["next_states"].astype(np.float32)

    state_mean = states.mean(axis=0, keepdims=True)
    state_std = states.std(axis=0, keepdims=True)
    state_std[state_std < 1e-8] = 1.0

    normalized_states = (states - state_mean) / state_std
    normalized_next_states = (next_states - state_mean) / state_std

    scaler = {
        "state_mean": state_mean,
        "state_std": state_std,
    }
    with Path(scaler_path).open("wb") as handle:
        pickle.dump(scaler, handle)

    np.savez(output_path, states=normalized_states, actions=actions, rewards=rewards, next_states=normalized_next_states)
    return normalized_states, actions, rewards, normalized_next_states, scaler


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "merged_dataset.npz"
    if not input_path.exists():
        raise FileNotFoundError(f"Merged dataset not found: {input_path}")

    _, _, _, _, scaler = normalize_dataset(input_path)
    print(f"Saved normalized dataset to: {PROCESSED_DIR / 'normalized_dataset.npz'}")
    print(f"Scaler keys: {list(scaler.keys())}")
