"""Normalize dataset splits using a scaler fitted only on the training split."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "states": data["states"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
            "rewards": data["rewards"].astype(np.float32),
            "next_states": data["next_states"].astype(np.float32),
            "episode_id": data["episode_id"].astype(np.int64),
            "time_step": data["time_step"].astype(np.int64),
            "terminated": data["terminated"].astype(bool),
            "truncated": data["truncated"].astype(bool),
        }


def normalize_dataset(
    train_path: str | Path,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    scaler_path: str | Path | None = None,
):
    train_path = Path(train_path)
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if scaler_path is None:
        scaler_path = output_dir / "scaler.pkl"

    train_data = _load_split(train_path)
    state_mean = train_data["states"].mean(axis=0, keepdims=True)
    state_std = train_data["states"].std(axis=0, keepdims=True)
    state_std[state_std < 1e-8] = 1.0

    scaler = {"state_mean": state_mean, "state_std": state_std}
    with Path(scaler_path).open("wb") as handle:
        pickle.dump(scaler, handle)

    saved_paths = {}
    for name, source in {"train": train_path, "validation": validation_path, "test": test_path}.items():
        if source is None:
            continue
        data = _load_split(source)
        normalized_states = (data["states"] - state_mean) / state_std
        normalized_next_states = (data["next_states"] - state_mean) / state_std
        output_path = output_dir / f"{name}.npz"
        np.savez(
            output_path,
            states=normalized_states,
            actions=data["actions"],
            rewards=data["rewards"],
            next_states=normalized_next_states,
            episode_id=data["episode_id"],
            time_step=data["time_step"],
            terminated=data["terminated"],
            truncated=data["truncated"],
        )
        saved_paths[name] = output_path

    return saved_paths, scaler


if __name__ == "__main__":
    train_path = PROCESSED_DIR / "train_raw.npz"
    validation_path = PROCESSED_DIR / "validation_raw.npz"
    test_path = PROCESSED_DIR / "test_raw.npz"

    for required in (train_path, validation_path, test_path):
        if not required.exists():
            raise FileNotFoundError(f"Required split dataset not found: {required}")

    saved, scaler = normalize_dataset(train_path, validation_path, test_path)
    print("Saved normalized splits:")
    for name, path in saved.items():
        print(f"- {name}: {path}")
    print(f"Scaler keys: {list(scaler.keys())}")
