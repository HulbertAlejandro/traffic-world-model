"""Merge a group of per-episode `.npz` files into a single pooled dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
PROCESSED_DIR = ROOT / "datasets" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_FIELDS = ("states", "actions", "rewards", "next_states", "terminated", "truncated", "episode_id", "time_step")
_FLOAT_FIELDS = {"states": np.float32, "rewards": np.float32, "next_states": np.float32}
_INT_FIELDS = {"actions": np.int64, "episode_id": np.int64, "time_step": np.int64}
_BOOL_FIELDS = {"terminated": bool, "truncated": bool}


def merge_files(files: list[Path], output_path: str | Path) -> dict[str, np.ndarray]:
    """Concatenate a specific list of episode files, in the given order.

    Unlike merging an entire directory, this function is meant to be called
    once per train/validation/test group of episode files (see
    ``split_dataset.py``), so that episode boundaries never cross a split.
    """
    if not files:
        raise ValueError("No files provided to merge.")

    pooled: dict[str, list[np.ndarray]] = {field: [] for field in _FIELDS}
    for path in files:
        with np.load(path) as data:
            for field in _FIELDS:
                pooled[field].append(data[field])

    dataset: dict[str, np.ndarray] = {}
    for field in _FIELDS:
        dtype = _FLOAT_FIELDS.get(field) or _INT_FIELDS.get(field) or _BOOL_FIELDS.get(field)
        dataset[field] = np.concatenate(pooled[field], axis=0).astype(dtype)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dataset)
    return dataset


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {RAW_DIR!s}")
    dataset = merge_files(files, PROCESSED_DIR / "all_raw.npz")
    print(f"Merged {len(files)} episodes -> {PROCESSED_DIR / 'all_raw.npz'}")
    print(f"Total transitions: {dataset['states'].shape[0]}")