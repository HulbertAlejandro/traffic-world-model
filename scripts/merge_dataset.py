"""Merge groups of per-episode `.npz` files into pooled datasets.

``merge_files`` is the reusable building block used by ``split_dataset.py``
to merge each train/validation/test group separately, without ever mixing
episodes across splits.

Running this file directly instead pools EVERY episode in ``datasets/raw/``
into a single ``all_episodes.npz``. That file is for exploration only (e.g.
inspecting the overall distribution across the whole collected dataset) --
never use ``all_episodes.npz`` to train or evaluate a model.
"""

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

    Does not decide WHICH episodes belong together -- that is
    ``split_dataset.py``'s responsibility.
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
    dataset = merge_files(files, PROCESSED_DIR / "all_episodes.npz")
    print(f"[exploration only] Pooled {len(files)} episodes -> {PROCESSED_DIR / 'all_episodes.npz'}")
    print("This file has NO train/validation/test separation. Use scripts/split_dataset.py to train.")
    print(f"Total transitions: {dataset['states'].shape[0]}")