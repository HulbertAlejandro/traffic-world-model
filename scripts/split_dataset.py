"""Split raw episodes into train/validation/test, at the episode level.

Splitting must happen over whole episode FILES, not over pooled transitions:
consecutive transitions within an episode are correlated, so cutting a
trajectory in half between train and test would leak information and inflate
apparent generalization performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.merge_dataset import merge_files

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
PROCESSED_DIR = ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def split_dataset(
    raw_dir: str | Path = RAW_DIR,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Assign whole episode files to train/validation/test and merge each
    group separately into `<split>_raw.npz` (still unnormalized).

    Also writes ``split_metadata.json`` next to the output files, recording
    the ratios, seed, and per-split episode/transition counts actually used
    for this run -- generated from real values instead of hand-maintained.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    files = sorted(raw_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {raw_dir!s}")

    min_required = max(
        1, *(int(np.ceil(1.0 / ratio)) for ratio in (train_ratio, validation_ratio, test_ratio))
    )
    if len(files) < min_required:
        raise ValueError(
            f"Found {len(files)} episode files, but ratios "
            f"({train_ratio}/{validation_ratio}/{test_ratio}) need at least "
            f"{min_required} episodes for every split to receive at least one. "
            "Collect more episodes or adjust the ratios."
        )

    rng = np.random.default_rng(seed)
    shuffled = [files[i] for i in rng.permutation(len(files))]

    n = len(shuffled)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    groups = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }

    saved_paths: dict[str, Path] = {}
    run_summary: dict[str, dict] = {}
    for name, group_files in groups.items():
        if not group_files:
            raise ValueError(
                f"Split '{name}' received 0 episode files out of {n} total. "
                "Collect more episodes or adjust the ratios."
            )
        output_path = output_dir / f"{name}_raw.npz"
        merged = merge_files(group_files, output_path)
        saved_paths[name] = output_path
        run_summary[name] = {
            "num_episodes": len(group_files),
            "num_transitions": int(merged["states"].shape[0]),
        }

    _write_split_metadata(output_dir, train_ratio, validation_ratio, test_ratio, seed, run_summary)

    return saved_paths


def _write_split_metadata(
    output_dir: Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    run_summary: dict[str, dict],
) -> None:
    """Persist the actual parameters and resulting sizes of this split run."""
    metadata = {
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "seed": seed,
        "splits": run_summary,
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    saved = split_dataset()
    print("Saved per-split raw (unnormalized) datasets:")
    for name, path in saved.items():
        print(f"- {name}: {path}")
    print(f"Run metadata written to: {PROCESSED_DIR / 'split_metadata.json'}")