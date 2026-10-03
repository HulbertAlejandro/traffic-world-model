"""Merge and normalize the v2 dataset for Phase 2 (docs/v2/ADDENDUM_AUTOENCODER.md, section 4).

Uses v1's merge_files and normalize_dataset unchanged: the splits come from the folders fixed in
docs/v2/ADDENDUM_DATASET.md, and the state scaler is fitted on train only. The ``ood`` split is
NOT read here: it is reserved for a final check. Before merging, every file's SHA-256 is checked
against docs/results/v2/dataset/manifest.json.

    python scripts/v2/prepare_dataset_v2.py      # -> datasets/v2/processed/{train,validation,test}.npz
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_dataset import merge_files  # noqa: E402
from scripts.normalize_dataset import normalize_dataset  # noqa: E402

RAW_DIR = ROOT_DIR / "datasets" / "v2" / "raw"
PROCESSED_DIR = ROOT_DIR / "datasets" / "v2" / "processed"
MANIFEST = ROOT_DIR / "docs" / "results" / "v2" / "dataset" / "manifest.json"
SPLITS = ("train", "validation", "test")  # never "ood"


def main() -> None:
    manifest = {e["file"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["episodes"]}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged = {}
    for split in SPLITS:
        files = sorted((RAW_DIR / split).glob("episode_*.npz"))
        for f in files:
            rel = f.relative_to(ROOT_DIR).as_posix()
            if hashlib.sha256(f.read_bytes()).hexdigest() != manifest[rel]["sha256"]:
                raise RuntimeError(f"{rel} does not match the manifest")
        merged[split] = PROCESSED_DIR / f"{split}_raw.npz"
        data = merge_files(files, merged[split])
        print(f"{split}: {len(files)} episodes, {len(data['states'])} transitions")
    normalize_dataset(merged["train"], merged["validation"], merged["test"], output_dir=PROCESSED_DIR)
    print(f"-> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
