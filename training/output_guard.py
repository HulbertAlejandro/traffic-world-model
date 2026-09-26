"""Refuse to overwrite official or archived results from a training script.

Same rule as training/train_controller_direct.py: a protected folder is never
written without --overwrite-official, and a folder that already holds any file
the run would produce is never written without --overwrite. Both checks run
before any training starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_output_arguments(parser: argparse.ArgumentParser, default_output_dir: Path, protected_help: str) -> None:
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_dir,
        help=f"folder for checkpoints, hyperparameters and the loss curve (default: {default_output_dir})",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing files already in --output-dir")
    parser.add_argument("--overwrite-official", action="store_true", help=f"allow writing into {protected_help}")


def check_output_dir(
    output_dir: Path,
    protected_dirs: tuple[Path, ...],
    output_files: tuple[str, ...],
    overwrite: bool,
    overwrite_official: bool,
) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir in {path.resolve() for path in protected_dirs} and not overwrite_official:
        raise SystemExit(
            f"{output_dir} holds official/archived checkpoints. Choose another --output-dir, "
            "or pass --overwrite-official if replacing them was explicitly approved."
        )
    existing = [name for name in output_files if (output_dir / name).exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"{output_dir} already contains {existing}. Choose an empty --output-dir, or pass --overwrite."
        )
    return output_dir
