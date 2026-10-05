"""MD5 of the official files that Phase 2 must not modify (docs/v2/ADDENDUM_AUTOENCODER.md, section 6).

    python scripts/v2/md5_official.py --tag before
    python scripts/v2/md5_official.py --tag after --compare before
    python scripts/v2/md5_official.py --control --tag control_before    # Phase 3 (docs/v2/ADDENDUM_CONTROL.md)
    python scripts/v2/md5_official.py --planning --tag planning_before  # v2.1-A (docs/v2/ADDENDUM_PLANIFICACION.md)

Covers v1's network, dataset (raw and processed), official checkpoints (Autoencoder, LSTM,
Transformer/TSMixer, the three PPO controllers) and the v2 network, route file and raw dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "results" / "v2" / "autoencoder"
PATTERNS = (
    "environments/single-intersection/*",
    "datasets/raw/*.npz",
    "datasets/processed/*",
    "models/checkpoints/*.pt",
    "models/checkpoints/*.json",
    "models/checkpoints/controller/best_model.zip",
    "models/checkpoints/controller_direct/best_model.zip",
    "models/checkpoints/controller_direct_30k/best_model.zip",
    "environments/four-intersection-corridor/*",
    "datasets/v2/raw/*/*.npz",
)
# --control adds what Phase 3 (control) reads and must not modify: the v2 state scaler and
# processed splits, the sequence data the world models were trained on, and the 20 world models
# that go to control (weights, hyperparameters and reward scalers).
CONTROL_PATTERNS = (
    "datasets/v2/processed/*",
    "models/checkpoints/v2/compression/selection/raw_data/*.npz",
    "models/checkpoints/v2/arch_comparison/lstm_raw_s*/*",
    "models/checkpoints/v2/arch_comparison/transformer_raw_s*/*",
)


# --planning adds, on top of --control, what v2.1-A reuses and must not modify: the 40 Phase 3
# controllers (dream and direct) and the Phase 3 results.
PLANNING_PATTERNS = (
    "models/checkpoints/v2/control/*/*",
    "docs/results/v2/control/*",
)


def snapshot(patterns: tuple[str, ...] = PATTERNS) -> dict[str, str]:
    files = sorted({p for pat in patterns for p in ROOT.glob(pat) if p.is_file()})
    return {p.relative_to(ROOT).as_posix(): hashlib.md5(p.read_bytes()).hexdigest() for p in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--control", action="store_true", help="also cover the Phase 3 inputs (CONTROL_PATTERNS)")
    parser.add_argument("--planning", action="store_true", help="--control plus the Phase 3 controllers and results")
    args = parser.parse_args()
    patterns = PATTERNS
    if args.control or args.planning:
        patterns += CONTROL_PATTERNS
    if args.planning:
        patterns += PLANNING_PATTERNS
    snap = snapshot(patterns)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"md5_{args.tag}.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print(f"{len(snap)} files -> md5_{args.tag}.json")
    if args.compare:
        old = json.loads((OUT / f"md5_{args.compare}.json").read_text(encoding="utf-8"))
        changed = [f for f in old if snap.get(f) != old[f]]
        added = [f for f in snap if f not in old]
        print(f"changed or missing: {changed or 'none'}; new: {added or 'none'}")


if __name__ == "__main__":
    main()
