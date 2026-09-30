"""Training loop for the Transformer temporal model on the raw normalized state.

Raw-state counterpart of training/train_world_model_transformer.py, with the same relation
that training/train_world_model_raw.py has to training/train_world_model.py: the training
protocol (seeds, optimizer, epochs, early stopping, loss, reward normalization) is IMPORTED
from train_world_model.py, and the architecture hyperparameters, model class and shared reward
scaler from train_world_model_transformer.py, so nothing can drift apart. Only the input changes:
the *_raw_seq.npz files (scripts/prepare_raw_sequence_dataset.py), whose size sets the input
dimension. --data-dir points at another folder with the same files (for example the 24-dim state
of the latent_dim exploration).

train() takes explicit paths, so the latent-vs-raw exploration runs BOTH of its branches through
this same function. Run with the official latent files, seed 0 and 100 epochs, it reproduces
models/checkpoints/world_model_transformer_best.pt tensor for tensor.

Usage (defaults: seed 0, 100 epochs, official 26-dim raw state):
    python training/train_world_model_transformer_raw.py --seed 3 --epochs 300 \\
        --data-dir models/checkpoints/exploratory_latent_sweep/data --output-dir <folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import WorldModelConfig
from datasets.latent_sequence_dataset import LatentSequenceDataset
from training.output_guard import add_output_arguments, check_output_dir
from training.train_world_model import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DIR,
    SEED,
    WEIGHT_DECAY,
    _set_seeds,
    train_one_epoch,
    validate,
)
from models.world_model import LatentDynamicsTransformer
from training.train_world_model_transformer import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    NHEAD,
    NUM_LAYERS,
    load_shared_reward_scaler,
)

ARCHITECTURE = "transformer"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "checkpoints" / "transformer_raw_state"
# models/checkpoints/ holds the official Transformer of Experimento 3.
PROTECTED_DIRS = (CHECKPOINT_DIR,)
OUTPUT_FILES = ("world_model_best.pt", "world_model_best.json", "world_model_last.pt",
                "world_model_last.json", "reward_scaler.json", "train_summary.json")


def build_model(config: WorldModelConfig) -> torch.nn.Module:
    return LatentDynamicsTransformer(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        sequence_length=config.sequence_length,
        dropout=DROPOUT,
    )


def save_checkpoint(model, config: WorldModelConfig, path: Path, seed: int, max_epochs: int) -> None:
    torch.save(model.state_dict(), path)
    hyperparams = {
        "architecture": ARCHITECTURE,
        "latent_dim": config.latent_dim,
        "action_dim": config.action_dim,
        "sequence_length": config.sequence_length,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "seed": seed,
        "max_epochs": max_epochs,
    }
    path.with_suffix(".json").write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")


def train(train_path: Path, validation_path: Path, seed: int, epochs: int, output_dir: Path) -> dict:
    """Same call order as train_world_model_transformer.main(), with explicit paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(train_path) as data:
        input_dim = int(data["z"].shape[1])
    config = WorldModelConfig(latent_dim=input_dim, sequence_length=16)

    reward_scaler = load_shared_reward_scaler(CHECKPOINT_DIR / "reward_scaler.json")
    (output_dir / "reward_scaler.json").write_text(json.dumps(reward_scaler, indent=2), encoding="utf-8")
    reward_mean = torch.tensor(reward_scaler["reward_mean"], device=device)
    reward_std = torch.tensor(reward_scaler["reward_std"], device=device)

    train_dataset = LatentSequenceDataset(train_path, sequence_length=config.sequence_length, action_dim=config.action_dim)
    validation_dataset = LatentSequenceDataset(validation_path, sequence_length=config.sequence_length, action_dim=config.action_dim)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_validation_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    stopped_early, epoch = False, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, reward_mean, reward_std)
        validation_loss = validate(model, validation_loader, device, reward_mean, reward_std)
        save_checkpoint(model, config, output_dir / "world_model_last.pt", seed, epochs)
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch = validation_loss, epoch
            save_checkpoint(model, config, output_dir / "world_model_best.pt", seed, epochs)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={validation_loss:.6f}")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping en época {epoch} (sin mejora en {EARLY_STOPPING_PATIENCE} épocas).")
            stopped_early = True
            break

    summary = {"architecture": ARCHITECTURE, "seed": seed, "input_dim": input_dim, "train": str(train_path),
               "max_epochs": epochs, "last_epoch": epoch, "best_epoch": best_epoch,
               "best_validation_loss": best_validation_loss, "stopped_early": stopped_early}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer temporal model on the raw normalized state.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"training seed (default: {SEED})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"maximum epochs; early stopping (patience {EARLY_STOPPING_PATIENCE}) may end earlier (default: {EPOCHS})")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR,
                        help="folder with train_raw_seq.npz and validation_raw_seq.npz (default: datasets/processed)")
    add_output_arguments(parser, DEFAULT_OUTPUT_DIR, "models/checkpoints/ (official Experimento 3 models)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = check_output_dir(args.output_dir, PROTECTED_DIRS, OUTPUT_FILES, args.overwrite, args.overwrite_official)
    train_path, validation_path = args.data_dir / "train_raw_seq.npz", args.data_dir / "validation_raw_seq.npz"
    for required in (train_path, validation_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}. Run scripts/prepare_raw_sequence_dataset.py first.")
    print(json.dumps(train(train_path, validation_path, args.seed, args.epochs, output_dir)))


if __name__ == "__main__":
    main()
