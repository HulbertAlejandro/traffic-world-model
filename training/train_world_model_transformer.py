"""Training loop for the Transformer latent dynamics model (Experimento 3).

Mirror of training/train_world_model.py with only the model class and the
checkpoint names changed. The training protocol (seeds, optimizer, epochs,
early stopping, loss, reward normalization) is imported from that script
rather than copied, so the three architectures are guaranteed to be trained
under exactly the same conditions.

The reward scaler is READ from models/checkpoints/reward_scaler.json, written
by training/train_world_model.py -- never recomputed here -- so every
architecture normalizes the reward with the same statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import RepresentationConfig, WorldModelConfig
from datasets.latent_sequence_dataset import LatentSequenceDataset
from models.world_model import LatentDynamicsTransformer
from training.train_world_model import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    _set_seeds,
    train_one_epoch,
    validate,
)

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.0


def load_shared_reward_scaler(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Corre training/train_world_model.py primero: ese script "
            "genera el reward_scaler.json que comparten las tres arquitecturas."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(model, config: WorldModelConfig, path: Path) -> None:
    torch.save(model.state_dict(), path)
    hyperparams = {
        "architecture": "transformer",
        "latent_dim": config.latent_dim,
        "action_dim": config.action_dim,
        "sequence_length": config.sequence_length,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
    }
    path.with_suffix(".json").write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")


def main() -> None:
    _set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_latent_path = PROCESSED_DIR / "train_latent.npz"
    validation_latent_path = PROCESSED_DIR / "validation_latent.npz"
    for required in (train_latent_path, validation_latent_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run scripts/encode_latent_dataset.py first."
            )

    encoder_hparams_path = CHECKPOINT_DIR / "autoencoder_best.json"
    encoder_hparams = json.loads(encoder_hparams_path.read_text(encoding="utf-8"))
    representation = RepresentationConfig(**encoder_hparams)

    config = WorldModelConfig(representation=representation, sequence_length=16)

    reward_scaler = load_shared_reward_scaler(CHECKPOINT_DIR / "reward_scaler.json")
    reward_mean = torch.tensor(reward_scaler["reward_mean"], device=device)
    reward_std = torch.tensor(reward_scaler["reward_std"], device=device)

    train_dataset = LatentSequenceDataset(
        train_latent_path, sequence_length=config.sequence_length, action_dim=config.action_dim
    )
    validation_dataset = LatentSequenceDataset(
        validation_latent_path, sequence_length=config.sequence_length, action_dim=config.action_dim
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LatentDynamicsTransformer(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        sequence_length=config.sequence_length,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_losses, validation_losses = [], []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, reward_mean, reward_std)
        validation_loss = validate(model, validation_loader, device, reward_mean, reward_std)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_transformer_last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_transformer_best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={validation_loss:.6f}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping en época {epoch} (sin mejora en "
                f"{EARLY_STOPPING_PATIENCE} épocas)."
            )
            break

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (latent + reward)")
    plt.title("World model (Transformer) training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "world_model_transformer_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
