"""Training loop for the latent dynamics model (LatentDynamicsLSTM).

Mirrors train_autoencoder.py: fixed seeds, checkpoint + hyperparameter
persistence, and a loss curve. The encoder used to produce the latent
dataset is NOT touched here -- it was already frozen when
scripts/encode_latent_dataset.py ran.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import RepresentationConfig, WorldModelConfig
from datasets.latent_sequence_dataset import LatentSequenceDataset
from models.world_model import LatentDynamicsLSTM

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REWARD_LOSS_WEIGHT = 1.0
WEIGHT_DECAY = 1e-4


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_reward_scaler(train_latent_path: Path) -> dict[str, float]:
    """Fit reward mean/std using ONLY the training split, same principle
    already used for state normalization in scripts/normalize_dataset.py."""
    with np.load(train_latent_path) as data:
        rewards = data["rewards"].astype(np.float32)
    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std())
    if reward_std < 1e-8:
        reward_std = 1.0
    return {"reward_mean": reward_mean, "reward_std": reward_std}


def save_reward_scaler(scaler: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def train_one_epoch(model, loader, optimizer, device, reward_mean, reward_std) -> float:
    model.train()
    total_loss = 0.0
    for latent_window, action_window, target_latent, target_reward, _, _ in loader:
        latent_window = latent_window.to(device)
        action_window = action_window.to(device)
        target_latent = target_latent.to(device)
        target_reward = target_reward.to(device)
        target_reward_normalized = (target_reward - reward_mean) / reward_std

        optimizer.zero_grad()
        pred_latent, pred_reward = model(latent_window, action_window)
        loss = nn.functional.mse_loss(pred_latent, target_latent) + REWARD_LOSS_WEIGHT * nn.functional.mse_loss(
            pred_reward, target_reward_normalized
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * latent_window.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, reward_mean, reward_std) -> float:
    model.eval()
    total_loss = 0.0
    for latent_window, action_window, target_latent, target_reward, _, _ in loader:
        latent_window = latent_window.to(device)
        action_window = action_window.to(device)
        target_latent = target_latent.to(device)
        target_reward = target_reward.to(device)
        target_reward_normalized = (target_reward - reward_mean) / reward_std

        pred_latent, pred_reward = model(latent_window, action_window)
        loss = nn.functional.mse_loss(pred_latent, target_latent) + REWARD_LOSS_WEIGHT * nn.functional.mse_loss(
            pred_reward, target_reward_normalized
        )
        total_loss += loss.item() * latent_window.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(model, config: WorldModelConfig, path: Path) -> None:
    torch.save(model.state_dict(), path)
    hyperparams = {
        "latent_dim": config.latent_dim,
        "action_dim": config.action_dim,
        "sequence_length": config.sequence_length,
        "hidden_dim": config.hidden_dim,
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

    reward_scaler = compute_reward_scaler(train_latent_path)
    save_reward_scaler(reward_scaler, CHECKPOINT_DIR / "reward_scaler.json")
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

    model = LatentDynamicsLSTM(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        sequence_length=config.sequence_length,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses, validation_losses = [], []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, reward_mean, reward_std)
        validation_loss = validate(model, validation_loader, device, reward_mean, reward_std)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_best.pt")
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
    plt.title("World model (LSTM) training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "world_model_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
