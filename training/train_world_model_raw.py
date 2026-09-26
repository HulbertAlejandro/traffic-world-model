"""Training loop for the latent dynamics model (LatentDynamicsLSTM) on raw state.

Experimento 0: same model and same training protocol as
training/train_world_model.py, except the input representation. Instead of the
Autoencoder's learned z, the LSTM is fed the raw normalized state (26 dims)
directly, via the *_raw_seq.npz files produced by
scripts/prepare_raw_sequence_dataset.py.

The protocol (seed, learning rate, batch size, epochs, weight decay, early
stopping, loss and reward normalization) is IMPORTED from train_world_model.py
rather than copied, the same pattern the Transformer and TSMixer scripts use, so
the two branches of the experiment cannot drift apart. An earlier version copied
the loop and was left without weight_decay and early stopping when those were
added to the z branch.

The reward scaler is fitted on this branch's own training split (same rewards
as the z branch) and saved under models/checkpoints/raw_state/, so that
evaluation.world_model_evaluation.load_reward_scaler() never mixes the two.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import WorldModelConfig
from datasets.latent_sequence_dataset import LatentSequenceDataset
from models.world_model import LatentDynamicsLSTM
from training.train_world_model import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    _set_seeds,
    compute_reward_scaler,
    save_reward_scaler,
    train_one_epoch,
    validate,
)

# Separate subfolder from the z-based experiment's checkpoints: it keeps this
# experiment's reward_scaler.json from being picked up by
# evaluation.world_model_evaluation.load_reward_scaler(), which looks for a
# fixed filename next to the checkpoint it is given.
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints" / "raw_state"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 16
ACTION_DIM = 2


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

    train_path = PROCESSED_DIR / "train_raw_seq.npz"
    validation_path = PROCESSED_DIR / "validation_raw_seq.npz"
    for required in (train_path, validation_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run scripts/prepare_raw_sequence_dataset.py first."
            )

    with np.load(train_path) as data:
        raw_state_dim = data["z"].shape[1]

    # hidden_dim comes from WorldModelConfig's own default (128), same value
    # the z-based experiment uses -- so the only real difference between the
    # two experiments is the input representation, not the model's capacity.
    config = WorldModelConfig(latent_dim=raw_state_dim, action_dim=ACTION_DIM, sequence_length=SEQUENCE_LENGTH)

    reward_scaler = compute_reward_scaler(train_path)
    save_reward_scaler(reward_scaler, CHECKPOINT_DIR / "reward_scaler.json")
    reward_mean = torch.tensor(reward_scaler["reward_mean"], device=device)
    reward_std = torch.tensor(reward_scaler["reward_std"], device=device)

    train_dataset = LatentSequenceDataset(
        train_path, sequence_length=config.sequence_length, action_dim=config.action_dim
    )
    validation_dataset = LatentSequenceDataset(
        validation_path, sequence_length=config.sequence_length, action_dim=config.action_dim
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

        save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_raw_last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, config, CHECKPOINT_DIR / "world_model_raw_best.pt")
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
    plt.title("World model (LSTM on raw state) training loss -- Experimento 0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "world_model_raw_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
