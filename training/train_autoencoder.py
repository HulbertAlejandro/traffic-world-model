"""Train the Autoencoder on normalized traffic states.

The Autoencoder learns only a compact representation of the traffic state.

It does NOT learn temporal dynamics.
It does NOT use actions.
It does NOT use rewards.

Those responsibilities belong to the temporal model implemented later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import RepresentationConfig
from datasets.transition_dataset import TransitionDataset
from models.representation import Autoencoder

DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"

TRAIN_DATASET = DATASET_DIR / "train.npz"
VALIDATION_DATASET = DATASET_DIR / "validation.npz"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_dataloader(
    dataset_path: Path,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader for one dataset split."""
    dataset = TransitionDataset(dataset_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(
    model: Autoencoder,
    dataloader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the Autoencoder for one epoch and return mean batch loss."""
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        states = batch[0].to(device)

        optimizer.zero_grad()

        _, reconstruction = model(states)
        loss = criterion(reconstruction, states)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: Autoencoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the Autoencoder without updating its parameters."""
    model.eval()
    running_loss = 0.0

    for batch in dataloader:
        states = batch[0].to(device)

        _, reconstruction = model(states)
        loss = criterion(reconstruction, states)

        running_loss += loss.item()

    return running_loss / len(dataloader)


def save_checkpoint(
    model: Autoencoder,
    output_path: Path,
    config: RepresentationConfig | None = None,
) -> None:
    """Save the model parameters and the config used to train it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    if config is not None:
        metadata = {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "activation": config.activation,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seed": config.seed,
            "checkpoint_dir": str(config.checkpoint_dir),
        }
        config_path = output_path.with_suffix(".json")
        config_path.write_text(json.dumps(metadata, indent=2))


def save_loss_curve(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save train and validation loss curves for the completed run."""
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axis.plot(epochs, history["train_loss"], label="Train Loss")
    axis.plot(epochs, history["validation_loss"], label="Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss")
    axis.set_title("Autoencoder reconstruction loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Train the Autoencoder using the normalized train/validation splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TransitionDataset(TRAIN_DATASET)
    input_dim = train_dataset.states.shape[1]
    config = RepresentationConfig(input_dim=input_dim, seed=0)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader = create_dataloader(
        TRAIN_DATASET,
        config.batch_size,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        VALIDATION_DATASET,
        config.batch_size,
        shuffle=False,
    )

    model = Autoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        activation=config.activation,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    best_validation_loss = float("inf")

    best_checkpoint = config.checkpoint_dir / "autoencoder_best.pt"
    last_checkpoint = config.checkpoint_dir / "autoencoder_last.pt"
    loss_curve = RESULTS_DIR / "autoencoder_loss.png"
    history = {"train_loss": [], "validation_loss": []}

    print()
    print("Starting training...")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1:03d}/{config.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Validation: {validation_loss:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        save_checkpoint(model, last_checkpoint, config=config)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, best_checkpoint, config=config)

    save_loss_curve(history, loss_curve)

    print()
    print("Training finished.")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best model saved to: {best_checkpoint}")
    print(f"Loss curve saved to: {loss_curve}")


if __name__ == "__main__":
    main()
