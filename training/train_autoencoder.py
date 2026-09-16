"""Train the Autoencoder on normalized traffic states.

The Autoencoder learns only a compact representation of the traffic state.

It does NOT learn temporal dynamics.
It does NOT use actions.
It does NOT use rewards.

Those responsibilities belong to the temporal model implemented later.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs import RepresentationConfig
from datasets.transition_dataset import TransitionDataset
from models.representation import Autoencoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"

TRAIN_DATASET = DATASET_DIR / "train.npz"
VALIDATION_DATASET = DATASET_DIR / "validation.npz"


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
) -> None:
    """Save the model parameters to a checkpoint path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def main() -> None:
    """Train the Autoencoder using the normalized train/validation splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TransitionDataset(TRAIN_DATASET)
    input_dim = train_dataset.states.shape[1]
    config = RepresentationConfig(input_dim=input_dim)

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
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    best_validation_loss = float("inf")

    best_checkpoint = config.checkpoint_dir / "autoencoder_best.pt"
    last_checkpoint = config.checkpoint_dir / "autoencoder_last.pt"

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

        save_checkpoint(model, last_checkpoint)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, best_checkpoint)

    print()
    print("Training finished.")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best model saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
