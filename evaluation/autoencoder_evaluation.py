"""Reusable evaluation utilities for the trained traffic-state Autoencoder."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import RepresentationConfig
from datasets.transition_dataset import TransitionDataset
from models.representation import Autoencoder


def load_autoencoder(
    checkpoint_path: str | Path,
    input_dim: int,
    device: torch.device,
) -> Autoencoder:
    """Load an Autoencoder checkpoint in evaluation mode."""
    config = RepresentationConfig(input_dim=input_dim)
    model = Autoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_reconstruction(
    model: Autoencoder,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Return the mean reconstruction MSE over a dataset split."""
    model.eval()
    total_squared_error = 0.0
    total_values = 0

    for batch in dataloader:
        states = batch[0].to(device)
        reconstruction = model.reconstruct(states)
        total_squared_error += torch.sum((reconstruction - states) ** 2).item()
        total_values += states.numel()

    if total_values == 0:
        raise ValueError("Cannot evaluate reconstruction on an empty dataset.")

    return total_squared_error / total_values


@torch.no_grad()
def encode_dataset(
    model: Autoencoder,
    dataset: TransitionDataset,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode all states in a TransitionDataset into one latent tensor."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_batches = []
    model.eval()

    for batch in loader:
        latent_batches.append(model.encode(batch[0].to(device)).cpu())

    if not latent_batches:
        raise ValueError("Cannot encode an empty dataset.")

    return torch.cat(latent_batches, dim=0)


def plot_reconstruction_examples(
    model: Autoencoder,
    dataset: TransitionDataset,
    output_path: str | Path,
    device: torch.device,
    num_examples: int = 3,
) -> None:
    """Plot original and reconstructed state vectors for selected samples."""
    if len(dataset) == 0:
        raise ValueError("Cannot plot reconstruction examples from an empty dataset.")

    num_examples = min(num_examples, len(dataset))
    states = dataset.states[:num_examples].to(device)
    with torch.no_grad():
        reconstructions = model.reconstruct(states).cpu()
    states = states.cpu()

    figure, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples), squeeze=False)
    for index in range(num_examples):
        axis = axes[index, 0]
        axis.plot(states[index].numpy(), label="Original", marker="o", markersize=3)
        axis.plot(
            reconstructions[index].numpy(),
            label="Reconstruido",
            marker="x",
            markersize=3,
        )
        error = torch.mean((states[index] - reconstructions[index]) ** 2).item()
        axis.set_title(f"Ejemplo {index + 1} | MSE: {error:.6f}")
        axis.set_xlabel("Componente del estado")
        axis.set_ylabel("Valor normalizado")
        axis.legend()
        axis.grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
