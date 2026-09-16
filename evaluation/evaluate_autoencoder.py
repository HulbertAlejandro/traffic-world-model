"""Evaluate the trained Autoencoder on the held-out test split."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transition_dataset import TransitionDataset
from evaluation.autoencoder_evaluation import (
    evaluate_reconstruction,
    load_autoencoder,
    plot_reconstruction_examples,
)

DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "test.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "autoencoder_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
EXAMPLES_PATH = RESULTS_DIR / "autoencoder_reconstructions.png"


def main() -> None:
    """Load the best checkpoint and report test reconstruction error."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TransitionDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    input_dim = dataset.states.shape[1]
    model = load_autoencoder(CHECKPOINT_PATH, input_dim, device)

    test_mse = evaluate_reconstruction(model, dataloader, device)
    plot_reconstruction_examples(model, dataset, EXAMPLES_PATH, device)

    print(f"Evaluation dataset: {DATASET_PATH}")
    print(f"Samples: {len(dataset)}")
    print(f"Reconstruction MSE: {test_mse:.6f}")
    print(f"Examples saved to: {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
