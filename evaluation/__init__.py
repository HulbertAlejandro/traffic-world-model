"""Evaluation utilities for trained project models."""

from .autoencoder_evaluation import (
    encode_dataset,
    evaluate_reconstruction,
    load_autoencoder,
    plot_reconstruction_examples,
)

__all__ = [
    "encode_dataset",
    "evaluate_reconstruction",
    "load_autoencoder",
    "plot_reconstruction_examples",
]
