from __future__ import annotations

import json

import torch

from configs import RepresentationConfig
from evaluation.autoencoder_evaluation import load_autoencoder
from models.representation import Autoencoder
from training.train_autoencoder import save_checkpoint


def test_autoencoder_respects_configured_activation() -> None:
    config = RepresentationConfig(
        input_dim=10,
        hidden_dim=8,
        latent_dim=4,
        activation="tanh",
    )
    model = Autoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        activation=config.activation,
    )
    x = torch.randn(2, config.input_dim)
    z, reconstruction = model(x)

    assert z.shape == (2, config.latent_dim)
    assert reconstruction.shape == x.shape


def test_checkpoint_keeps_hyperparameters(tmp_path) -> None:
    config = RepresentationConfig(
        input_dim=10,
        hidden_dim=8,
        latent_dim=4,
        seed=123,
    )
    model = Autoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        activation=config.activation,
    )
    checkpoint_path = tmp_path / "autoencoder_best.pt"

    save_checkpoint(model, checkpoint_path, config=config)

    assert checkpoint_path.exists()
    metadata_path = checkpoint_path.with_suffix(".json")
    assert metadata_path.exists()

    payload = json.loads(metadata_path.read_text())
    assert payload["latent_dim"] == config.latent_dim
    assert payload["seed"] == config.seed

    device = torch.device("cpu")
    loaded = load_autoencoder(checkpoint_path, config.input_dim, device)
    assert loaded is not None
