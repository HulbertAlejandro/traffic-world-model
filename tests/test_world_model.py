from __future__ import annotations

import pytest
import torch

from configs import RepresentationConfig, WorldModelConfig
from models.world_model import LatentDynamicsLSTM


def test_world_model_latent_dim_tracks_representation_config() -> None:
    representation = RepresentationConfig(input_dim=26, latent_dim=8)
    world_model = WorldModelConfig(representation=representation)

    assert world_model.latent_dim == 8


def test_world_model_lstm_predicts_next_latent_state() -> None:
    representation = RepresentationConfig(input_dim=26, latent_dim=8)
    config = WorldModelConfig(representation=representation, sequence_length=12)
    model = LatentDynamicsLSTM(
        latent_dim=config.latent_dim,
        action_dim=3,
        hidden_dim=config.hidden_dim,
        sequence_length=config.sequence_length,
    )

    latent_sequence = torch.randn(4, config.sequence_length, config.latent_dim)
    action_sequence = torch.randn(4, config.sequence_length, 3)
    prediction = model(latent_sequence, action_sequence)

    assert prediction.shape == (4, config.latent_dim)


def test_lstm_rejects_invalid_action_dim() -> None:
    with pytest.raises(ValueError, match="action_dim must be positive"):
        LatentDynamicsLSTM(latent_dim=8, action_dim=0)


def test_lstm_rejects_mismatched_sequence_length() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=3, sequence_length=5)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(2, 5, 3)

    with pytest.raises(ValueError, match="same sequence_length"):
        model(latent_sequence, action_sequence)


def test_lstm_rejects_mismatched_batch_size() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=3)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(3, 4, 3)

    with pytest.raises(ValueError, match="same batch size"):
        model(latent_sequence, action_sequence)


def test_lstm_rejects_invalid_action_tensor_dimensions() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=3)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(2, 4)

    with pytest.raises(ValueError, match="action_sequence must have shape"):
        model(latent_sequence, action_sequence)
