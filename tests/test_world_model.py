from __future__ import annotations

import pytest
import torch

from configs import RepresentationConfig, WorldModelConfig
from models.world_model import LatentDynamicsLSTM


def _make_config(latent_dim: int = 8, sequence_length: int = 12) -> WorldModelConfig:
    representation = RepresentationConfig(input_dim=26, latent_dim=latent_dim)
    return WorldModelConfig(representation=representation, sequence_length=sequence_length)


def test_world_model_latent_dim_tracks_representation_config() -> None:
    config = _make_config(latent_dim=8)
    assert config.latent_dim == 8


def test_world_model_default_action_dim_is_two() -> None:
    config = _make_config()
    assert config.action_dim == 2


def test_world_model_lstm_predicts_next_latent_and_reward() -> None:
    config = _make_config()
    model = LatentDynamicsLSTM(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        sequence_length=config.sequence_length,
    )

    latent_sequence = torch.randn(4, config.sequence_length, config.latent_dim)
    action_sequence = torch.randn(4, config.sequence_length, config.action_dim)
    next_latent, next_reward = model(latent_sequence, action_sequence)

    assert next_latent.shape == (4, config.latent_dim)
    assert next_reward.shape == (4,)


def test_lstm_rejects_invalid_action_dim() -> None:
    with pytest.raises(ValueError, match="action_dim must be positive"):
        LatentDynamicsLSTM(latent_dim=8, action_dim=0)


def test_lstm_rejects_mismatched_sequence_length() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=2, sequence_length=5)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(2, 5, 2)

    with pytest.raises(ValueError, match="same sequence_length"):
        model(latent_sequence, action_sequence)


def test_lstm_rejects_mismatched_batch_size() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=2)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(3, 4, 2)

    with pytest.raises(ValueError, match="same batch size"):
        model(latent_sequence, action_sequence)


def test_lstm_rejects_invalid_action_tensor_dimensions() -> None:
    model = LatentDynamicsLSTM(latent_dim=8, action_dim=2)
    latent_sequence = torch.randn(2, 4, 8)
    action_sequence = torch.randn(2, 4)

    with pytest.raises(ValueError, match="action_sequence must have shape"):
        model(latent_sequence, action_sequence)
