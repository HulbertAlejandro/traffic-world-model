from __future__ import annotations

import json
import pickle

import gymnasium as gym
import numpy as np
import torch

import environments.encoded_traffic_environment as encoded_module
from environments.encoded_traffic_environment import EncodedTrafficEnvironment
from models.representation import Autoencoder


INPUT_DIM = 26
HIDDEN_DIM = 8
LATENT_DIM = 4


def _make_checkpoint(tmp_path):
    model = Autoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    checkpoint_path = tmp_path / "autoencoder_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps({"input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM}),
        encoding="utf-8",
    )
    return checkpoint_path


def _make_scaler(tmp_path, state_mean, state_std):
    # Same structure written by scripts/normalize_dataset.py.
    scaler_path = tmp_path / "scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(
            {
                "state_mean": state_mean.reshape(1, -1).astype(np.float32),
                "state_std": state_std.reshape(1, -1).astype(np.float32),
            },
            handle,
        )
    return scaler_path


class _FakeTrafficEnvironment:
    """Stands in for TrafficEnvironment so the test never launches SUMO."""

    raw_state: np.ndarray

    def __init__(self, config=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(INPUT_DIM,), dtype=np.float32,
        )

    def reset(self, **kwargs):
        return self.raw_state.copy(), {}

    def step(self, action):
        return self.raw_state.copy(), 0.0, False, False, {}

    def close(self):
        pass


def test_state_is_normalized_with_scaler_before_encode(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    state_mean = rng.uniform(5.0, 50.0, size=INPUT_DIM).astype(np.float32)
    state_std = rng.uniform(2.0, 10.0, size=INPUT_DIM).astype(np.float32)
    expected_normalized = rng.normal(size=INPUT_DIM).astype(np.float32)
    expected_normalized -= expected_normalized.mean()
    raw_state = state_mean + state_std * expected_normalized

    _FakeTrafficEnvironment.raw_state = raw_state
    monkeypatch.setattr(encoded_module, "TrafficEnvironment", _FakeTrafficEnvironment)

    env = EncodedTrafficEnvironment(
        autoencoder_checkpoint=_make_checkpoint(tmp_path),
        scaler_path=_make_scaler(tmp_path, state_mean, state_std),
        device=torch.device("cpu"),
    )

    captured = []
    original_encode = env.autoencoder.encode

    def spy_encode(x):
        captured.append(x.detach().cpu().clone())
        return original_encode(x)

    monkeypatch.setattr(env.autoencoder, "encode", spy_encode)

    z, _ = env.reset()

    assert len(captured) == 1
    encoder_input = captured[0].squeeze(0).numpy()
    # The raw state sits far from zero (mean ~27); what reaches the Encoder
    # must be the scaler-normalized version, centred near zero.
    assert abs(raw_state.mean()) > 5.0
    assert abs(encoder_input.mean()) < 1e-4
    np.testing.assert_allclose(encoder_input, expected_normalized, atol=1e-4)
    assert z.shape == (LATENT_DIM,)
