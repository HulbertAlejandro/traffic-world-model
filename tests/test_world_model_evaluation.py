from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.world_model_evaluation import (
    aggregate_by_horizon,
    build_world_model,
    load_episodes,
    rollout_episode,
)
from models.world_model import LatentDynamicsLSTM, LatentDynamicsTransformer, LatentDynamicsTSMixer


def _save_latent_dataset(path, episode_lengths, latent_dim=4, seed=0):
    """Write a synthetic encoded-latent .npz with the same schema as
    scripts/encode_latent_dataset.py produces."""
    rng = np.random.default_rng(seed)
    z, actions, rewards, episode_id, time_step = [], [], [], [], []

    for ep_id, length in enumerate(episode_lengths):
        z.append(rng.normal(size=(length, latent_dim)).astype(np.float32))
        actions.append(rng.integers(0, 2, size=length).astype(np.int64))
        rewards.append(rng.normal(size=length).astype(np.float32))
        episode_id.append(np.full(length, ep_id, dtype=np.int64))
        time_step.append(np.arange(length, dtype=np.int64))

    z_all = np.concatenate(z)
    np.savez(
        path,
        z=z_all,
        next_z=np.roll(z_all, -1, axis=0),  # not read by load_episodes, kept for schema realism
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        episode_id=np.concatenate(episode_id),
        time_step=np.concatenate(time_step),
        terminated=np.zeros(sum(episode_lengths), dtype=bool),
        truncated=np.zeros(sum(episode_lengths), dtype=bool),
    )


def test_load_episodes_groups_and_sorts(tmp_path):
    path = tmp_path / "test_latent.npz"
    _save_latent_dataset(path, episode_lengths=[5, 8])

    episodes = load_episodes(path)

    assert set(episodes) == {0, 1}
    assert episodes[0]["z"].shape == (5, 4)
    assert episodes[1]["z"].shape == (8, 4)
    for episode in episodes.values():
        assert episode["actions"].shape[0] == episode["z"].shape[0]
        assert episode["rewards"].shape[0] == episode["z"].shape[0]


def test_rollout_skips_episodes_shorter_than_window_plus_horizon():
    latent_dim, action_dim, sequence_length = 4, 2, 6
    model = LatentDynamicsLSTM(latent_dim=latent_dim, action_dim=action_dim, sequence_length=sequence_length)
    rng = np.random.default_rng(0)

    short_episode = {
        "z": rng.normal(size=(sequence_length, latent_dim)).astype(np.float32),  # too short: needs +horizon
        "actions": rng.integers(0, action_dim, size=sequence_length).astype(np.int64),
        "rewards": rng.normal(size=sequence_length).astype(np.float32),
    }

    records = rollout_episode(
        model, short_episode, sequence_length, action_dim, max_horizon=3, device=torch.device("cpu")
    )

    assert records == []


def test_rollout_produces_every_horizon_and_correct_number_of_starts():
    latent_dim, action_dim, sequence_length, max_horizon = 3, 2, 4, 5
    model = LatentDynamicsLSTM(latent_dim=latent_dim, action_dim=action_dim, sequence_length=sequence_length)
    rng = np.random.default_rng(1)

    episode_length = 15
    episode = {
        "z": rng.normal(size=(episode_length, latent_dim)).astype(np.float32),
        "actions": rng.integers(0, action_dim, size=episode_length).astype(np.int64),
        "rewards": rng.normal(size=episode_length).astype(np.float32),
    }

    records = rollout_episode(
        model, episode, sequence_length, action_dim, max_horizon, device=torch.device("cpu")
    )

    expected_starts = episode_length - sequence_length - max_horizon + 1
    assert expected_starts > 0
    assert len(records) == expected_starts * max_horizon

    horizons_seen = {r["horizon"] for r in records}
    assert horizons_seen == set(range(1, max_horizon + 1))

    for record in records:
        for key in ("model_latent_se", "model_reward_se", "baseline_latent_se", "baseline_reward_se"):
            assert record[key] >= 0.0


def test_horizon_one_matches_a_direct_forward_pass():
    """Sanity-checks the index convention: predicting horizon 1 from a window
    ending at `start + sequence_length` must be numerically identical to a
    single direct call to the model with that exact window -- no off-by-one
    drift in how the window or the target indices are built."""
    latent_dim, action_dim, sequence_length = 4, 2, 6
    model = LatentDynamicsLSTM(latent_dim=latent_dim, action_dim=action_dim, sequence_length=sequence_length)
    model.eval()
    rng = np.random.default_rng(2)

    episode_length = 10
    z = rng.normal(size=(episode_length, latent_dim)).astype(np.float32)
    actions = rng.integers(0, action_dim, size=episode_length).astype(np.int64)
    rewards = rng.normal(size=episode_length).astype(np.float32)
    episode = {"z": z, "actions": actions, "rewards": rewards}

    records = rollout_episode(
        model, episode, sequence_length, action_dim, max_horizon=1, device=torch.device("cpu")
    )

    start = 0
    import torch.nn.functional as F

    window_z = torch.from_numpy(z[start : start + sequence_length]).unsqueeze(0)
    window_actions = F.one_hot(
        torch.from_numpy(actions[start : start + sequence_length]), num_classes=action_dim
    ).float().unsqueeze(0)
    with torch.no_grad():
        expected_z, expected_r = model(window_z, window_actions)

    true_z = z[start + sequence_length]
    true_r = rewards[start + sequence_length - 1]
    expected_latent_se = float(np.sum((expected_z.squeeze(0).numpy() - true_z) ** 2))
    expected_reward_se = float((expected_r.item() - true_r) ** 2)

    assert records[0]["horizon"] == 1
    assert records[0]["model_latent_se"] == pytest_approx(expected_latent_se)
    assert records[0]["model_reward_se"] == pytest_approx(expected_reward_se)


def pytest_approx(value, rel=1e-5):
    import pytest

    return pytest.approx(value, rel=rel)


def test_baseline_reward_does_not_peek_at_its_own_target():
    """Regression test for a real bug caught during manual end-to-end testing:
    the persistence baseline for reward was initially defined at the same
    index as the horizon-1 target, so it trivially scored a perfect (and
    meaningless) 0.0 error at h=1 on every run. With continuous random
    rewards, a genuine baseline error is essentially never exactly zero."""
    latent_dim, action_dim, sequence_length = 4, 2, 6
    model = LatentDynamicsLSTM(latent_dim=latent_dim, action_dim=action_dim, sequence_length=sequence_length)
    rng = np.random.default_rng(3)

    episode_length = 20
    episode = {
        "z": rng.normal(size=(episode_length, latent_dim)).astype(np.float32),
        "actions": rng.integers(0, action_dim, size=episode_length).astype(np.int64),
        "rewards": rng.normal(size=episode_length).astype(np.float32),
    }

    records = rollout_episode(
        model, episode, sequence_length, action_dim, max_horizon=1, device=torch.device("cpu")
    )

    horizon_one_baseline_errors = [r["baseline_reward_se"] for r in records if r["horizon"] == 1]
    assert len(horizon_one_baseline_errors) > 1
    assert not all(error == 0.0 for error in horizon_one_baseline_errors)


def test_aggregate_by_horizon_computes_per_dimension_mse():
    latent_dim = 4
    records = [
        {"horizon": 1, "model_latent_se": 4.0, "model_reward_se": 1.0, "baseline_latent_se": 8.0, "baseline_reward_se": 2.0},
        {"horizon": 1, "model_latent_se": 0.0, "model_reward_se": 3.0, "baseline_latent_se": 4.0, "baseline_reward_se": 4.0},
        {"horizon": 2, "model_latent_se": 8.0, "model_reward_se": 2.0, "baseline_latent_se": 8.0, "baseline_reward_se": 2.0},
    ]

    summary = aggregate_by_horizon(records, latent_dim)

    assert set(summary) == {1, 2}
    assert summary[1]["n_samples"] == 2
    # (4/4 + 0/4) / 2 = 0.5
    assert summary[1]["model_latent_mse"] == pytest_approx(0.5)
    assert summary[1]["model_reward_mse"] == pytest_approx(2.0)
    assert summary[1]["baseline_latent_mse"] == pytest_approx(1.5)
    assert summary[2]["n_samples"] == 1


def test_build_world_model_defaults_to_lstm_for_legacy_checkpoints() -> None:
    # Checkpoints written before Experimento 3 have no "architecture" key.
    hparams = {"latent_dim": 4, "action_dim": 2, "sequence_length": 3, "hidden_dim": 8}
    assert isinstance(build_world_model(hparams), LatentDynamicsLSTM)


def test_build_world_model_dispatches_on_architecture() -> None:
    common = {"latent_dim": 4, "action_dim": 2, "sequence_length": 3, "dropout": 0.0}
    transformer = build_world_model(
        {**common, "architecture": "transformer", "d_model": 8, "nhead": 2,
         "num_layers": 1, "dim_feedforward": 16}
    )
    tsmixer = build_world_model(
        {**common, "architecture": "tsmixer", "hidden_dim": 8, "num_blocks": 1}
    )
    assert isinstance(transformer, LatentDynamicsTransformer)
    assert isinstance(tsmixer, LatentDynamicsTSMixer)


def test_build_world_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown world model architecture"):
        build_world_model({"architecture": "gru", "latent_dim": 4, "action_dim": 2})
