from __future__ import annotations

import numpy as np

from datasets.latent_sequence_dataset import LatentSequenceDataset


def _make_latent_dataset(path, episode_lengths, latent_dim=4):
    z, next_z, actions, rewards, episode_id, time_step = [], [], [], [], [], []
    rng = np.random.default_rng(0)

    for ep_id, length in enumerate(episode_lengths):
        episode_z = rng.normal(size=(length, latent_dim)).astype(np.float32)
        z.append(episode_z)
        next_z.append(np.roll(episode_z, -1, axis=0))
        actions.append(rng.integers(0, 2, size=length).astype(np.int64))
        rewards.append(rng.normal(size=length).astype(np.float32))
        episode_id.append(np.full(length, ep_id, dtype=np.int64))
        time_step.append(np.arange(length, dtype=np.int64))

    np.savez(
        path,
        z=np.concatenate(z),
        next_z=np.concatenate(next_z),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        episode_id=np.concatenate(episode_id),
        time_step=np.concatenate(time_step),
        terminated=np.zeros(sum(episode_lengths), dtype=bool),
        truncated=np.zeros(sum(episode_lengths), dtype=bool),
    )


def test_windows_never_cross_episode_boundaries(tmp_path):
    path = tmp_path / "train_latent.npz"
    _make_latent_dataset(path, episode_lengths=[10, 10])

    dataset = LatentSequenceDataset(path, sequence_length=4, action_dim=2)

    for idx in range(len(dataset)):
        _, _, _, _, episode_id, start = dataset._windows[idx]
        assert 0 <= start <= 10 - 4

    episode_ids_seen = {dataset._windows[i][4] for i in range(len(dataset))}
    assert episode_ids_seen == {0, 1}


def test_short_episodes_are_discarded(tmp_path):
    path = tmp_path / "train_latent.npz"
    _make_latent_dataset(path, episode_lengths=[3, 10])

    dataset = LatentSequenceDataset(path, sequence_length=4, action_dim=2)

    episode_ids_seen = {dataset._windows[i][4] for i in range(len(dataset))}
    assert episode_ids_seen == {1}


def test_item_shapes(tmp_path):
    path = tmp_path / "train_latent.npz"
    _make_latent_dataset(path, episode_lengths=[10])

    dataset = LatentSequenceDataset(path, sequence_length=4, action_dim=2)
    latent_window, action_window, target_latent, target_reward, _, _ = dataset[0]

    assert latent_window.shape == (4, 4)
    assert action_window.shape == (4, 2)
    assert target_latent.shape == (4,)
    assert target_reward.shape == ()
