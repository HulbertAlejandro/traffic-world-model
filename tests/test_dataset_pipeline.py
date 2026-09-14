from __future__ import annotations

import numpy as np

from datasets.transition_dataset import TransitionDataset
from scripts.merge_dataset import merge_files
from scripts.normalize_dataset import normalize_dataset


def _make_episode(path, episode_id: int, num_transitions: int = 10, state_dim: int = 4):
    rng = np.random.default_rng(episode_id)
    np.savez(
        path,
        states=rng.normal(size=(num_transitions, state_dim)).astype(np.float32),
        actions=rng.integers(0, 2, size=num_transitions).astype(np.int64),
        rewards=rng.normal(size=num_transitions).astype(np.float32),
        next_states=rng.normal(size=(num_transitions, state_dim)).astype(np.float32),
        terminated=np.zeros(num_transitions, dtype=bool),
        truncated=np.array([False] * (num_transitions - 1) + [True], dtype=bool),
        episode_id=np.full(num_transitions, episode_id, dtype=np.int64),
        time_step=np.arange(num_transitions, dtype=np.int64),
    )


def test_transition_dataset_returns_full_contract(tmp_path):
    raw_path = tmp_path / "episode_000.npz"
    _make_episode(raw_path, episode_id=0)

    merged_path = tmp_path / "train_raw.npz"
    merge_files([raw_path], merged_path)

    scaler_path = tmp_path / "scaler.pkl"
    saved, _ = normalize_dataset(
        train_path=merged_path,
        output_dir=tmp_path,
        scaler_path=scaler_path,
    )

    dataset = TransitionDataset(saved["train"])

    assert len(dataset) == 10
    state, action, reward, next_state, episode_id, time_step, terminated, truncated = dataset[0]
    assert state.shape == (4,)
    assert next_state.shape == (4,)
    assert int(episode_id) == 0
    assert int(time_step) == 0
    assert bool(terminated) is False


def test_split_does_not_mix_episodes(tmp_path):
    """Regression test for the train/validation/test leakage bug: no episode
    id should ever appear in more than one merged split."""
    episode_paths = []
    for episode_id in range(6):
        path = tmp_path / f"episode_{episode_id:03d}.npz"
        _make_episode(path, episode_id=episode_id)
        episode_paths.append(path)

    group_a = episode_paths[:3]
    group_b = episode_paths[3:]

    merge_files(group_a, tmp_path / "train_raw.npz")
    merge_files(group_b, tmp_path / "test_raw.npz")

    with np.load(tmp_path / "train_raw.npz") as train_data, np.load(tmp_path / "test_raw.npz") as test_data:
        train_ids = set(train_data["episode_id"].tolist())
        test_ids = set(test_data["episode_id"].tolist())

    assert train_ids.isdisjoint(test_ids)