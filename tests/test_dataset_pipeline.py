from __future__ import annotations

import numpy as np

from datasets.transition_dataset import TransitionDataset
from scripts.merge_dataset import merge_files
from scripts.normalize_dataset import normalize_dataset
from scripts.split_dataset import split_dataset


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


def test_split_dataset_does_not_mix_episodes(tmp_path):
    """Regression test for the train/validation/test leakage bug.

    Calls the REAL ``split_dataset()`` end-to-end instead of manually
    pre-partitioning episodes, so a future regression in its assignment
    logic (e.g. reverting to a flat-index cut over pooled transitions)
    would actually be caught here.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for episode_id in range(10):
        _make_episode(raw_dir / f"episode_{episode_id:03d}.npz", episode_id=episode_id)

    output_dir = tmp_path / "processed"
    saved_paths = split_dataset(
        raw_dir=raw_dir,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        output_dir=output_dir,
        seed=0,
    )

    episode_ids_per_split = {}
    for name, path in saved_paths.items():
        with np.load(path) as data:
            episode_ids_per_split[name] = set(data["episode_id"].tolist())

    all_splits = list(episode_ids_per_split.items())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            name_a, ids_a = all_splits[i]
            name_b, ids_b = all_splits[j]
            assert ids_a.isdisjoint(ids_b), (
                f"Episode overlap detected between '{name_a}' and '{name_b}': "
                f"{ids_a & ids_b}"
            )

    total_ids = set().union(*episode_ids_per_split.values())
    assert total_ids == set(range(10))