from __future__ import annotations

import numpy as np

from datasets.transition_dataset import TransitionDataset


def test_transition_dataset_loads_numpy_arrays(tmp_path):
    data = {
        "states": np.arange(20, dtype=np.float32).reshape(10, 2),
        "actions": np.arange(10, dtype=np.float32),
        "rewards": np.arange(10, dtype=np.float32),
        "next_states": np.arange(20, dtype=np.float32).reshape(10, 2),
    }
    dataset_path = tmp_path / "sample.npz"
    np.savez(dataset_path, **data)

    dataset = TransitionDataset(dataset_path)

    assert len(dataset) == 10
    assert dataset[0][0].shape == (2,)
    assert dataset[0][1].shape == ()
    assert dataset[0][2].shape == ()
    assert dataset[0][3].shape == (2,)
