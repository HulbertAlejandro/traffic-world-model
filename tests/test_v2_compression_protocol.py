"""v2 Phase 2: the compressed (z) and uncompressed (raw) branches must train the LSTM under the
SAME protocol, differing only in the input dimension (docs/v2/ADDENDUM_AUTOENCODER.md, section 1).

In v1 the raw branch once trained without weight decay and early stopping because its loop was a
copy of the z branch's. Here both branches go through training.v2_compression_experiment.train_lstm
with the same LSTMProtocol object; these tests keep it that way.
"""

import dataclasses
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import training.train_world_model as v1_lstm
import training.v2_compression_experiment as exp


def test_both_branches_call_the_same_training_function_with_the_same_protocol(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(exp, "train_lstm", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    exp.train_branch("z", Path("t"), Path("v"), 3, tmp_path / "z", latent_dim=32)
    exp.train_branch("raw", Path("t"), Path("v"), 3, tmp_path / "raw")
    (z_args, z_kwargs), (raw_args, raw_kwargs) = calls
    # (train_seq, val_seq, input_dim, seed, out_dir, protocol)
    assert z_args[2] == 32 and raw_args[2] == exp.STATE_DIM
    assert z_args[5] is raw_args[5] is exp.LSTM_PROTOCOL
    assert z_args[0] == raw_args[0] and z_args[1] == raw_args[1] and z_args[3] == raw_args[3]


def test_train_lstm_applies_weight_decay_and_early_stopping_from_the_protocol():
    source = inspect.getsource(exp.train_lstm)
    assert "weight_decay=protocol.weight_decay" in source
    assert "since >= protocol.early_stopping_patience" in source
    assert "v1_lstm.train_one_epoch" in source and "v1_lstm.validate" in source
    assert "v1_lstm.compute_reward_scaler(train_seq)" in source


def test_protocol_values_are_v1_experiment_0_redo():
    p = exp.LSTM_PROTOCOL
    assert (p.learning_rate, p.weight_decay, p.batch_size, p.early_stopping_patience, p.reward_loss_weight) == (
        v1_lstm.LEARNING_RATE, v1_lstm.WEIGHT_DECAY, v1_lstm.BATCH_SIZE, v1_lstm.EARLY_STOPPING_PATIENCE,
        v1_lstm.REWARD_LOSS_WEIGHT)
    assert p.weight_decay > 0 and p.max_epochs == 300 and p.optimizer == "adam"


def _synthetic_split(path: Path, dim: int, episodes: int, rng) -> None:
    n = 20
    z = rng.normal(size=(episodes * n, dim)).astype(np.float32)
    np.savez(path, z=z, next_z=np.roll(z, -1, axis=0), actions=rng.integers(0, 2, (episodes * n, 4)),
             rewards=rng.normal(-40, 10, episodes * n).astype(np.float32),
             episode_id=np.repeat(np.arange(episodes), n), time_step=np.tile(np.arange(n), episodes))


def test_tiny_real_runs_of_both_branches_record_identical_protocols(tmp_path):
    rng = np.random.default_rng(0)
    tiny = dataclasses.replace(exp.LSTM_PROTOCOL, max_epochs=40, early_stopping_patience=2, hidden_dim=8,
                               sequence_length=4)
    runs = {}
    for branch, dim in (("z", 16), ("raw", exp.STATE_DIM)):
        _synthetic_split(tmp_path / f"{branch}_train.npz", dim, 3, rng)
        _synthetic_split(tmp_path / f"{branch}_val.npz", dim, 2, rng)
        runs[branch] = exp.train_branch(branch, tmp_path / f"{branch}_train.npz", tmp_path / f"{branch}_val.npz",
                                        0, tmp_path / branch, latent_dim=16 if branch == "z" else None,
                                        protocol=tiny)
        saved = json.loads((tmp_path / branch / "world_model_best.json").read_text(encoding="utf-8"))
        assert saved["protocol"] == dataclasses.asdict(tiny)
        # Noise targets: validation stops improving quickly, so early stopping must end the run.
        assert saved["stopped_epoch"] < tiny.max_epochs and not saved["hit_max_epochs"]
    z, raw = runs["z"], runs["raw"]
    assert z["protocol"] == raw["protocol"]
    differing = {k for k in z if k in raw and z[k] != raw[k]}
    assert differing <= {"latent_dim", "best_epoch", "stopped_epoch", "best_validation_loss"}
    assert z["latent_dim"] == 16 and raw["latent_dim"] == exp.STATE_DIM
