"""The raw-state Transformer/TSMixer trainers must use exactly their latent counterparts' protocol.

Same guarantee as test_experiment_0_protocol.py for the LSTM: the training protocol comes from
train_world_model.py and the architecture hyperparameters and shared reward scaler from the
official Experimento 3 scripts, imported, never copied.
"""

import inspect

import pytest

import training.train_world_model as protocol
import training.train_world_model_transformer as transformer
import training.train_world_model_transformer_raw as transformer_raw
import training.train_world_model_tsmixer as tsmixer
import training.train_world_model_tsmixer_raw as tsmixer_raw

PROTOCOL_FUNCTIONS = ("_set_seeds", "train_one_epoch", "validate")
PROTOCOL_CONSTANTS = ("SEED", "LEARNING_RATE", "BATCH_SIZE", "EPOCHS", "EARLY_STOPPING_PATIENCE", "WEIGHT_DECAY")
PAIRS = [
    (transformer_raw, transformer, ("D_MODEL", "NHEAD", "NUM_LAYERS", "DIM_FEEDFORWARD", "DROPOUT")),
    (tsmixer_raw, tsmixer, ("NUM_BLOCKS", "DROPOUT")),
]


@pytest.mark.parametrize("raw, latent, arch_constants", PAIRS)
def test_raw_trainer_imports_the_same_protocol_and_architecture(raw, latent, arch_constants):
    for name in PROTOCOL_FUNCTIONS:
        assert getattr(raw, name) is getattr(protocol, name), name
    for name in PROTOCOL_CONSTANTS:
        assert getattr(raw, name) == getattr(protocol, name), name
    for name in arch_constants:
        assert getattr(raw, name) == getattr(latent, name), name
    assert raw.load_shared_reward_scaler is latent.load_shared_reward_scaler


@pytest.mark.parametrize("raw", [transformer_raw, tsmixer_raw])
def test_raw_trainer_applies_weight_decay_and_early_stopping(raw):
    source = inspect.getsource(raw.train)
    assert "weight_decay=WEIGHT_DECAY" in source
    assert "epochs_without_improvement >= EARLY_STOPPING_PATIENCE" in source


@pytest.mark.parametrize("raw", [transformer_raw, tsmixer_raw])
def test_raw_trainer_refuses_the_official_checkpoint_folder(raw, tmp_path):
    with pytest.raises(SystemExit):
        raw.main(["--output-dir", str(protocol.CHECKPOINT_DIR), "--data-dir", str(tmp_path)])
