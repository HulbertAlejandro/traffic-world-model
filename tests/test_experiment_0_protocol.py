"""Both branches of Experimento 0 (z and raw state) must train under the same protocol.

The raw branch once copied the training loop from train_world_model.py and was
left without weight_decay and early stopping when those were added to the z
branch, which made the published comparison favor the Autoencoder. The raw
script now imports the protocol; these tests keep it that way.
"""

import inspect

import training.train_world_model as z_branch
import training.train_world_model_raw as raw_branch

PROTOCOL_CONSTANTS = ("SEED", "LEARNING_RATE", "BATCH_SIZE", "EPOCHS", "EARLY_STOPPING_PATIENCE",
                      "WEIGHT_DECAY", "REWARD_LOSS_WEIGHT")
PROTOCOL_FUNCTIONS = ("_set_seeds", "train_one_epoch", "validate", "compute_reward_scaler")


def test_raw_branch_imports_the_same_protocol_objects():
    for name in PROTOCOL_CONSTANTS[:-1]:
        assert getattr(raw_branch, name) == getattr(z_branch, name), name
    # REWARD_LOSS_WEIGHT is used inside train_one_epoch/validate, which are shared below.
    for name in PROTOCOL_FUNCTIONS:
        assert getattr(raw_branch, name) is getattr(z_branch, name), name


def test_both_branches_apply_weight_decay_and_early_stopping():
    for module in (z_branch, raw_branch):
        source = inspect.getsource(module.main)
        assert "weight_decay=WEIGHT_DECAY" in source, module.__name__
        assert "epochs_without_improvement >= EARLY_STOPPING_PATIENCE" in source, module.__name__
