"""train_controller_direct.py must never overwrite a previous result by accident."""

import os

import pytest

from training.train_controller_direct import (
    CHECKPOINT_DIR,
    CONTROLLER_DIRECT_DIR,
    DIRECT_TOTAL_TIMESTEPS,
    check_output_dir,
    parse_args,
)


def test_defaults_match_the_official_run():
    args = parse_args([])
    assert args.seed == 0
    assert args.total_timesteps == DIRECT_TOTAL_TIMESTEPS == 10_000
    assert args.output_dir == CONTROLLER_DIRECT_DIR
    assert not args.overwrite and not args.overwrite_official


@pytest.mark.parametrize("folder", ["controller_direct", "controller_direct_30k"])
def test_official_folders_are_refused_even_with_overwrite(folder):
    with pytest.raises(SystemExit):
        check_output_dir(CHECKPOINT_DIR / folder, overwrite=True, overwrite_official=False)


def test_official_folder_reached_through_a_relative_path_is_refused():
    relative = os.path.relpath(CONTROLLER_DIRECT_DIR / ".." / "controller_direct", os.getcwd())
    with pytest.raises(SystemExit):
        check_output_dir(relative, overwrite=False, overwrite_official=False)


def test_official_folder_needs_its_own_explicit_flag():
    assert check_output_dir(
        CHECKPOINT_DIR / "controller_direct_30k", overwrite=True, overwrite_official=True
    ) == (CHECKPOINT_DIR / "controller_direct_30k").resolve()


def test_folder_with_a_previous_result_is_refused_without_overwrite(tmp_path):
    (tmp_path / "best_model.zip").write_bytes(b"fake")
    with pytest.raises(SystemExit):
        check_output_dir(tmp_path, overwrite=False, overwrite_official=False)
    assert check_output_dir(tmp_path, overwrite=True, overwrite_official=False) == tmp_path.resolve()


def test_new_or_empty_folder_is_accepted(tmp_path):
    assert check_output_dir(tmp_path / "seed0", overwrite=False, overwrite_official=False) == (
        tmp_path / "seed0"
    ).resolve()
