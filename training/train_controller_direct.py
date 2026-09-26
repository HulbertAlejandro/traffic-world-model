"""Train a PPO controller DIRECTLY against real SUMO -- the "direct RL"
baseline (Sección 18 of the proposal), no Autoencoder, no Dream Environment.

This answers the project's core research question: the Dream-trained PPO
(training/train_controller.py) needed ZERO real SUMO timesteps to train.
This script measures what a directly-trained policy achieves with a
comparatively modest real-interaction budget -- intentionally smaller than
the Dream-trained model's 50,176 imagined timesteps, since real SUMO steps
are far more expensive than imagined ones. This asymmetry in budget IS the
point being measured, not a flaw in the comparison.

Every episode (training and periodic evaluation) uses a fresh or fixed SUMO
seed via ReseedingWrapper -- without it, sumo_rl reuses the previous seed on
every reset() that doesn't specify one, and this policy would train and be
evaluated against a single traffic scenario. See environments/reseeding_wrapper.py.

Usage (defaults reproduce the official run: seed 0, 10,000 steps):
    python training/train_controller_direct.py --output-dir models/checkpoints/controller_direct_fixed/seed0
    python training/train_controller_direct.py --seed 1 --total-timesteps 30000 --output-dir <folder>

The run refuses to write into a folder that already holds any file it would
produce, and into the official folders (controller_direct/, controller_direct_30k/)
at all, unless the matching flag is passed explicitly (--overwrite,
--overwrite-official). Running it with no arguments therefore stops instead of
replacing the official checkpoint.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ControllerConfig
from environments.reseeding_wrapper import ReseedingWrapper
from environments.traffic_environment import TrafficEnvironment
from training.train_controller import (
    SaveVecNormalizeOnBest,
    build_normalized_envs,
    save_hyperparameters,
    vecnormalize_path,
)

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIRECT_DIR = CHECKPOINT_DIR / "controller_direct"
# Official / archived result folders: never written without --overwrite-official.
PROTECTED_DIRS = (CONTROLLER_DIRECT_DIR, CHECKPOINT_DIR / "controller_direct_30k")
# Every file this script (and its EvalCallback) writes into the output folder.
OUTPUT_FILES = (
    "best_model.zip",
    "best_model.json",
    "best_model_vecnormalize.pkl",
    "evaluations.npz",
    "ppo_controller_direct_final.zip",
    "ppo_controller_direct_final.json",
    "ppo_controller_direct_final_vecnormalize.pkl",
)

# Deliberately smaller than ControllerConfig.total_timesteps (50_000, used for
# Dream-Environment training): real SUMO steps are far more expensive than
# imagined ones, and the Dream-trained policy used ZERO real timesteps to
# train. This budget represents a "reasonably modest" direct-training
# investment in real SUMO time, for a fair sample-efficiency comparison --
# not an attempt at matching the imagined budget 1:1.
DIRECT_TOTAL_TIMESTEPS = 10_000


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the direct-RL PPO against real SUMO.")
    parser.add_argument("--seed", type=int, default=0, help="training seed (default: 0, the official run)")
    parser.add_argument(
        "--total-timesteps", type=int, default=DIRECT_TOTAL_TIMESTEPS,
        help=f"real SUMO training steps (default: {DIRECT_TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CONTROLLER_DIRECT_DIR,
        help="folder for checkpoints and evaluations (default: the official controller_direct/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing files already in --output-dir")
    parser.add_argument(
        "--overwrite-official", action="store_true",
        help="allow writing into controller_direct/ or controller_direct_30k/",
    )
    return parser.parse_args(argv)


def check_output_dir(output_dir: Path, overwrite: bool, overwrite_official: bool) -> Path:
    """Refuse to write anywhere a previous result could be lost, before any SUMO step runs."""
    output_dir = Path(output_dir).resolve()
    if output_dir in {path.resolve() for path in PROTECTED_DIRS} and not overwrite_official:
        raise SystemExit(
            f"{output_dir} holds official/archived checkpoints. Choose another --output-dir, "
            "or pass --overwrite-official if replacing them was explicitly approved."
        )
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"{output_dir} already contains {existing}. Choose an empty --output-dir, "
            "or pass --overwrite."
        )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = check_output_dir(args.output_dir, args.overwrite, args.overwrite_official)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run config, fixed up front because it now also builds the envs:
    # - total_timesteps: the direct budget actually used, not the Dream default.
    # - seed=0 by default: ControllerConfig's default moved to 1 for the Dream
    #   PPO; keeping 0 here means observation normalization is the only change
    #   from the previous direct-RL run.
    # - normalize_obs=True: unlike the Dream PPO (z, unit scale), this policy
    #   receives the RAW 26-dim state, whose per-dimension std spans 0.02 to 38
    #   (and far beyond the dataset range under bad early policies).
    # dream_max_steps is still written to the .json (it is a ControllerConfig
    # field) but has no effect here: this run never touches the Dream Environment.
    config = replace(
        ControllerConfig(), seed=args.seed, total_timesteps=args.total_timesteps, normalize_obs=True
    )
    _set_seeds(config.seed)

    train_env, eval_env = build_normalized_envs(
        lambda: ReseedingWrapper(TrafficEnvironment(), ReseedingWrapper.training_seeds()),
        lambda: ReseedingWrapper(TrafficEnvironment(), ReseedingWrapper.fixed_eval_seeds()),
        config,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        seed=config.seed,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=max(config.n_steps, 1000),
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeOnBest(output_dir / "best_model.zip"),
    )

    print(f"Training PPO DIRECTLY against real SUMO -- seed {config.seed}, {config.total_timesteps} real timesteps -> {output_dir}")
    print("This will be noticeably slower than Dream Environment training (real SUMO, not imagined).")
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)

    final_path = output_dir / "ppo_controller_direct_final.zip"
    model.save(final_path)
    save_hyperparameters(final_path, config)
    save_hyperparameters(output_dir / "best_model.zip", config)
    # best_model_vecnormalize.pkl was written by SaveVecNormalizeOnBest when the
    # best model was saved; these are the (different) end-of-training statistics
    # that belong to the final policy.
    train_env.save(str(vecnormalize_path(final_path)))
    print(f"Final policy saved to: {final_path}")
    print(f"Best policy saved to: {output_dir / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
