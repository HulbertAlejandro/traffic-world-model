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
"""

from __future__ import annotations

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
from training.train_controller import build_normalized_envs, save_hyperparameters

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIRECT_DIR = CHECKPOINT_DIR / "controller_direct"
CONTROLLER_DIRECT_DIR.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    config = ControllerConfig()
    _set_seeds(config.seed)

    # Reward normalization only (see build_normalized_envs). NOTE: unlike the
    # Dream PPO, this policy receives the RAW, unnormalized 26-dim state;
    # observation normalization is deliberately left out of this change so only
    # one variable (reward scale) differs from earlier runs.
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
        best_model_save_path=str(CONTROLLER_DIRECT_DIR),
        log_path=str(CONTROLLER_DIRECT_DIR),
        eval_freq=max(config.n_steps, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training PPO DIRECTLY against real SUMO -- {DIRECT_TOTAL_TIMESTEPS} real timesteps.")
    print("This will be noticeably slower than Dream Environment training (real SUMO, not imagined).")
    model.learn(total_timesteps=DIRECT_TOTAL_TIMESTEPS, callback=eval_callback)

    # Record the budget actually used, not ControllerConfig's Dream default.
    # dream_max_steps is still written (it is a ControllerConfig field) but has
    # no effect here: this run never touches the Dream Environment.
    run_config = replace(config, total_timesteps=DIRECT_TOTAL_TIMESTEPS)
    final_path = CONTROLLER_DIRECT_DIR / "ppo_controller_direct_final.zip"
    model.save(final_path)
    save_hyperparameters(final_path, run_config)
    save_hyperparameters(CONTROLLER_DIRECT_DIR / "best_model.zip", run_config)
    train_env.save(str(CONTROLLER_DIRECT_DIR / "vec_normalize.pkl"))
    print(f"Final policy saved to: {final_path}")
    print(f"Best policy saved to: {CONTROLLER_DIRECT_DIR / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
