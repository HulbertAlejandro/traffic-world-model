"""Train a PPO controller inside the Dream Environment (no SUMO interaction).

This is the core experiment enabled by the World Model: the controller learns
entirely from imagined rollouts produced by the trained LatentDynamicsLSTM,
never touching the real simulator during training.
"""

from __future__ import annotations

import random
import sys
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
from environments.dream_environment import DreamEnvironment

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIR = CHECKPOINT_DIR / "controller"
CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LATENT_PATH = PROCESSED_DIR / "train_latent.npz"
VALIDATION_LATENT_PATH = PROCESSED_DIR / "validation_latent.npz"
WORLD_MODEL_CHECKPOINT = CHECKPOINT_DIR / "world_model_best.pt"


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

    for required in (TRAIN_LATENT_PATH, VALIDATION_LATENT_PATH, WORLD_MODEL_CHECKPOINT):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}. Run the earlier pipeline stages first.")

    train_env = DreamEnvironment(
        checkpoint_path=WORLD_MODEL_CHECKPOINT,
        latent_episodes_path=TRAIN_LATENT_PATH,
        max_dream_steps=config.dream_max_steps,
    )
    eval_env = DreamEnvironment(
        checkpoint_path=WORLD_MODEL_CHECKPOINT,
        latent_episodes_path=VALIDATION_LATENT_PATH,
        max_dream_steps=config.dream_max_steps,
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
        best_model_save_path=str(CONTROLLER_DIR),
        log_path=str(CONTROLLER_DIR),
        eval_freq=max(config.n_steps, 1000),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)

    final_path = CONTROLLER_DIR / "ppo_controller_final.zip"
    model.save(final_path)
    print(f"Final policy saved to: {final_path}")
    print(f"Best policy (by validation-seeded dream reward) saved to: {CONTROLLER_DIR / 'best_model.zip'}")


if __name__ == "__main__":
    main()
