"""Train a PPO controller inside the Dream Environment.

This is the core experiment enabled by the World Model: the controller learns
entirely from imagined rollouts produced by the trained LatentDynamicsLSTM,
never touching the real simulator during training. Checkpoint SELECTION is the
one exception: see the EvalCallback comment in main().
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ControllerConfig
from environments.dream_environment import DreamEnvironment
from environments.encoded_traffic_environment import EncodedTrafficEnvironment
from environments.reseeding_wrapper import ReseedingWrapper

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIR = CHECKPOINT_DIR / "controller"
CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LATENT_PATH = PROCESSED_DIR / "train_latent.npz"
WORLD_MODEL_CHECKPOINT = CHECKPOINT_DIR / "world_model_best.pt"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def save_hyperparameters(checkpoint_path: Path, config: ControllerConfig) -> None:
    """Write the ControllerConfig used in this run next to a PPO checkpoint,
    same sibling-.json pattern as autoencoder_best.json / world_model_best.json."""
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )


def build_normalized_envs(make_train_env, make_eval_env, config: ControllerConfig):
    """Wrap a training and an evaluation env for PPO with reward normalization.

    Reward (not observation) normalization: the value function never learned
    (explained_variance ~0 in every run) and value_loss matched the variance of
    unnormalized returns (std in the hundreds to thousands) -- see
    PROJECT_STATUS.md. Observations are left untouched, so a trained policy is
    used at inference exactly as before and no normalization statistics are
    needed to evaluate it.

    Monitor sits BELOW VecNormalize so episode logs keep the real reward. The
    eval env must be wrapped the same way: EvalCallback copies the training
    statistics into it before every evaluation (sync_envs_normalization) and
    fails otherwise. training=False freezes those statistics; norm_reward=False
    keeps the reported evaluation reward real, comparable with earlier tables.
    """
    train_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(make_train_env())]),
        norm_obs=False,
        norm_reward=config.normalize_reward,
        clip_reward=config.reward_clip,
        gamma=config.gamma,
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(make_eval_env())]),
        training=False,
        norm_obs=False,
        norm_reward=False,
    )
    return train_env, eval_env


def main() -> None:
    config = ControllerConfig()
    _set_seeds(config.seed)

    for required in (TRAIN_LATENT_PATH, WORLD_MODEL_CHECKPOINT):
        if not required.exists():
            raise FileNotFoundError(f"Missing {required}. Run the earlier pipeline stages first.")

    # Checkpoint selection runs on REAL SUMO, not on imagined rollouts: an
    # investigation found imagined reward does not predict real reward (Pearson
    # ~0.08 over 25 checkpoints of the same run), so picking the best
    # checkpoint by imagined reward was, in practice, close to random -- see
    # PROJECT_STATUS.md. Training itself still uses zero real SUMO steps, but
    # this selection costs real interactions: 10 evaluations x 5 episodes x 60
    # steps = 3,000 real SUMO steps per run. ReseedingWrapper gives every
    # evaluation the same 5 fixed traffic seeds (20000-20004); without it,
    # sumo_rl would reuse one seed for all episodes.
    train_env, eval_env = build_normalized_envs(
        lambda: DreamEnvironment(
            checkpoint_path=WORLD_MODEL_CHECKPOINT,
            latent_episodes_path=TRAIN_LATENT_PATH,
            max_dream_steps=config.dream_max_steps,
        ),
        lambda: ReseedingWrapper(EncodedTrafficEnvironment(), ReseedingWrapper.fixed_eval_seeds()),
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
        best_model_save_path=str(CONTROLLER_DIR),
        log_path=str(CONTROLLER_DIR),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)

    final_path = CONTROLLER_DIR / "ppo_controller_final.zip"
    model.save(final_path)
    save_hyperparameters(final_path, config)
    save_hyperparameters(CONTROLLER_DIR / "best_model.zip", config)
    # Only needed to resume training (or if norm_obs is ever enabled);
    # evaluating the policy does not require it.
    train_env.save(str(CONTROLLER_DIR / "vec_normalize.pkl"))
    print(f"Final policy saved to: {final_path}")
    print(f"Best policy (by real-SUMO evaluation reward) saved to: {CONTROLLER_DIR / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
