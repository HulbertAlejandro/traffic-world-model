"""Diagnose whether the PPO controller is exploiting reward clipping.

Compares, for PPO and each baseline policy, not just HOW OFTEN reward_clipped
fires (already measured in evaluate_controller.py) but the MAGNITUDE of the
clip when it happens: |raw_predicted_reward - clipped_bound|. A policy that
is merely landing slightly outside the empirical range looks very different
from one that is deep in hallucinated territory being masked by the clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.dream_environment import DreamEnvironment, REWARD_CLIP_MAX, REWARD_CLIP_MIN

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIR = CHECKPOINT_DIR / "controller"

TEST_LATENT_PATH = PROCESSED_DIR / "test_latent.npz"
WORLD_MODEL_CHECKPOINT = CHECKPOINT_DIR / "world_model_best.pt"
PPO_MODEL_PATH = CONTROLLER_DIR / "best_model.zip"

NUM_EPISODES = 30


def run_policy(env, policy_fn, num_episodes, seed_base=2000):
    clip_magnitudes = []
    actions_taken = []
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed_base + i)
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            action = policy_fn(obs, step)
            actions_taken.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if info["reward_clipped"]:
                bound = REWARD_CLIP_MIN if info["raw_predicted_reward"] < REWARD_CLIP_MIN else REWARD_CLIP_MAX
                clip_magnitudes.append(abs(info["raw_predicted_reward"] - bound))
            step += 1
    return np.array(clip_magnitudes), np.array(actions_taken)


def main() -> None:
    env = DreamEnvironment(checkpoint_path=WORLD_MODEL_CHECKPOINT, latent_episodes_path=TEST_LATENT_PATH)
    ppo_model = PPO.load(PPO_MODEL_PATH)

    def ppo_policy(obs, step):
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)

    policies = {
        "PPO (entrenado)": ppo_policy,
        "Alternando cada paso": lambda obs, step: step % 2,
        "Siempre mantener (0)": lambda obs, step: 0,
        "Siempre cambiar (1)": lambda obs, step: 1,
    }

    print(f"Rango de recorte: [{REWARD_CLIP_MIN}, {REWARD_CLIP_MAX}]")
    print()
    print(f"{'Politica':25s} | {'n_clips':>8s} | {'mag_media':>10s} | {'mag_max':>10s} | {'accion=0':>9s} | {'accion=1':>9s}")
    print("-" * 85)

    for name, policy_fn in policies.items():
        magnitudes, actions = run_policy(env, policy_fn, NUM_EPISODES)
        n_zeros = int(np.sum(actions == 0))
        n_ones = int(np.sum(actions == 1))
        mag_mean = magnitudes.mean() if len(magnitudes) else 0.0
        mag_max = magnitudes.max() if len(magnitudes) else 0.0
        print(
            f"{name:25s} | {len(magnitudes):8d} | {mag_mean:10.2f} | {mag_max:10.2f} | "
            f"{n_zeros:9d} | {n_ones:9d}"
        )


if __name__ == "__main__":
    main()
