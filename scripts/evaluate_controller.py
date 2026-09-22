"""Evaluate the trained PPO controller against simple baseline policies,
inside the Dream Environment seeded from the held-out test split.

This mirrors the manual sanity check already used to validate
DreamEnvironment itself (see PROJECT_STATUS.md), now applied to check whether
PPO learned something better than trivial policies -- not a claim about
real-world SUMO performance, which requires a separate evaluation stage
(Sección 20 of the proposal).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.dream_environment import DreamEnvironment

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROLLER_DIR = CHECKPOINT_DIR / "controller"

TEST_LATENT_PATH = PROCESSED_DIR / "test_latent.npz"
WORLD_MODEL_CHECKPOINT = CHECKPOINT_DIR / "world_model_best.pt"
PPO_MODEL_PATH = CONTROLLER_DIR / "best_model.zip"

NUM_EPISODES = 30


def run_policy(env, policy_fn, num_episodes, seed_base=1000):
    totals, streak_counts, clip_counts, total_steps = [], 0, 0, 0
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed_base + i)
        total_reward = 0.0
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            action = policy_fn(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if info.get("consecutive_action_streak", 0) >= 5:
                streak_counts += 1
            if info.get("reward_clipped", False):
                clip_counts += 1
            step += 1
        totals.append(total_reward)
    totals = np.array(totals)
    return totals, streak_counts, clip_counts, total_steps


def main() -> None:
    if not TEST_LATENT_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_LATENT_PATH}.")
    if not WORLD_MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing {WORLD_MODEL_CHECKPOINT}.")
    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PPO_MODEL_PATH}. Run training/train_controller.py first.")

    env = DreamEnvironment(checkpoint_path=WORLD_MODEL_CHECKPOINT, latent_episodes_path=TEST_LATENT_PATH)
    ppo_model = PPO.load(PPO_MODEL_PATH)

    def ppo_policy(obs, step):
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)

    policies = {
        "PPO (entrenado)": ppo_policy,
        "Accion aleatoria": lambda obs, step: int(np.random.randint(0, 2)),
        "Siempre mantener (0)": lambda obs, step: 0,
        "Siempre cambiar (1)": lambda obs, step: 1,
        "Alternando cada paso": lambda obs, step: step % 2,
    }

    print(f"Evaluando en Dream Environment sembrado con {TEST_LATENT_PATH.name} (split de prueba, nunca visto)")
    print(f"{'Politica':25s} | {'mean':>10s} | {'std':>8s} | {'min':>10s} | {'max':>10s} | {'streak>=5':>14s} | {'clipped':>14s}")
    print("-" * 105)

    for name, policy_fn in policies.items():
        totals, streak_counts, clip_counts, total_steps = run_policy(env, policy_fn, NUM_EPISODES)
        streak_pct = 100 * streak_counts / total_steps if total_steps else 0.0
        clip_pct = 100 * clip_counts / total_steps if total_steps else 0.0
        print(
            f"{name:25s} | {totals.mean():10.2f} | {totals.std():8.2f} | "
            f"{totals.min():10.2f} | {totals.max():10.2f} | "
            f"{streak_counts:4d}/{total_steps:<4d} ({streak_pct:4.1f}%) | "
            f"{clip_counts:4d}/{total_steps:<4d} ({clip_pct:4.1f}%)"
        )


if __name__ == "__main__":
    main()
