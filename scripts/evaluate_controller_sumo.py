"""Evaluate the Dream-trained PPO controller against SUMO real, comparing
against a fixed-time baseline and a random baseline (Sección 18 of the
proposal). Uses REAL traffic metrics from TrafficEnvironment (waiting time,
queue length, arrivals), not just abstract reward -- this is the
evaluation that actually validates (or invalidates) everything built so far.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.encoded_traffic_environment import EncodedTrafficEnvironment

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
PPO_MODEL_PATH = CHECKPOINT_DIR / "controller" / "best_model.zip"

NUM_EPISODES = 15  # SUMO real is slow; keep this modest
FIXED_TIME_SWITCH_EVERY = 5  # matches min_green, a reasonable fixed-time cycle


def run_policy(env, policy_fn, num_episodes, seed_base=3000):
    rewards, waiting, queue, throughput = [], [], [], []
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed_base + i)
        total_reward = total_waiting = total_queue = total_throughput = 0.0
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            action = policy_fn(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_waiting += info.get("waiting_total", 0.0)
            total_queue += info.get("queue_total", 0.0)
            total_throughput += info["arrivals_total"]  # vehicles that arrived; not the legacy info["throughput"]
            step += 1
        rewards.append(total_reward)
        waiting.append(total_waiting / max(step, 1))
        queue.append(total_queue / max(step, 1))
        throughput.append(total_throughput)
    rewards, waiting, queue, throughput = map(np.array, (rewards, waiting, queue, throughput))
    return (
        rewards.mean(), rewards.std(), waiting.mean(), waiting.std(),
        queue.mean(), queue.std(), throughput.mean(), throughput.std(),
        rewards,
    )


def main() -> None:
    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PPO_MODEL_PATH}. Run training/train_controller.py first.")

    env = EncodedTrafficEnvironment()
    ppo_model = PPO.load(PPO_MODEL_PATH)

    def ppo_policy(obs, step):
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)

    def fixed_time_policy(obs, step):
        return 1 if step % FIXED_TIME_SWITCH_EVERY == 0 else 0

    def random_policy(obs, step):
        return int(np.random.randint(0, 2))

    policies = {
        "PPO (entrenado en el sueno)": ppo_policy,
        "Tiempo fijo (ciclo=5)": fixed_time_policy,
        "Accion aleatoria": random_policy,
    }

    print("Evaluando contra SUMO REAL (no Dream Environment) -- esto es lo que realmente importa.")
    print(f"{'Politica':30s} | {'reward':>16s} | {'espera_prom':>16s} | {'cola_prom':>14s} | {'llegadas':>14s}")
    print("-" * 105)

    per_policy_rewards = {}
    for name, policy_fn in policies.items():
        r_mean, r_std, w_mean, w_std, q_mean, q_std, t_mean, t_std, rewards = run_policy(
            env, policy_fn, NUM_EPISODES
        )
        per_policy_rewards[name] = rewards
        reward_str = f"{r_mean:7.2f} +/- {r_std:5.2f}"
        waiting_str = f"{w_mean:6.2f} +/- {w_std:5.2f}"
        queue_str = f"{q_mean:5.2f} +/- {q_std:4.2f}"
        throughput_str = f"{t_mean:5.2f} +/- {t_std:4.2f}"
        print(
            f"{name:30s} | {reward_str:>16s} | {waiting_str:>16s} | "
            f"{queue_str:>14s} | {throughput_str:>14s}"
        )

    print()
    print("Reward por episodio (uno por uno):")
    for name, rewards in per_policy_rewards.items():
        print(f"{name}:")
        for episode_index, reward in enumerate(rewards):
            print(f"  episodio {episode_index:2d} | reward={reward:10.2f}")

    env.close()


if __name__ == "__main__":
    main()
