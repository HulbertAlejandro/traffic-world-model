"""Final comparison: PPO trained directly against SUMO vs. the official
Dream-trained PPO (v1, dream_max_steps=7) vs. fixed-time control -- all
evaluated on real SUMO. This answers the project's core research question
(Sección 18): does the World Model approach reduce the real SUMO interactions
needed to reach a given level of traffic control quality?

Uses the SAME evaluation seeds already used and documented throughout
PROJECT_STATUS.md (3000, 5000) -- distinct from the training/EvalCallback
seeds used inside train_controller_direct.py (10000+, 20000-20004).
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
from environments.traffic_environment import TrafficEnvironment
from training.train_controller import load_obs_normalizer

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
DREAM_PPO_PATH = CHECKPOINT_DIR / "controller" / "best_model.zip"
DIRECT_PPO_PATH = CHECKPOINT_DIR / "controller_direct" / "best_model.zip"

NUM_EPISODES = 15
SEED_BASES = [3000, 5000]
FIXED_TIME_SWITCH_EVERY = 5


def run_policy(env, policy_fn, num_episodes, seed_base):
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
            total_throughput += info.get("throughput", 0.0)
            step += 1
        rewards.append(total_reward)
        waiting.append(total_waiting / max(step, 1))
        queue.append(total_queue / max(step, 1))
        throughput.append(total_throughput)
    rewards, waiting, queue, throughput = map(np.array, (rewards, waiting, queue, throughput))
    return (
        rewards.mean(), rewards.std(), waiting.mean(), waiting.std(),
        queue.mean(), queue.std(), throughput.mean(), throughput.std(), rewards,
    )


def main() -> None:
    if not DREAM_PPO_PATH.exists():
        raise FileNotFoundError(f"Missing {DREAM_PPO_PATH}.")
    if not DIRECT_PPO_PATH.exists():
        raise FileNotFoundError(f"Missing {DIRECT_PPO_PATH}. Run training/train_controller_direct.py first.")

    dream_env = EncodedTrafficEnvironment()
    direct_env = TrafficEnvironment()

    dream_model = PPO.load(DREAM_PPO_PATH)
    direct_model = PPO.load(DIRECT_PPO_PATH)
    # Identity for checkpoints trained without observation normalization;
    # otherwise the statistics saved together with this exact checkpoint.
    normalize_direct = load_obs_normalizer(DIRECT_PPO_PATH, direct_env)

    def dream_policy(obs, step):
        action, _ = dream_model.predict(obs, deterministic=True)
        return int(action)

    def direct_policy(obs, step):
        action, _ = direct_model.predict(normalize_direct(obs), deterministic=True)
        return int(action)

    def fixed_time_policy(obs, step):
        return 1 if step % FIXED_TIME_SWITCH_EVERY == 0 else 0

    for seed_base in SEED_BASES:
        print(f"\n=== Evaluacion con seed_base={seed_base} ===")
        print(f"{'Politica':35s} | {'reward':>18s} | {'espera_prom':>16s} | {'cola_prom':>14s} | {'throughput':>14s}")
        print("-" * 110)

        for name, env, policy_fn in [
            ("PPO (sueno, v1 oficial)", dream_env, dream_policy),
            ("PPO (RL directo, real SUMO)", direct_env, direct_policy),
            ("Tiempo fijo (ciclo=5)", direct_env, fixed_time_policy),
        ]:
            r_mean, r_std, w_mean, w_std, q_mean, q_std, t_mean, t_std, rewards = run_policy(
                env, policy_fn, NUM_EPISODES, seed_base
            )
            reward_str = f"{r_mean:8.2f} +/- {r_std:6.2f}"
            waiting_str = f"{w_mean:6.2f} +/- {w_std:5.2f}"
            queue_str = f"{q_mean:5.2f} +/- {q_std:4.2f}"
            throughput_str = f"{t_mean:5.2f} +/- {t_std:4.2f}"
            print(
                f"{name:35s} | {reward_str:>18s} | {waiting_str:>16s} | "
                f"{queue_str:>14s} | {throughput_str:>14s}"
            )
            print(f"    valores individuales: {[round(r, 2) for r in rewards]}")

    dream_env.close()
    direct_env.close()


if __name__ == "__main__":
    main()
