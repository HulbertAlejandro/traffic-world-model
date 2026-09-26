"""Final comparison on real SUMO, consolidating what the direct-RL block
investigated with one-off scripts (see PROJECT_STATUS.md, "Baseline de RL
directo y hallazgo final"):

1. Performance table, same protocol as every previous evaluation (15 episodes,
   seed_base=3000 and 5000): Dream-trained PPO, direct-RL PPO, fixed-time
   control, and the simple rule "always request the opposite phase"
   (action = 1 - green_phase), which both PPOs converged to under the
   symmetric-demand scenario.

2. Counterfactual action agreement on the SAME real states: the Dream-trained
   PPO drives SUMO on all 30 evaluation seeds; on every real state the script
   asks what the direct-RL PPO and the rule would have chosen, without
   executing them. A step counts as "blocked" when
   time_since_last_phase_change < yellow_time + min_green: there no action can
   change the traffic (only the 0.1 phase_change penalty), so agreement is
   reported separately on the non-blocked steps, the ones that matter.
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

PAIRS = [("sueno", "directo"), ("sueno", "regla"), ("directo", "regla")]


def _traffic_signal(env: TrafficEnvironment):
    sumo_env = env.env
    return sumo_env.traffic_signals[sumo_env.ts_ids[0]]


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
            total_throughput += info["arrivals_total"]  # vehicles that arrived; not the legacy info["throughput"]
            step += 1
        rewards.append(total_reward)
        waiting.append(total_waiting / max(step, 1))
        queue.append(total_queue / max(step, 1))
        throughput.append(total_throughput)
    return {name: np.array(values) for name, values in
            (("reward", rewards), ("waiting", waiting), ("queue", queue), ("throughput", throughput))}


def counterfactual_agreement(dream_env, dream_model, direct_model, normalize_direct, seeds):
    """Dream PPO drives SUMO; every real state is also shown to the direct PPO
    and the rule. Returns per-pair disagreement counts and the dream PPO's
    episode rewards (to check they reproduce the performance table)."""
    raw_env = dream_env._env
    counts = {"steps": 0, "blocked": 0, **{p: {"disagree": 0, "disagree_blocked": 0} for p in PAIRS}}
    dream_rewards = []
    for seed in seeds:
        raw_state, _ = raw_env.reset(seed=seed)
        ts = _traffic_signal(raw_env)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            actions = {
                "sueno": int(dream_model.predict(dream_env._encode(raw_state), deterministic=True)[0]),
                "directo": int(direct_model.predict(normalize_direct(raw_state), deterministic=True)[0]),
                "regla": 1 - ts.green_phase,
            }
            blocked = ts.time_since_last_phase_change < ts.yellow_time + ts.min_green
            counts["steps"] += 1
            counts["blocked"] += int(blocked)
            for a, b in PAIRS:
                if actions[a] != actions[b]:
                    counts[(a, b)]["disagree"] += 1
                    counts[(a, b)]["disagree_blocked"] += int(blocked)
            # Only the Dream-trained PPO's action is executed in SUMO.
            raw_state, reward, terminated, truncated, _ = raw_env.step(actions["sueno"])
            total_reward += reward
        dream_rewards.append(total_reward)
    return counts, np.array(dream_rewards)


def _fmt(values, width, decimals):
    return f"{values.mean():{width}.{decimals}f} +/- {values.std():{width - 2}.{decimals}f}"


def main() -> None:
    for path, hint in ((DREAM_PPO_PATH, "training/train_controller.py"),
                       (DIRECT_PPO_PATH, "training/train_controller_direct.py")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run {hint} first.")

    dream_env = EncodedTrafficEnvironment()
    direct_env = TrafficEnvironment()
    dream_model = PPO.load(DREAM_PPO_PATH)
    direct_model = PPO.load(DIRECT_PPO_PATH)
    # Identity for checkpoints trained without observation normalization;
    # otherwise the statistics saved together with this exact checkpoint.
    normalize_direct = load_obs_normalizer(DIRECT_PPO_PATH, direct_env)

    policies = [
        ("PPO (sueno)", dream_env,
         lambda obs, step: int(dream_model.predict(obs, deterministic=True)[0])),
        ("PPO (RL directo)", direct_env,
         lambda obs, step: int(direct_model.predict(normalize_direct(obs), deterministic=True)[0])),
        ("Tiempo fijo (ciclo=5)", direct_env,
         lambda obs, step: 1 if step % FIXED_TIME_SWITCH_EVERY == 0 else 0),
        ("Regla: pedir fase contraria", direct_env,
         lambda obs, step: 1 - _traffic_signal(direct_env).green_phase),
    ]

    print("=" * 118)
    print("1) DESEMPENO EN SUMO REAL (15 episodios por semilla base)")
    print("=" * 118)
    dream_table_rewards = {}
    for seed_base in SEED_BASES:
        print(f"\n--- seed_base={seed_base} ---")
        print(f"{'Politica':30s} | {'reward':>20s} | {'espera_prom':>16s} | {'cola_prom':>14s} | {'llegadas':>16s}")
        print("-" * 110)
        for name, env, policy_fn in policies:
            result = run_policy(env, policy_fn, NUM_EPISODES, seed_base)
            if name == "PPO (sueno)":
                dream_table_rewards[seed_base] = result["reward"]
            print(f"{name:30s} | {_fmt(result['reward'], 9, 2):>20s} | {_fmt(result['waiting'], 7, 2):>16s} | "
                  f"{_fmt(result['queue'], 6, 2):>14s} | {_fmt(result['throughput'], 7, 2):>16s}")
            print(f"    valores individuales: {[round(float(r), 2) for r in result['reward']]}")

    print()
    print("=" * 118)
    print("2) ACUERDO CONTRAFACTUAL SOBRE LOS MISMOS ESTADOS REALES (el PPO del sueno conduce SUMO)")
    print("=" * 118)
    seeds = [base + i for base in SEED_BASES for i in range(NUM_EPISODES)]
    counts, dream_rewards = counterfactual_agreement(dream_env, dream_model, direct_model, normalize_direct, seeds)
    expected = np.concatenate([dream_table_rewards[base] for base in SEED_BASES])
    reproduced = bool(np.allclose(dream_rewards, expected, atol=0.05))
    print(f"Trayectorias del PPO del sueno reproducen la tabla 1: {'si' if reproduced else 'NO'}")
    n, n_blocked = counts["steps"], counts["blocked"]
    n_free = n - n_blocked
    print(f"{n} pasos, {n_blocked} bloqueados por min_green ({100 * n_blocked / n:.1f}%), {n_free} no bloqueados")
    print(f"{'Par':20s} | {'acuerdo total':>13s} | {'desacuerdos':>11s} | {'en pasos bloqueados':>22s} | "
          f"{'acuerdo en pasos NO bloqueados':>32s}")
    print("-" * 110)
    for a, b in PAIRS:
        d = counts[(a, b)]
        free_disagree = d["disagree"] - d["disagree_blocked"]
        pct_blocked = 100 * d["disagree_blocked"] / d["disagree"] if d["disagree"] else float("nan")
        free_agree = 100 * (n_free - free_disagree) / n_free if n_free else float("nan")
        blocked_str = f"{d['disagree_blocked']} ({pct_blocked:.1f}%)"
        free_str = f"{free_agree:.1f}% ({free_disagree} de {n_free})"
        print(f"{a + ' / ' + b:20s} | {100 * (n - d['disagree']) / n:12.1f}% | {d['disagree']:11d} | "
              f"{blocked_str:>22s} | {free_str:>32s}")

    dream_env.close()
    direct_env.close()


if __name__ == "__main__":
    main()
