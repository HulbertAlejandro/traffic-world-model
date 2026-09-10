"""Smoke test del proyecto."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import TrafficEnvironment


def main():
    env = TrafficEnvironment()

    obs, info = env.reset()

    print("=" * 60)
    print("Observación inicial")
    print(obs)
    print("=" * 60)

    total_reward = 0.0

    for step in range(60):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step:03d} | Action={action} | Reward={reward:.3f}")

        if terminated or truncated:
            break

    print()
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()