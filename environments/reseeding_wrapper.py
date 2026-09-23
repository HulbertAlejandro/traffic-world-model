"""Gymnasium wrapper that assigns a new SUMO seed on every reset() that doesn't
already specify one.

sumo_rl only changes its traffic seed when reset() explicitly receives one --
otherwise it reuses whatever seed was last used (see
sumo_rl.environment.env.SumoEnvironment.reset). Stable-Baselines3 calls
env.reset() without a seed on every episode boundary during training and
during EvalCallback's periodic evaluations, so without this wrapper:
- Training would run ~167 episodes (10_000 timesteps / 60 steps) against the
  SAME traffic realization, learning to memorize one scenario instead of
  generalizing.
- EvalCallback would evaluate every checkpoint against a single fixed seed,
  making "best checkpoint" a comparison against one scenario, not a
  representative sample.

Two seeding modes are supported:
- An infinite, ever-increasing sequence (for training): every episode gets a
  genuinely new seed, for maximum traffic variety during learning.
- A fixed, cyclically-repeated list (for evaluation): the same N seeds repeat
  across every periodic evaluation, so successive checkpoints are compared on
  identical traffic scenarios.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import gymnasium as gym


class ReseedingWrapper(gym.Wrapper):
    """Assigns env.reset(seed=next(seed_iterator)) whenever reset() is called
    without an explicit seed. Passing an explicit seed to reset() always wins
    -- this wrapper only fills in the gap SB3 leaves during normal training
    and evaluation loops."""

    def __init__(self, env: gym.Env, seed_iterator: Iterator[int]) -> None:
        super().__init__(env)
        self._seed_iterator = seed_iterator

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = next(self._seed_iterator)
        return self.env.reset(seed=seed, options=options)

    @staticmethod
    def training_seeds(start: int = 10_000) -> Iterator[int]:
        """Ever-increasing sequence: 10000, 10001, 10002, ... -- a fresh
        traffic realization every episode during training."""
        return itertools.count(start)

    @staticmethod
    def fixed_eval_seeds(seeds: list[int] | None = None) -> Iterator[int]:
        """Cyclically repeats a fixed, small set of seeds -- so every
        periodic evaluation during training compares checkpoints on the
        exact same traffic scenarios, not a new random one each time."""
        seeds = seeds or [20_000, 20_001, 20_002, 20_003, 20_004]
        return itertools.cycle(seeds)
