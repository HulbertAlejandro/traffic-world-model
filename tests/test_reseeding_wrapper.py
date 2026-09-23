"""Tests ReseedingWrapper with a minimal fake env -- no SUMO required, this
is pure Python logic."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from environments.reseeding_wrapper import ReseedingWrapper


class _FakeEnv(gym.Env):
    """Records every seed it was actually reset with."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.seeds_seen: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        self.seeds_seen.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_training_seeds_increase_every_reset_without_explicit_seed():
    env = ReseedingWrapper(_FakeEnv(), ReseedingWrapper.training_seeds(start=10_000))
    for _ in range(5):
        env.reset()
    assert env.unwrapped.seeds_seen == [10_000, 10_001, 10_002, 10_003, 10_004]


def test_fixed_eval_seeds_cycle_through_a_short_list():
    env = ReseedingWrapper(_FakeEnv(), ReseedingWrapper.fixed_eval_seeds([20_000, 20_001]))
    for _ in range(5):
        env.reset()
    assert env.unwrapped.seeds_seen == [20_000, 20_001, 20_000, 20_001, 20_000]


def test_explicit_seed_always_overrides_the_wrapper():
    env = ReseedingWrapper(_FakeEnv(), ReseedingWrapper.training_seeds(start=10_000))
    env.reset(seed=999)
    env.reset()
    env.reset(seed=888)
    assert env.unwrapped.seeds_seen == [999, 10_000, 888]
