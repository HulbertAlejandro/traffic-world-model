"""Two v2 corridor environments open in the same process must not read each other.

Same check as tests/test_concurrent_environments.py (v1 bug C1: the state
builder read lanes through the global ``traci`` module, which points to the
LAST simulation started), for the four-intersection environment. Any v2
training with periodic evaluation runs two corridor simulations at once.
Here the two instances also differ in pulse offset, so each must use its own
route file as well as its own TraCI connection.
"""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_environment import CorridorTrafficEnvironment

SEED_A, SEED_B = 19900, 19901
_rng = np.random.default_rng(0)
ACTIONS_A = _rng.integers(0, 2, size=(14, 4))
ACTIONS_B = _rng.integers(0, 2, size=(14, 4))


def _solo_run(seed, actions):
    env = CorridorTrafficEnvironment()
    try:
        state, info = env.reset(seed=seed)
        states, rewards, offset = [state], [], info["pulse_offset"]
        for action in actions:
            state, reward, *_ = env.step(action)
            states.append(state)
            rewards.append(reward)
    finally:
        env.close()
    return np.array(states), np.array(rewards), offset


def test_two_concurrent_corridor_environments_each_read_their_own_simulation():
    solo_states_a, solo_rewards_a, offset_a = _solo_run(SEED_A, ACTIONS_A)
    solo_states_b, solo_rewards_b, offset_b = _solo_run(SEED_B, ACTIONS_B)

    env_a, env_b = CorridorTrafficEnvironment(), CorridorTrafficEnvironment()
    try:
        # B starts last, so the global traci module points to B from here on.
        states_a, states_b = [env_a.reset(seed=SEED_A)[0]], [env_b.reset(seed=SEED_B)[0]]
        rewards_a, rewards_b = [], []
        for action_a, action_b in zip(ACTIONS_A, ACTIONS_B):
            state, reward, *_ = env_a.step(action_a)
            states_a.append(state)
            rewards_a.append(reward)
            state, reward, *_ = env_b.step(action_b)
            states_b.append(state)
            rewards_b.append(reward)
    finally:
        env_a.close()
        env_b.close()

    # Sanity checks: the two scenarios really differ, so a cross-read would show.
    assert offset_a != offset_b
    assert not np.array_equal(solo_states_a, solo_states_b)
    assert not np.array_equal(solo_rewards_a, solo_rewards_b)
    np.testing.assert_array_equal(np.array(states_a), solo_states_a)
    np.testing.assert_array_equal(np.array(rewards_a), solo_rewards_a)
    np.testing.assert_array_equal(np.array(states_b), solo_states_b)
    np.testing.assert_array_equal(np.array(rewards_b), solo_rewards_b)
