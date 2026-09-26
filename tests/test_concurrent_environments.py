"""Two SUMO environments open in the same process must not read each other.

train_controller_direct.py runs a training TrafficEnvironment and an
EvalCallback TrafficEnvironment in one process. CustomStateBuilder used to read
lanes through the global ``traci`` module, which points to the LAST simulation
started, so after every periodic evaluation the training environment observed
(and was rewarded with) the evaluation simulation's lanes until its next reset.
"""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import TrafficEnvironment

SEED_A, SEED_B = 1, 2
ACTIONS_A = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
ACTIONS_B = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]


def _solo_run(seed, actions):
    env = TrafficEnvironment()
    try:
        states, rewards = [env.reset(seed=seed)[0]], []
        for action in actions:
            state, reward, *_ = env.step(action)
            states.append(state)
            rewards.append(reward)
    finally:
        env.close()
    return np.array(states), np.array(rewards)


def test_two_concurrent_environments_each_read_their_own_simulation():
    solo_states_a, solo_rewards_a = _solo_run(SEED_A, ACTIONS_A)
    solo_states_b, solo_rewards_b = _solo_run(SEED_B, ACTIONS_B)

    env_a, env_b = TrafficEnvironment(), TrafficEnvironment()
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

    # Sanity check: the two scenarios really differ, so a cross-read would show.
    assert not np.array_equal(solo_states_a, solo_states_b)
    np.testing.assert_array_equal(np.array(states_a), solo_states_a)
    np.testing.assert_array_equal(np.array(rewards_a), solo_rewards_a)
    np.testing.assert_array_equal(np.array(states_b), solo_states_b)
    np.testing.assert_array_equal(np.array(rewards_b), solo_rewards_b)
