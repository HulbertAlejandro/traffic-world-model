"""v2 environment with state, joint action and reward (SUMO in the loop)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_environment import (
    KEEP,
    SWITCH,
    CorridorActionSpace,
    CorridorStateBuilder,
    CorridorTrafficEnvironment,
)
from environments.four_intersections import DELTA_TIME, SIMULATION_SECONDS, TRAFFIC_SIGNAL_IDS
from scripts.v2.corridor_policies import make_collection_policy

PHASE0_RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "demand_calibration" / "it5_300s.json"
STATE_SIZE = 26 * 4


def _run(env, policy, seed):
    state, info = env.reset(seed=seed)
    states, rewards, truncs = [state], [], []
    step, done = 0, False
    while not done:
        state, reward, terminated, truncated, info = env.step(policy(env, step))
        states.append(state)
        rewards.append(reward)
        truncs.append(truncated)
        done = terminated or truncated
        step += 1
    return np.array(states), np.array(rewards), truncs


def test_full_episode_state_size_and_reward_sign():
    env = CorridorTrafficEnvironment()
    try:
        assert env.observation_space.shape == (STATE_SIZE,)
        states, rewards, truncs = _run(env, make_collection_policy("aleatoria", 7), seed=7)
        assert env.state_builder.state_size() == STATE_SIZE
    finally:
        env.close()
    assert states.shape == (SIMULATION_SECONDS // DELTA_TIME + 1, STATE_SIZE)
    assert np.isfinite(states).all() and np.isfinite(rewards).all()
    assert truncs[-1] and not any(truncs[:-1])  # runs the whole 300 s, never cut early
    assert (rewards <= 0).all() and rewards.sum() < 0


@pytest.mark.parametrize("policy", ["fijo_2_3", "cola_mas_larga"])
def test_reward_reproduces_phase0_measurement(policy):
    """With the static it5 route file (offset 0), the episode return equals the
    reward_v1_style the Phase 0 validator measured for the same policy and seed:
    same reward definition, same sign and scale, summed over the four signals."""
    seed = 11000
    phase0 = json.loads(PHASE0_RESULTS.read_text(encoding="utf-8"))
    expected = next(e["reward_v1_style"] for e in phase0["episodes"]
                    if e["policy"] == policy and e["seed"] == seed)
    env = CorridorTrafficEnvironment(random_pulse_offset=False)
    try:
        _, rewards, _ = _run(env, make_collection_policy(policy, seed), seed=seed)
    finally:
        env.close()
    assert rewards.sum() == pytest.approx(expected, rel=1e-5)


def test_keep_and_switch_actions():
    env = CorridorTrafficEnvironment(seed=3)
    try:
        env.reset(seed=3)
        assert env.green_phases() == dict.fromkeys(TRAFFIC_SIGNAL_IDS, 0)
        # A switch needs yellow_time + min_green = 7 s since the last change: blocked at t=0 and t=5.
        for _ in range(2):
            _, _, _, _, info = env.step([SWITCH] * 4)
            assert info["phase_switched"].tolist() == [0.0] * 4
        _, _, _, _, info = env.step([SWITCH, KEEP, SWITCH, KEEP])  # t=10
        assert info["phase_switched"].tolist() == [1.0, 0.0, 1.0, 0.0]
        assert env.green_phases() == {"A0": 1, "B0": 0, "C0": 1, "D0": 0}
        _, _, _, _, info = env.step([SWITCH] * 4)  # t=15: A0 and C0 switched 5 s ago, blocked
        assert info["phase_switched"].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert env.green_phases() == {"A0": 1, "B0": 1, "C0": 1, "D0": 1}
        _, _, _, _, info = env.step([SWITCH, SWITCH, KEEP, KEEP])  # t=20
        # B0 switched at t=15, only 5 s ago: blocked even though it asks to switch.
        assert info["phase_switched"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert env.green_phases() == {"A0": 0, "B0": 1, "C0": 1, "D0": 1}
        with pytest.raises(ValueError):
            env.step([1, 0, 2, 0])
    finally:
        env.close()


def test_action_encoding_round_trip():
    space = CorridorActionSpace()
    for current in ([0, 0, 0, 0], [1, 0, 1, 1]):
        cur = dict(zip(TRAFFIC_SIGNAL_IDS, current))
        for bits in range(16):
            action = np.array([(bits >> i) & 1 for i in range(4)])
            targets = space.to_target_phases(action, cur)
            back = space.from_target_phases([targets[ts] for ts in TRAFFIC_SIGNAL_IDS], current)
            np.testing.assert_array_equal(back, action)


def test_state_columns_follow_v1_layout_per_signal():
    builder = CorridorStateBuilder()
    assert [builder.signal_slice(ts) for ts in TRAFFIC_SIGNAL_IDS] == [
        slice(0, 26), slice(26, 52), slice(52, 78), slice(78, 104)]


def test_pulse_offset_comes_from_the_episode_seed():
    env = CorridorTrafficEnvironment()
    try:
        first, info_a = env.reset(seed=20001)
        steps_a = [env.step([KEEP] * 4)[0] for _ in range(12)]
        again, info_b = env.reset(seed=20001)
        steps_b = [env.step([KEEP] * 4)[0] for _ in range(12)]
        _, info_c = env.reset(seed=20002)
    finally:
        env.close()
    assert info_a["pulse_offset"] == info_b["pulse_offset"] != info_c["pulse_offset"]
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(np.array(steps_a), np.array(steps_b))
