import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.reward import RewardConfig
from environments import TrafficEnvironment
from environments.traffic_state import TrafficState


def test_environment():
    """Verifica que el wrapper del entorno funcione correctamente."""
    env = TrafficEnvironment()
    try:
        obs, _ = env.reset()
        assert obs is not None

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(int(action))

        assert isinstance(float(reward), float)
        assert "throughput" in info
        assert "waiting_total" in info
        assert "queue_total" in info
    finally:
        env.close()


def test_traffic_state_round_trip():
    """Verifica que el estado del dominio se convierta a vector sin perder forma."""
    phase_one_hot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    state = TrafficState(
        vehicle_counts=np.array([1, 2, 3, 4], dtype=np.float32),
        queue_lengths=np.array([1, 2, 3, 4], dtype=np.float32),
        waiting_times=np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        mean_speeds=np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        occupancies=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        phase_one_hot=phase_one_hot,
        elapsed_phase_time=8.0,
        remaining_phase_time=12.5,
    )

    vector = state.to_vector()

    assert state.size == 26  # 4 lanes * 5 + 4 phases + 2
    assert vector.shape == (26,)
    assert vector.dtype == np.float32
    assert float(vector[0]) == 1.0
    assert float(vector[-2]) == 8.0
    assert float(vector[-1]) == 12.5


# Phase timing depends only on the actions (delta_time=5, yellow_time=2,
# min_green=5) and every reset starts at green_phase=0 with 0 s elapsed, so this
# sequence is deterministic regardless of the traffic seed: blocked request for
# phase 1, still blocked, request of the current phase, switch 0 -> 1, blocked
# right after a switch, switch 1 -> 0.
PHASE_ACTIONS = [1, 1, 0, 1, 0, 0]
EXPECTED_PHASE_CHANGE = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
EXPECTED_PHASE_SWITCHED = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def _run_phase_sequence(env):
    env.reset(seed=0)
    return [env.step(action) for action in PHASE_ACTIONS]


def test_phase_switched_measures_actual_changes():
    env = TrafficEnvironment()
    try:
        steps = _run_phase_sequence(env)
    finally:
        env.close()

    assert [info["phase_change"] for *_, info in steps] == EXPECTED_PHASE_CHANGE
    assert [info["phase_switched"] for *_, info in steps] == EXPECTED_PHASE_SWITCHED


def test_reward_config_selects_the_phase_penalty():
    default_env = TrafficEnvironment()
    switch_env = TrafficEnvironment(reward_config=RewardConfig(phase_penalty="actual_switch"))
    try:
        default_steps = _run_phase_sequence(default_env)
        switch_steps = _run_phase_sequence(switch_env)
    finally:
        default_env.close()
        switch_env.close()

    delta = RewardConfig().delta
    for (_, default_reward, *_, info), (_, switch_reward, *_) in zip(default_steps, switch_steps):
        expected_gap = delta * (info["phase_change"] - info["phase_switched"])
        assert switch_reward - default_reward == pytest.approx(expected_gap)


def test_arrivals_total_matches_an_independent_per_second_count():
    """info["arrivals_total"] must equal the arrivals counted second by second
    over the whole control interval (a step runs delta_time simulated seconds).
    The legacy info["throughput"] only saw the last second of each interval."""
    env = TrafficEnvironment()
    try:
        env.reset(seed=123)
        sumo_env = env.env
        per_second = {"arrived": 0}
        original_step = sumo_env._sumo_step

        def counting_step():
            original_step()
            per_second["arrived"] += sumo_env.sumo.simulation.getArrivedNumber()

        sumo_env._sumo_step = counting_step
        reported = []
        for step in range(20):
            before = per_second["arrived"]
            *_, info = env.step(step % 2)
            reported.append(info["arrivals_total"])
            assert info["arrivals_total"] == per_second["arrived"] - before
    finally:
        env.close()

    # Sanity check: the 20 steps (100 simulated seconds) really had arrivals to count.
    assert sum(reported) > 0
