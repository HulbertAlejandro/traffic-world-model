from __future__ import annotations

import pytest

from configs.reward import RewardConfig
from environments.project_reward_function import ProjectRewardFunction

# Congestion terms at zero, so the reward is exactly -delta * phase_penalty.
BASE_INFO = {"waiting_total": 0.0, "queue_total": 0.0, "throughput": 0.0}


def test_default_mode_penalizes_requested_phase_1() -> None:
    reward_function = ProjectRewardFunction()
    info = {**BASE_INFO, "phase_change": 1.0, "phase_switched": 0.0}

    assert reward_function.config.phase_penalty == "requested_phase_1"
    assert reward_function.compute(None, 1, None, info) == pytest.approx(-0.1)


def test_actual_switch_mode_penalizes_only_real_switches() -> None:
    reward_function = ProjectRewardFunction(RewardConfig(phase_penalty="actual_switch"))
    requested_but_blocked = {**BASE_INFO, "phase_change": 1.0, "phase_switched": 0.0}
    switched_to_phase_0 = {**BASE_INFO, "phase_change": 0.0, "phase_switched": 1.0}

    assert reward_function.compute(None, 1, None, requested_but_blocked) == pytest.approx(0.0)
    assert reward_function.compute(None, 0, None, switched_to_phase_0) == pytest.approx(-0.1)


def test_phase_penalty_rejects_invalid_mode_and_missing_key() -> None:
    with pytest.raises(ValueError, match="phase_penalty must be one of"):
        RewardConfig(phase_penalty="changes")

    reward_function = ProjectRewardFunction(RewardConfig(phase_penalty="actual_switch"))
    with pytest.raises(KeyError, match="phase_switched"):
        reward_function.compute(None, 1, None, {**BASE_INFO, "phase_change": 1.0})
