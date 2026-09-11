import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import TrafficEnvironment
from environments.traffic_state import TrafficState


def test_environment():
    """Verifica que el wrapper del entorno funcione correctamente."""

    env = TrafficEnvironment()

    try:
        obs, _ = env.reset()

        assert obs is not None

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        assert isinstance(float(reward), float)

    finally:
        env.close()


def test_traffic_state_round_trip():
    """Verifica que el estado del dominio se convierta a vector sin perder forma."""

    state = TrafficState(
        vehicle_counts=np.array([1, 2, 3, 4], dtype=np.float32),
        queue_lengths=np.array([1, 2, 3, 4], dtype=np.float32),
        waiting_times=np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        mean_speeds=np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        occupancies=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        current_phase=1,
        remaining_phase_time=12.5,
    )

    vector = state.to_vector()

    assert state.size == 22
    assert vector.shape == (22,)
    assert vector.dtype == np.float32
    assert float(vector[0]) == 1.0
    assert float(vector[-2]) == 1.0
    assert float(vector[-1]) == 12.5