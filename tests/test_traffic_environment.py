import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments import TrafficEnvironment


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