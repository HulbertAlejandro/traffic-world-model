"""Environment-level configuration for the SUMO simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class EnvironmentConfig:
    """Default configuration values used by the project environment.

    Episode length is governed by ``simulation_seconds`` alone (passed as
    ``num_seconds`` to ``sumo_rl.SumoEnvironment``). A separate
    ``max_episode_steps`` field existed previously but was never read
    anywhere in the project; removed to avoid two disconnected knobs that
    could silently disagree about how long an episode runs.
    """

    net_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "environments" / "single-intersection" / "single-intersection.net.xml"
    )
    route_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "environments" / "single-intersection" / "single-intersection.rou.xml"
    )
    use_gui: bool = False
    simulation_seconds: int = 300
    single_agent: bool = True
    seed: int = 42