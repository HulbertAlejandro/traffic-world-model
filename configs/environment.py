"""Environment-level configuration for the SUMO simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class EnvironmentConfig:
    """Default configuration values used by the project environment."""

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
    max_episode_steps: int = 300
