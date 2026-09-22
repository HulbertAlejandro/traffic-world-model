"""Bridge between the real SUMO environment and the PPO controller trained
inside the Dream Environment.

The PPO policy was trained entirely on z (the Autoencoder's latent space), so
it cannot consume TrafficEnvironment's raw 26-dimensional state directly. This
wrapper runs the frozen Encoder live, on every real step, translating
TrafficEnvironment's raw observations into z before the policy sees them --
this is the ONLY place in the project where the Encoder runs against live
SUMO data instead of pre-encoded episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.environment import EnvironmentConfig
from environments.traffic_environment import TrafficEnvironment
from evaluation.autoencoder_evaluation import load_autoencoder

DEFAULT_AUTOENCODER_CHECKPOINT = ROOT_DIR / "models" / "checkpoints" / "autoencoder_best.pt"


class EncodedTrafficEnvironment(gym.Env):
    """Wraps TrafficEnvironment, encoding every observation to z via the
    frozen Autoencoder before returning it -- so a policy trained on
    DreamEnvironment (z-space) can run against real SUMO unchanged."""

    def __init__(
        self,
        autoencoder_checkpoint: str | Path = DEFAULT_AUTOENCODER_CHECKPOINT,
        environment_config: EnvironmentConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._env = TrafficEnvironment(environment_config)

        input_dim = self._env.observation_space.shape[0]
        self.autoencoder = load_autoencoder(autoencoder_checkpoint, input_dim, self.device)
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Encode one dummy state to discover latent_dim from the real model,
        # instead of hardcoding it -- same "single source of truth" principle
        # used throughout this project (e.g. RepresentationConfig.input_dim).
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, device=self.device)
            self.latent_dim = self.autoencoder.encode(dummy).shape[-1]

        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32,
        )

    def _encode(self, raw_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.from_numpy(raw_state).float().unsqueeze(0).to(self.device)
            z = self.autoencoder.encode(state_tensor)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self, **kwargs):
        raw_state, info = self._env.reset(**kwargs)
        return self._encode(raw_state), info

    def step(self, action):
        raw_next_state, reward, terminated, truncated, info = self._env.step(action)
        return self._encode(raw_next_state), reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()
