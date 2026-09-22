"""Dream Environment: imagine trajectories using the trained LatentDynamicsLSTM,
without ever touching SUMO.

This is the core mechanism the project's research question depends on: whether
a controller can learn (or plan) using a model-imagined environment instead of
interacting with the real simulator every time.

Design
------
- Each imagined episode is SEEDED with a real window of `sequence_length` steps
  taken from an actual recorded episode (the model was trained on windows of
  this exact length; it cannot meaningfully predict from less context). The
  historical (z, action) pairs in that seed window are real, factual data --
  only the actions chosen from ``reset()`` onward, via ``step()``, are
  hypothetical.
- ``step(action)`` replaces ONLY the last action in the current window (the one
  paired with the current/last z, i.e. the action about to be taken) with the
  agent's chosen action, then calls ``predict_next_step`` (shared with
  ``evaluation.world_model_evaluation.rollout_episode``, so the windowing logic
  has a single source of truth for both evaluation and imagination).
- ``max_dream_steps`` caps how long an imagined episode runs before being
  truncated. Default is 10, matching the horizon range actually validated in
  Experimento 1 (evaluate_world_model.py measured up to horizon 10); beyond
  that, no evidence exists about how reliable the model's predictions are.
- This environment operates entirely in latent (z) space. It never calls the
  Encoder live -- it reuses already-encoded episodes (the same *_latent.npz
  files produced by scripts/encode_latent_dataset.py) as the source of seed
  windows, and predicts purely in z-space from there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.world_model_evaluation import (
    load_episodes,
    load_reward_scaler,
    load_world_model,
    predict_next_step,
)

DEFAULT_LATENT_EPISODES_PATH = ROOT_DIR / "datasets" / "processed" / "train_latent.npz"
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "models" / "checkpoints" / "world_model_best.pt"


class DreamEnvironment(gym.Env):
    """Gymnasium-compatible environment that imagines transitions with the
    trained LatentDynamicsLSTM, seeded from real recorded episodes.

    Intended to be used exactly like TrafficEnvironment from the controller's
    point of view (e.g. passed directly to Stable-Baselines3's PPO), except
    every step happens inside the model's imagination, never inside SUMO.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        latent_episodes_path: str | Path = DEFAULT_LATENT_EPISODES_PATH,
        max_dream_steps: int = 10,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if max_dream_steps <= 0:
            raise ValueError(f"max_dream_steps must be positive, got {max_dream_steps}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_dream_steps = max_dream_steps

        self.model, hparams = load_world_model(checkpoint_path, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.latent_dim = hparams["latent_dim"]
        self.action_dim = hparams["action_dim"]
        self.sequence_length = hparams["sequence_length"]

        reward_scaler = load_reward_scaler(checkpoint_path)
        self.reward_mean = reward_scaler["reward_mean"]
        self.reward_std = reward_scaler["reward_std"]

        all_episodes = load_episodes(latent_episodes_path)
        self._seed_episodes = {
            ep_id: ep for ep_id, ep in all_episodes.items() if len(ep["z"]) >= self.sequence_length
        }
        if not self._seed_episodes:
            raise ValueError(
                f"No episode in {latent_episodes_path} has at least "
                f"{self.sequence_length} steps -- cannot seed any dream window."
            )

        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32,
        )

        self._window_z: torch.Tensor | None = None
        self._window_actions_onehot: torch.Tensor | None = None
        self._steps_taken = 0
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        episode_ids = list(self._seed_episodes.keys())
        episode = self._seed_episodes[episode_ids[self._rng.integers(len(episode_ids))]]

        max_start = len(episode["z"]) - self.sequence_length
        start = int(self._rng.integers(max_start + 1))

        z = torch.from_numpy(episode["z"][start : start + self.sequence_length])
        actions = torch.from_numpy(episode["actions"][start : start + self.sequence_length])

        self._window_z = z.clone()
        self._window_actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        self._steps_taken = 0

        observation = self._window_z[-1].numpy().astype(np.float32)
        info: dict = {"seeded_from_real_data": True}
        return observation, info

    def step(self, action: int):
        if self._window_z is None:
            raise RuntimeError("Call reset() before step().")
        if action not in range(self.action_dim):
            raise ValueError(f"Invalid action: {action!r}")

        # Only the LAST action in the window (paired with the current/last z)
        # is hypothetical -- it is what we are choosing to imagine right now.
        # Everything before it in the window is real, factual history.
        action_window = self._window_actions_onehot.clone()
        action_onehot = F.one_hot(torch.tensor(action), num_classes=self.action_dim).float()
        action_window[-1] = action_onehot

        pred_z, pred_r = predict_next_step(
            self.model,
            self._window_z,
            action_window,
            self.device,
            self.reward_mean,
            self.reward_std,
        )

        # Roll the window forward: drop the oldest step, append the imagined
        # transition. The action just taken becomes the new last "historical"
        # action, to be overwritten again on the next step() call.
        self._window_z = torch.cat([self._window_z[1:], pred_z.unsqueeze(0)], dim=0)
        self._window_actions_onehot = torch.cat(
            [self._window_actions_onehot[1:], action_onehot.unsqueeze(0)], dim=0
        )

        self._steps_taken += 1
        terminated = False  # the traffic dynamics have no terminal state, same as TrafficEnvironment
        truncated = self._steps_taken >= self.max_dream_steps

        observation = pred_z.numpy().astype(np.float32)
        reward = float(pred_r.item())
        info = {"dream_step": self._steps_taken, "imagined": True}

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        pass
