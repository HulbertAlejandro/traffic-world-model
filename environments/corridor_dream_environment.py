"""v2 Dream Environment: imagine corridor transitions with a trained temporal model, never SUMO.

Same design as v1's environments/dream_environment.py (read its docstring for the reasons behind
each choice), applied to the corridor (docs/v2/DISENO_CONTROL.md, section 2.5):

- Each imagined episode is seeded with a REAL window of ``sequence_length`` (16) steps from a
  recorded episode of the train split; only the action paired with the last state of the window
  is hypothetical, the earlier ones are the recorded actions.
- The state is the 104-dim corridor state normalized with datasets/v2/processed/scaler.pkl (there
  is no Encoder in v2), the same values the models were trained on.
- The action is the corridor's joint action, MultiDiscrete([2, 2, 2, 2]) (keep/switch per signal),
  turned into the model's 8-wide input by ``encode_actions``, the function that built the training
  windows -- there is no second encoding that could drift from it.
- The temporal model is a parameter: ``model_dir`` holds a world_model_best.json/.pt pair of any
  architecture ``build_temporal_model`` knows (LSTM or Transformer here), and its reward_scaler.json.
- ``max_dream_steps`` = 7 and the reward clip to the 1st/99th percentile of the train split's real
  rewards are v1's rules; the percentiles are recomputed for the v2 dataset (constants below).

Shared with v1 and with the world-model evaluation, imported: ``predict_next_step`` (windowing and
reward de-normalization) and ``load_episodes``.

``window_alignment`` (docs/v2/ADDENDUM_PLANIFICACION.md, section 11):
- ``"legacy"`` (default, every Phase 3 result): after a step the action window is shifted from the
  window as it was BEFORE the chosen action was inserted, as v1's DreamEnvironment does. The action
  applied at a state is therefore never paired with that state in later windows; from the third
  imagined step (the second, if the action differs from the recorded one) each state carries the
  previous step's action. Kept so published results reproduce exactly.
- ``"aligned"``: the window is shifted from the window WITH the chosen action, so every state stays
  paired with the action applied at it -- Experiment 1's rollout_episode alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from datasets.latent_sequence_dataset import encode_actions
from evaluation.world_model_evaluation import load_episodes, predict_next_step
from training.v2_compression_experiment import build_temporal_model

ROOT_DIR = Path(__file__).resolve().parents[1]
ARCH_COMPARISON_DIR = ROOT_DIR / "models" / "checkpoints" / "v2" / "arch_comparison"
SEQUENCE_DATA_DIR = ROOT_DIR / "models" / "checkpoints" / "v2" / "compression" / "selection" / "raw_data"
DEFAULT_SEED_EPISODES_PATH = SEQUENCE_DATA_DIR / "train_seq.npz"

# 1st/99th percentile of the real rewards of the v2 train split (datasets/v2/processed/train.npz,
# 6,720 transitions; min -777.0, mean -45.65), computed from that file on 4 October 2026. Same rule
# as v1's REWARD_CLIP_MIN/MAX; recompute if the dataset is regenerated.
REWARD_CLIP_MIN = -345.43
REWARD_CLIP_MAX = 0.0
WINDOW_ALIGNMENTS = ("legacy", "aligned")


def world_model_dir(architecture: str, seed: int) -> Path:
    """The parameter-matched world model of the given architecture and seed (ADDENDUM_CONTROL.md, 1)."""
    return ARCH_COMPARISON_DIR / f"{architecture}_raw_s{seed}"


def load_temporal_model(model_dir: str | Path) -> tuple[torch.nn.Module, dict, dict]:
    """(frozen model on CPU, its world_model_best.json, its reward_scaler.json)."""
    model_dir = Path(model_dir)
    hparams = json.loads((model_dir / "world_model_best.json").read_text(encoding="utf-8"))
    reward_scaler = json.loads((model_dir / "reward_scaler.json").read_text(encoding="utf-8"))
    model = build_temporal_model(hparams)
    model.load_state_dict(torch.load(model_dir / "world_model_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, hparams, reward_scaler


class CorridorDreamEnvironment(gym.Env):
    """Gymnasium env over the model's imagination of the corridor; usable by SB3's PPO as is."""

    def __init__(self, model_dir: str | Path, seed_episodes_path: str | Path = DEFAULT_SEED_EPISODES_PATH,
                 max_dream_steps: int = 7, eval_seed: int | None = None, window_alignment: str = "legacy") -> None:
        super().__init__()
        if max_dream_steps <= 0:
            raise ValueError(f"max_dream_steps must be positive, got {max_dream_steps}")
        if window_alignment not in WINDOW_ALIGNMENTS:
            raise ValueError(f"window_alignment must be one of {WINDOW_ALIGNMENTS}, got {window_alignment!r}")
        self.window_alignment = window_alignment
        self.max_dream_steps = max_dream_steps
        self.device = torch.device("cpu")
        self.model, hparams, reward_scaler = load_temporal_model(model_dir)
        self.architecture = hparams["architecture"]
        self.state_dim = hparams["latent_dim"]
        self.action_dim = hparams["action_dim"]
        self.sequence_length = hparams["sequence_length"]
        self.n_signals = self.action_dim // 2
        self.reward_mean = reward_scaler["reward_mean"]
        self.reward_std = reward_scaler["reward_std"]

        self._seed_episodes = {ep_id: ep for ep_id, ep in load_episodes(seed_episodes_path).items()
                               if len(ep["z"]) >= self.sequence_length}
        if not self._seed_episodes:
            raise ValueError(f"No episode in {seed_episodes_path} has {self.sequence_length} steps.")
        self._episode_ids = sorted(self._seed_episodes)

        self.action_space = gym.spaces.MultiDiscrete([2] * self.n_signals)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,),
                                                dtype=np.float32)
        self._window_z: torch.Tensor | None = None
        self._window_actions: torch.Tensor | None = None
        self._steps_taken = 0
        self.seed_window: tuple[int, int] | None = None
        self._rng = np.random.default_rng(eval_seed)

    def encode_action(self, action) -> torch.Tensor:
        """The model's 8-wide input for one joint action: encode_actions on a 1-step sequence."""
        action = torch.as_tensor(np.asarray(action, dtype=np.int64)).reshape(1, self.n_signals)
        return encode_actions(action, self.action_dim)[0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """``options={"episode_id": e, "start": s}`` fixes the seed window (fidelity diagnostic);
        otherwise episode and start are drawn as in v1."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "episode_id" in options:
            episode_id = int(options["episode_id"])
            episode = self._seed_episodes[episode_id]
            start = int(options["start"])
            if not 0 <= start <= len(episode["z"]) - self.sequence_length:
                raise ValueError(f"start {start} out of range for episode {episode_id}")
        else:
            episode_id = self._episode_ids[self._rng.integers(len(self._episode_ids))]
            episode = self._seed_episodes[episode_id]
            start = int(self._rng.integers(len(episode["z"]) - self.sequence_length + 1))
        end = start + self.sequence_length
        self._window_z = torch.from_numpy(episode["z"][start:end]).clone()
        self._window_actions = encode_actions(torch.from_numpy(episode["actions"][start:end]), self.action_dim)
        self._steps_taken = 0
        self.seed_window = (episode_id, start)
        return self._window_z[-1].numpy().astype(np.float32), {"seeded_from_real_data": True,
                                                               "seed_episode": episode_id, "seed_start": start}

    def step(self, action):
        if self._window_z is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(np.asarray(action, dtype=np.int64)):
            raise ValueError(f"Invalid joint action: {action!r}")
        action_encoded = self.encode_action(action)
        # Only the LAST action of the window (paired with the current state) is hypothetical.
        action_window = self._window_actions.clone()
        action_window[-1] = action_encoded

        pred_z, pred_r = predict_next_step(self.model, self._window_z, action_window, self.device,
                                           self.reward_mean, self.reward_std)
        raw_reward = float(pred_r.item())
        reward = float(np.clip(raw_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

        self._window_z = torch.cat([self._window_z[1:], pred_z.unsqueeze(0)], dim=0)
        # "legacy" shifts the window from before the chosen action was inserted (see the module docstring);
        # "aligned" shifts the window that holds it. The new last row is a placeholder replaced next step.
        previous = self._window_actions if self.window_alignment == "legacy" else action_window
        self._window_actions = torch.cat([previous[1:], action_encoded.unsqueeze(0)], dim=0)
        self._steps_taken += 1
        truncated = self._steps_taken >= self.max_dream_steps
        info = {"dream_step": self._steps_taken, "imagined": True,
                "reward_clipped": raw_reward != reward, "raw_predicted_reward": raw_reward}
        return pred_z.numpy().astype(np.float32), reward, False, truncated, info

    def close(self) -> None:
        pass
