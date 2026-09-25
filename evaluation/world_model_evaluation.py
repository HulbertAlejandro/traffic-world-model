"""Reusable evaluation utilities for the trained latent-dynamics World Model.

This module implements Experimento 1 (Sección 20) and Objetivo específico 6
of the project proposal: evaluate the World Model's predictive capability at
one step and at several autoregressive steps, quantifying the compounding
error that appears when predictions are projected forward using the model's
own previous outputs instead of ground-truth latents.

Design notes
------------
- One-step and multi-step evaluation share a single code path
  (``rollout_episode``) called with ``max_horizon=1`` or ``max_horizon=H``.
  This avoids re-deriving the windowing/index convention twice and keeps it
  tested once. The convention mirrors ``LatentSequenceDataset`` exactly:
  a window ``z[start:end]`` (with ``end = start + sequence_length``) predicts
  ``z[end]`` and ``rewards[end - 1]``.
- The comparison baseline is a *persistence* forecast: "the state and reward
  stay exactly as they were at the last observed true step". This is the
  simple baseline required by the proposal's success criteria (Sección 32:
  "El modelo predictivo supera un baseline simple de predicción") and needs
  no extra statistics from the training split, so it cannot leak information
  between splits.
- Actions fed into the rolled-forward window are always the *true* actions
  that were actually executed in the episode. This function evaluates the
  dynamics model in isolation (Experimento 1), not a policy imagining
  hypothetical actions -- that is the Dream Environment's job, which is a
  later, separate stage of the project.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from models.world_model import LatentDynamicsLSTM, LatentDynamicsTransformer, LatentDynamicsTSMixer


def build_world_model(hparams: dict) -> nn.Module:
    """Instantiate an untrained temporal model from its hyperparameter dict.

    ``hparams["architecture"]`` selects the class (``"lstm"``, ``"transformer"``
    or ``"tsmixer"``, Experimento 3). Checkpoints written before Experimento 3
    have no ``architecture`` key and are always LSTMs, so that is the default.
    """
    architecture = hparams.get("architecture", "lstm")
    if architecture == "lstm":
        return LatentDynamicsLSTM(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            hidden_dim=hparams["hidden_dim"],
            sequence_length=hparams["sequence_length"],
        )
    if architecture == "transformer":
        return LatentDynamicsTransformer(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            num_layers=hparams["num_layers"],
            dim_feedforward=hparams["dim_feedforward"],
            sequence_length=hparams["sequence_length"],
            dropout=hparams["dropout"],
        )
    if architecture == "tsmixer":
        return LatentDynamicsTSMixer(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            hidden_dim=hparams["hidden_dim"],
            num_blocks=hparams["num_blocks"],
            sequence_length=hparams["sequence_length"],
            dropout=hparams["dropout"],
        )
    raise ValueError(
        f"Unknown world model architecture {architecture!r}; "
        "expected 'lstm', 'transformer' or 'tsmixer'."
    )


def load_world_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load a trained temporal model from its checkpoint + hyperparameters.

    Mirrors ``evaluation.autoencoder_evaluation.load_autoencoder``: the
    ``.json`` file saved alongside the checkpoint by the training script is
    the source of truth for the model's architecture and shape, so the caller
    never has to hardcode ``latent_dim``/``action_dim``.
    """
    checkpoint_path = Path(checkpoint_path)
    hparams_path = checkpoint_path.with_suffix(".json")
    if not hparams_path.exists():
        raise FileNotFoundError(
            f"Missing hyperparameter file for checkpoint: {hparams_path}"
        )
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))

    model = build_world_model(hparams).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def load_reward_scaler(checkpoint_path: str | Path) -> dict[str, float]:
    """Load the reward normalization stats saved by training/train_world_model.py.

    Lives next to the checkpoint under a fixed name (not the checkpoint's own
    stem) because it is a property of the training run, shared by both the
    ``_last`` and ``_best`` checkpoints it produced.
    """
    scaler_path = Path(checkpoint_path).parent / "reward_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing reward scaler file: {scaler_path}. Run training/train_world_model.py first."
        )
    return json.loads(scaler_path.read_text(encoding="utf-8"))


def load_episodes(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """Group an encoded latent dataset (``z``/``actions``/``rewards``) by episode.

    Returns ``{episode_id: {"z": (T, latent_dim), "actions": (T,), "rewards": (T,)}}``,
    each array sorted by ``time_step``. ``next_z`` is intentionally not
    returned: for a continuous episode ``next_z[i] == z[i + 1]``, so working
    directly with ``z`` is enough to roll a trajectory forward and is one
    fewer array to keep in sync.
    """
    with np.load(Path(path)) as data:
        z = data["z"].astype(np.float32)
        actions = data["actions"].astype(np.int64)
        rewards = data["rewards"].astype(np.float32)
        episode_id = data["episode_id"].astype(np.int64)
        time_step = data["time_step"].astype(np.int64)

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for ep_id in np.unique(episode_id):
        mask = episode_id == ep_id
        order = np.argsort(time_step[mask])
        episodes[int(ep_id)] = {
            "z": z[mask][order],
            "actions": actions[mask][order],
            "rewards": rewards[mask][order],
        }
    return episodes


@torch.no_grad()
def predict_next_step(
    model: LatentDynamicsLSTM,
    window_z: torch.Tensor,
    window_actions_onehot: torch.Tensor,
    device: torch.device,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the next latent state and de-normalized reward for one window.

    ``window_z`` has shape (sequence_length, latent_dim), ``window_actions_onehot``
    has shape (sequence_length, action_dim) -- both CPU tensors for a single
    (non-batched) window. Returns CPU tensors: ``pred_z`` (latent_dim,) and
    ``pred_r`` (scalar tensor), the latter already de-normalized to real units.

    Shared by ``rollout_episode`` (Experimento 1, fed the episode's true recorded
    actions) and ``environments.dream_environment.DreamEnvironment`` (fed a
    policy's candidate/imagined actions) -- the windowing and de-normalization
    logic is a single source of truth for both, instead of being duplicated.
    """
    pred_z, pred_r = model(
        window_z.unsqueeze(0).to(device),
        window_actions_onehot.unsqueeze(0).to(device),
    )
    pred_z = pred_z.squeeze(0).cpu()
    pred_r = pred_r.squeeze(0).cpu()
    pred_r = pred_r * reward_std + reward_mean
    return pred_z, pred_r


@torch.no_grad()
def rollout_episode(
    model: LatentDynamicsLSTM,
    episode: dict[str, np.ndarray],
    sequence_length: int,
    action_dim: int,
    max_horizon: int,
    device: torch.device,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
) -> list[dict[str, float]]:
    """Autoregressively roll one episode forward and score every horizon.

    For every valid starting point, builds a true context window of length
    ``sequence_length`` and predicts ``max_horizon`` steps ahead, feeding
    each prediction back into the window for the next step (autoregressive
    rollout). At every horizon ``h`` (1-indexed) it records the squared
    error of the model's prediction *and* of the persistence baseline
    against the true latent/reward, so the two can be compared at exactly
    the same points.

    Episodes shorter than ``sequence_length + max_horizon`` cannot support
    even one full rollout and are skipped (returns ``[]`, same spirit as
    ``LatentSequenceDataset`` discarding short episodes).
    """
    z = torch.from_numpy(episode["z"])
    actions = torch.from_numpy(episode["actions"])
    rewards = torch.from_numpy(episode["rewards"])
    actions_onehot = F.one_hot(actions, num_classes=action_dim).float()

    num_steps = z.shape[0]
    last_start = num_steps - sequence_length - max_horizon
    if last_start < 0:
        return []

    records: list[dict[str, float]] = []

    for start in range(last_start + 1):
        # Rolling context: starts as ground truth, its last row gets replaced
        # by the model's own prediction as the rollout advances.
        window_z = z[start : start + sequence_length].clone()

        # Persistence baseline: "nothing changes after the last true observation".
        #
        # For the latent, "the last true observation" is z[start + L - 1], the
        # final state inside the context window -- a value the model window
        # already contains, never the value being predicted, so this is a
        # fair reference point for every horizon.
        #
        # For the reward the equivalent reference point is NOT
        # rewards[start + L - 1]: that index is exactly the horizon-1 target
        # (see the true_idx/true_reward derivation below -- the window's own
        # last (z, a) pair is what produces reward index start + L - 1), so
        # using it as the baseline would compare that target against itself
        # and trivially score a perfect (and meaningless) 0 error at h=1.
        # The reward one step further back, rewards[start + L - 2], is the
        # most recent reward a persistence predictor could actually know
        # about before the transition being scored happens.
        baseline_z = z[start + sequence_length - 1].clone()
        baseline_reward_idx = max(start + sequence_length - 2, 0)
        baseline_reward = rewards[baseline_reward_idx].clone()

        for h in range(1, max_horizon + 1):
            action_window = actions_onehot[start + h - 1 : start + h - 1 + sequence_length]
            pred_z, pred_r = predict_next_step(model, window_z, action_window, device, reward_mean, reward_std)

            true_idx = start + sequence_length + h - 1
            true_z = z[true_idx]
            true_reward = rewards[true_idx - 1]

            records.append(
                {
                    "horizon": h,
                    "model_latent_se": torch.sum((pred_z - true_z) ** 2).item(),
                    "model_reward_se": ((pred_r - true_reward) ** 2).item(),
                    "baseline_latent_se": torch.sum((baseline_z - true_z) ** 2).item(),
                    "baseline_reward_se": ((baseline_reward - true_reward) ** 2).item(),
                }
            )

            # Roll the window forward: drop the oldest step, append the prediction.
            window_z = torch.cat([window_z[1:], pred_z.unsqueeze(0)], dim=0)

    return records


def aggregate_by_horizon(
    records: list[dict[str, float]],
    latent_dim: int,
) -> dict[int, dict[str, float]]:
    """Average squared errors per horizon into per-dimension MSE.

    Latent error is divided by ``latent_dim`` so it reads as a per-dimension
    MSE, comparable across configurations with different ``latent_dim``.
    """
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"model_latent": [], "model_reward": [], "baseline_latent": [], "baseline_reward": []}
    )
    for record in records:
        bucket = buckets[record["horizon"]]
        bucket["model_latent"].append(record["model_latent_se"] / latent_dim)
        bucket["model_reward"].append(record["model_reward_se"])
        bucket["baseline_latent"].append(record["baseline_latent_se"] / latent_dim)
        bucket["baseline_reward"].append(record["baseline_reward_se"])

    summary: dict[int, dict[str, float]] = {}
    for horizon in sorted(buckets):
        bucket = buckets[horizon]
        summary[horizon] = {
            "model_latent_mse": float(np.mean(bucket["model_latent"])),
            "model_reward_mse": float(np.mean(bucket["model_reward"])),
            "baseline_latent_mse": float(np.mean(bucket["baseline_latent"])),
            "baseline_reward_mse": float(np.mean(bucket["baseline_reward"])),
            "n_samples": len(bucket["model_latent"]),
        }
    return summary


def save_metrics_report(summary: dict[int, dict[str, float]], output_path: str | Path) -> None:
    """Persist the per-horizon summary as JSON for the thesis report/appendix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(horizon): metrics for horizon, metrics in summary.items()}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_compounding_error(
    summary: dict[int, dict[str, float]],
    output_path: str | Path,
    model_label: str = "World Model (LSTM)",
) -> None:
    """Plot latent and reward MSE vs. horizon for the model and the baseline."""
    horizons = sorted(summary)
    model_latent = [summary[h]["model_latent_mse"] for h in horizons]
    baseline_latent = [summary[h]["baseline_latent_mse"] for h in horizons]
    model_reward = [summary[h]["model_reward_mse"] for h in horizons]
    baseline_reward = [summary[h]["baseline_reward_mse"] for h in horizons]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(horizons, model_latent, marker="o", label=model_label)
    axes[0].plot(horizons, baseline_latent, marker="x", linestyle="--", label="Baseline persistente")
    axes[0].set_xlabel("Horizonte (pasos)")
    axes[0].set_ylabel("MSE latente (por dimensión)")
    axes[0].set_title("Error de predicción latente vs. horizonte")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(horizons, model_reward, marker="o", label=model_label)
    axes[1].plot(horizons, baseline_reward, marker="x", linestyle="--", label="Baseline persistente")
    axes[1].set_xlabel("Horizonte (pasos)")
    axes[1].set_ylabel("MSE de recompensa")
    axes[1].set_title("Error de predicción de recompensa vs. horizonte")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)