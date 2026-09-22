"""Evaluate the trained latent-dynamics World Model (LatentDynamicsLSTM).

Implements Experimento 1 (Sección 20) and Objetivo específico 6 of the
project proposal: measure one-step and multi-step (autoregressive)
predictive error on the held-out test split, quantify how the error
compounds with the horizon, and compare against a persistence baseline
(Sección 32's "supera un baseline simple de predicción" criterion).

Usage:
    python scripts/evaluate_world_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.world_model_evaluation import (
    aggregate_by_horizon,
    load_episodes,
    load_world_model,
    plot_compounding_error,
    rollout_episode,
    save_metrics_report,
)

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_PATH = ROOT_DIR / "models" / "checkpoints" / "world_model_best.pt"
RESULTS_DIR = ROOT_DIR / "results"
TEST_LATENT_PATH = PROCESSED_DIR / "test_latent.npz"

# Upper bound on how many autoregressive steps to evaluate. The actual
# horizon used is capped by the shortest test episode (see main()), so this
# is a ceiling, not a guarantee.
MAX_HORIZON = 10


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not TEST_LATENT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_LATENT_PATH}. Run scripts/encode_latent_dataset.py first."
        )
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT_PATH}. Run training/train_world_model.py first."
        )

    model, hparams = load_world_model(CHECKPOINT_PATH, device)
    sequence_length = hparams["sequence_length"]
    action_dim = hparams["action_dim"]
    latent_dim = hparams["latent_dim"]

    episodes = load_episodes(TEST_LATENT_PATH)
    if not episodes:
        raise ValueError(f"No episodes found in {TEST_LATENT_PATH}.")

    shortest_episode = min(len(ep["z"]) for ep in episodes.values())
    max_horizon = min(MAX_HORIZON, shortest_episode - sequence_length)
    if max_horizon < 1:
        raise ValueError(
            f"No test episode is long enough for sequence_length={sequence_length} "
            "plus at least a 1-step horizon. Collect longer episodes or reduce "
            "sequence_length."
        )

    all_records = []
    skipped_episodes = 0
    for episode in episodes.values():
        records = rollout_episode(
            model, episode, sequence_length, action_dim, max_horizon, device
        )
        if not records:
            skipped_episodes += 1
            continue
        all_records.extend(records)

    if not all_records:
        raise ValueError(
            "No episode in the test split produced a valid rollout window for "
            f"sequence_length={sequence_length} + max_horizon={max_horizon}."
        )

    summary = aggregate_by_horizon(all_records, latent_dim)

    print(f"Test episodes: {len(episodes)} (skipped as too short: {skipped_episodes})")
    print(f"sequence_length={sequence_length} | max_horizon={max_horizon}")
    print()
    header = (
        f"{'h':>3} | {'latent_mse (model)':>18} | {'latent_mse (baseline)':>21} | "
        f"{'reward_mse (model)':>18} | {'reward_mse (baseline)':>21} | n"
    )
    print(header)
    print("-" * len(header))
    for horizon, metrics in summary.items():
        print(
            f"{horizon:>3} | {metrics['model_latent_mse']:>18.6f} | "
            f"{metrics['baseline_latent_mse']:>21.6f} | "
            f"{metrics['model_reward_mse']:>18.6f} | "
            f"{metrics['baseline_reward_mse']:>21.6f} | {metrics['n_samples']}"
        )

    one_step = summary[1]
    print()
    if one_step["model_latent_mse"] < one_step["baseline_latent_mse"]:
        print("El World Model SUPERA al baseline persistente a un paso (latente).")
    else:
        print("ADVERTENCIA: el World Model NO supera al baseline persistente a un paso (latente).")

    if one_step["model_reward_mse"] < one_step["baseline_reward_mse"]:
        print("El World Model SUPERA al baseline persistente a un paso (recompensa).")
    else:
        print("ADVERTENCIA: el World Model NO supera al baseline persistente a un paso (recompensa).")

    last_horizon = max(summary)
    growth = summary[last_horizon]["model_latent_mse"] / max(summary[1]["model_latent_mse"], 1e-12)
    print(
        f"\nCompounding error: el MSE latente del modelo crece {growth:.2f}x "
        f"entre el horizonte 1 y el horizonte {last_horizon}."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "world_model_evaluation.json"
    plot_path = RESULTS_DIR / "world_model_compounding_error.png"
    save_metrics_report(summary, report_path)
    plot_compounding_error(summary, plot_path)

    print(f"\nReporte guardado en: {report_path}")
    print(f"Gráfica guardada en: {plot_path}")


if __name__ == "__main__":
    main()