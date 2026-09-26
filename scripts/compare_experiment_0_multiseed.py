"""Experimento 0 with several training seeds per branch, compared seed by seed.

Each seed folder of the z branch (LatentDynamicsLSTM on the Autoencoder's z)
is paired with the folder of the SAME seed of the raw branch (same LSTM on the
raw normalized state). Both are evaluated exactly like scripts/evaluate_world_model.py
and scripts/evaluate_world_model_raw.py: same test split, autoregressive rollouts
with the true actions, horizons 1-10, reward de-normalized with the scaler saved
next to each checkpoint.

Decision metric, as in compare_experiment_0.py: reward_mse (the only metric
comparable between the two input spaces). Reported per horizon:
- how many seed pairs the z branch wins (paired by seed, not mean vs mean);
- the median, over seed pairs, of the relative reduction (raw - z) / raw
  (positive = the Autoencoder helps).
At the longest horizon it also reports, per seed pair, the median over test
episodes of each episode's reward error and how many episodes each branch wins:
with 12 test episodes, two or three congested ones carry 40-60% of the mean
error there, so the mean alone can be decided by a single episode.

Usage:
    python scripts/compare_experiment_0_multiseed.py \\
        --z-dirs models/checkpoints/exp0_multiseed/z/seed0 ... \\
        --raw-dirs models/checkpoints/exp0_multiseed/raw/seed0 ... \\
        --output docs/results/experiment_0_multiseed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.world_model_evaluation import (
    aggregate_by_horizon,
    load_episodes,
    load_reward_scaler,
    load_world_model,
    rollout_episode,
)

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
MAX_HORIZON = 10


def evaluate(checkpoint: Path, test_path: Path, device: torch.device):
    """Same computation as evaluate_world_model.py, returned instead of written to results/.
    Also returns, per test episode, the mean reward squared error at MAX_HORIZON."""
    model, hparams = load_world_model(checkpoint, device)
    scaler = load_reward_scaler(checkpoint)
    episodes = load_episodes(test_path)
    records = []
    per_episode = {}
    for episode_id, episode in episodes.items():
        episode_records = rollout_episode(
            model, episode, hparams["sequence_length"], hparams["action_dim"], MAX_HORIZON, device,
            reward_mean=scaler["reward_mean"], reward_std=scaler["reward_std"],
        )
        records.extend(episode_records)
        per_episode[str(episode_id)] = float(
            np.mean([r["model_reward_se"] for r in episode_records if r["horizon"] == MAX_HORIZON])
        )
    return aggregate_by_horizon(records, hparams["latent_dim"]), per_episode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--z-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.z_dirs) != len(args.raw_dirs):
        parser.error("--z-dirs and --raw-dirs must list the same seeds, in the same order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = []
    for z_dir, raw_dir in zip(args.z_dirs, args.raw_dirs):
        z_seed = json.loads((z_dir / "world_model_best.json").read_text(encoding="utf-8")).get("seed")
        raw_seed = json.loads((raw_dir / "world_model_raw_best.json").read_text(encoding="utf-8")).get("seed")
        if z_seed != raw_seed:
            parser.error(f"seed mismatch in pair {z_dir} (seed {z_seed}) / {raw_dir} (seed {raw_seed})")
        z, z_episodes = evaluate(z_dir / "world_model_best.pt", PROCESSED_DIR / "test_latent.npz", device)
        raw, raw_episodes = evaluate(raw_dir / "world_model_raw_best.pt", PROCESSED_DIR / "test_raw_seq.npz", device)
        pairs.append({"seed": z_seed, "z_dir": z_dir.as_posix(), "raw_dir": raw_dir.as_posix(),
                      "z": {str(h): v for h, v in z.items()}, "raw": {str(h): v for h, v in raw.items()},
                      f"z_h{MAX_HORIZON}_per_episode": z_episodes,
                      f"raw_h{MAX_HORIZON}_per_episode": raw_episodes})
        print(f"seed {z_seed}: reward_mse h=1 z={z[1]['model_reward_mse']:.3f} raw={raw[1]['model_reward_mse']:.3f} | "
              f"h=10 z={z[MAX_HORIZON]['model_reward_mse']:.3f} raw={raw[MAX_HORIZON]['model_reward_mse']:.3f}")

    horizons = [str(h) for h in range(1, MAX_HORIZON + 1)]
    n = len(pairs)
    summary = {}
    print()
    print(f"{'h':>3} | {'z gana (pares)':>14} | {'reduccion mediana (raw-z)/raw':>30} | "
          f"{'reward_mse z (mediana)':>22} | {'reward_mse crudo (mediana)':>26}")
    print("-" * 110)
    all_reductions = []
    for h in horizons:
        z_mse = np.array([p["z"][h]["model_reward_mse"] for p in pairs])
        raw_mse = np.array([p["raw"][h]["model_reward_mse"] for p in pairs])
        reductions = (raw_mse - z_mse) / raw_mse
        all_reductions.extend(reductions.tolist())
        summary[h] = {
            "z_wins": int((z_mse < raw_mse).sum()),
            "pairs": n,
            "median_reduction": float(np.median(reductions)),
            "reductions": reductions.tolist(),
            "median_z_reward_mse": float(np.median(z_mse)),
            "median_raw_reward_mse": float(np.median(raw_mse)),
        }
        s = summary[h]
        print(f"{h:>3} | {s['z_wins']:>9}/{n:<4} | {100 * s['median_reduction']:>29.1f}% | "
              f"{s['median_z_reward_mse']:>22.3f} | {s['median_raw_reward_mse']:>26.3f}")

    total_wins = sum(s["z_wins"] for s in summary.values())
    median_of_horizon_medians = float(np.median([s["median_reduction"] for s in summary.values()]))
    print()
    print(f"z gana {total_wins}/{n * len(horizons)} pares (semilla x horizonte). "
          f"Reduccion mediana por horizonte (mediana de las medianas): {100 * median_of_horizon_medians:.1f}%. "
          f"Mediana sobre los {len(all_reductions)} pares: {100 * float(np.median(all_reductions)):.1f}% "
          "(positivo = el Autoencoder ayuda).")

    print()
    print(f"h={MAX_HORIZON}, por episodio de test (misma semilla): mediana por episodio y episodios ganados")
    print(f"{'sem.':>4} | {'z gana ep.':>10} | {'mediana ep. z':>13} | {'mediana ep. crudo':>17} | {'media z':>8} | {'media crudo':>11}")
    episode_summary = []
    for p in pairs:
        ze, re_ = p[f"z_h{MAX_HORIZON}_per_episode"], p[f"raw_h{MAX_HORIZON}_per_episode"]
        wins = sum(ze[e] < re_[e] for e in ze)
        row = {"seed": p["seed"], "z_wins_episodes": wins, "episodes": len(ze),
               "median_episode_z": float(np.median(list(ze.values()))),
               "median_episode_raw": float(np.median(list(re_.values())))}
        episode_summary.append(row)
        print(f"{p['seed']:>4} | {wins:>6}/{len(ze):<3} | {row['median_episode_z']:13.1f} | {row['median_episode_raw']:17.1f} | "
              f"{p['z'][str(MAX_HORIZON)]['model_reward_mse']:8.1f} | {p['raw'][str(MAX_HORIZON)]['model_reward_mse']:11.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "pairs": pairs,
        "per_horizon": summary,
        "z_wins_total": total_wins,
        "median_of_horizon_median_reductions": median_of_horizon_medians,
        "median_reduction_all_pairs": float(np.median(all_reductions)),
        f"h{MAX_HORIZON}_per_episode_summary": episode_summary,
    }, indent=2), encoding="utf-8")
    print(f"Resultados: {args.output}")


if __name__ == "__main__":
    main()
