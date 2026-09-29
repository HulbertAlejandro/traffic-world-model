"""Exploratory latent_dim sweep (see MANIFEST.md). Not part of the official pipeline.

Every training/evaluation step IMPORTS the official protocol instead of copying it:
- Autoencoder: train_one_epoch / validate / create_dataloader / save_checkpoint from
  training/train_autoencoder.py, same seeding and call order as its main().
- LSTM: _set_seeds / train_one_epoch / validate / compute_reward_scaler and the constants
  from training/train_world_model.py, same call order as train_world_model_raw.py's main().
- Evaluation: evaluate() from scripts/compare_experiment_0_multiseed.py.
Only the input/output paths differ, so the runners can be checked against official weights.

Subcommands:
    make-data                         24-dim splits (drop columns 22, 23) -> DATA_DIR
    train-ae --latent K [--data-dir D] [--out O]
    train-lstm --train T --val V --seed S --out O
    compare --z-dirs ... --raw-dirs ... --test-z P --test-raw P --output J
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from configs import RepresentationConfig, WorldModelConfig  # noqa: E402
from datasets.latent_sequence_dataset import LatentSequenceDataset  # noqa: E402
from datasets.transition_dataset import TransitionDataset  # noqa: E402
from evaluation.autoencoder_evaluation import load_autoencoder  # noqa: E402
from models.representation import Autoencoder  # noqa: E402
from models.world_model import LatentDynamicsLSTM  # noqa: E402
from training import train_autoencoder as tae  # noqa: E402
from training import train_world_model as twm  # noqa: E402

_spec = importlib.util.spec_from_file_location("cmp0", ROOT / "scripts" / "compare_experiment_0_multiseed.py")
cmp0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp0)

OFFICIAL_PROCESSED = ROOT / "datasets" / "processed"
WORK = ROOT / "models" / "checkpoints" / "exploratory_latent_sweep"
DATA_DIR = WORK / "data"
DROP_COLUMNS = (22, 23)
MAX_HORIZON = 10


def make_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with np.load(OFFICIAL_PROCESSED / f"{split}.npz") as data:
            payload = {k: data[k] for k in data.files}
        for key in ("states", "next_states"):
            assert np.all(payload[key][:, list(DROP_COLUMNS)] == 0), (split, key)
            payload[key] = np.delete(payload[key], list(DROP_COLUMNS), axis=1)
        np.savez(DATA_DIR / f"{split}.npz", **payload)
        # Same renaming as scripts/prepare_raw_sequence_dataset.py, for the raw-24 LSTM branch.
        np.savez(
            DATA_DIR / f"{split}_raw_seq.npz",
            z=payload["states"].astype(np.float32), next_z=payload["next_states"].astype(np.float32),
            actions=payload["actions"].astype(np.int64), rewards=payload["rewards"].astype(np.float32),
            episode_id=payload["episode_id"].astype(np.int64), time_step=payload["time_step"].astype(np.int64),
        )
        print(f"{split}: states {payload['states'].shape}, keys {sorted(payload)}")


def train_ae(latent: int, data_dir: Path, out: Path, seed: int = 0) -> None:
    """training/train_autoencoder.py main(), with hidden_dim = latent_dim and explicit paths."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path, validation_path = data_dir / "train.npz", data_dir / "validation.npz"

    train_dataset = TransitionDataset(train_path)
    input_dim = train_dataset.states.shape[1]
    config = RepresentationConfig(input_dim=input_dim, hidden_dim=latent, latent_dim=latent, seed=seed, checkpoint_dir=out)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader = tae.create_dataloader(train_path, config.batch_size, shuffle=True)
    validation_loader = tae.create_dataloader(validation_path, config.batch_size, shuffle=False)
    model = Autoencoder(input_dim=config.input_dim, hidden_dim=config.hidden_dim,
                        latent_dim=config.latent_dim, activation=config.activation).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    best_validation_loss, best_epoch = float("inf"), 0
    best_checkpoint, last_checkpoint = out / "autoencoder_best.pt", out / "autoencoder_last.pt"

    for epoch in range(config.epochs):
        train_loss = tae.train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss = tae.validate(model, validation_loader, criterion, device)
        print(f"Epoch {epoch + 1:03d}/{config.epochs} | Train: {train_loss:.6f} | Validation: {validation_loss:.6f}")
        tae.save_checkpoint(model, last_checkpoint, config=config)
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch = validation_loss, epoch + 1
            tae.save_checkpoint(model, best_checkpoint, config=config)

    # Encode the three splits with the best checkpoint (scripts/encode_latent_dataset.py logic).
    for split in ("train", "validation", "test"):
        with np.load(data_dir / f"{split}.npz") as data:
            fields = {k: data[k] for k in data.files}
        ae = load_autoencoder(checkpoint_path=best_checkpoint, input_dim=fields["states"].shape[1], device=device)
        ae.eval()
        with torch.no_grad():
            z = ae.encode(torch.from_numpy(fields["states"].astype(np.float32)).to(device)).cpu().numpy()
            next_z = ae.encode(torch.from_numpy(fields["next_states"].astype(np.float32)).to(device)).cpu().numpy()
        np.savez(out / f"{split}_latent.npz", z=z.astype(np.float32), next_z=next_z.astype(np.float32),
                 actions=fields["actions"].astype(np.int64), rewards=fields["rewards"].astype(np.float32),
                 episode_id=fields["episode_id"].astype(np.int64), time_step=fields["time_step"].astype(np.int64),
                 terminated=fields["terminated"].astype(bool), truncated=fields["truncated"].astype(bool))
    summary = {"latent_dim": latent, "hidden_dim": latent, "seed": seed, "input_dim": input_dim, "best_epoch": best_epoch,
               "best_validation_loss": best_validation_loss, "seconds": round(time.time() - t0, 1)}
    (out / "ae_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def train_lstm(train_path: Path, validation_path: Path, seed: int, out: Path, epochs: int = 300) -> None:
    """training/train_world_model_raw.py main() with explicit data paths (protocol imported)."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    twm._set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(train_path) as data:
        input_dim = data["z"].shape[1]
    config = WorldModelConfig(latent_dim=input_dim, action_dim=2, sequence_length=16)

    reward_scaler = twm.compute_reward_scaler(train_path)
    twm.save_reward_scaler(reward_scaler, out / "reward_scaler.json")
    reward_mean = torch.tensor(reward_scaler["reward_mean"], device=device)
    reward_std = torch.tensor(reward_scaler["reward_std"], device=device)

    train_dataset = LatentSequenceDataset(train_path, sequence_length=config.sequence_length, action_dim=config.action_dim)
    validation_dataset = LatentSequenceDataset(validation_path, sequence_length=config.sequence_length, action_dim=config.action_dim)
    train_loader = DataLoader(train_dataset, batch_size=twm.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=twm.BATCH_SIZE, shuffle=False)

    model = LatentDynamicsLSTM(latent_dim=config.latent_dim, action_dim=config.action_dim,
                               hidden_dim=config.hidden_dim, sequence_length=config.sequence_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=twm.LEARNING_RATE, weight_decay=twm.WEIGHT_DECAY)

    best_validation_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    stopped_early, epoch = False, 0
    for epoch in range(1, epochs + 1):
        train_loss = twm.train_one_epoch(model, train_loader, optimizer, device, reward_mean, reward_std)
        validation_loss = twm.validate(model, validation_loader, device, reward_mean, reward_std)
        twm.save_checkpoint(model, config, out / "world_model_last.pt", seed, epochs)
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch = validation_loss, epoch
            twm.save_checkpoint(model, config, out / "world_model_best.pt", seed, epochs)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={validation_loss:.6f}")
        if epochs_without_improvement >= twm.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping en epoca {epoch} (sin mejora en {twm.EARLY_STOPPING_PATIENCE} epocas).")
            stopped_early = True
            break
    summary = {"seed": seed, "input_dim": input_dim, "train": str(train_path), "last_epoch": epoch,
               "best_epoch": best_epoch, "best_validation_loss": best_validation_loss,
               "stopped_early": stopped_early, "seconds": round(time.time() - t0, 1)}
    (out / "lstm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def compare(z_dirs, raw_dirs, test_z: Path, test_raw: Path, output: Path) -> None:
    """Same report as scripts/compare_experiment_0_multiseed.py, with explicit test paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = []
    for z_dir, raw_dir in zip(z_dirs, raw_dirs):
        z_seed = json.loads((z_dir / "world_model_best.json").read_text(encoding="utf-8"))["seed"]
        raw_seed = json.loads((raw_dir / "world_model_best.json").read_text(encoding="utf-8"))["seed"]
        assert z_seed == raw_seed, (z_dir, raw_dir)
        z, z_episodes = cmp0.evaluate(z_dir / "world_model_best.pt", test_z, device)
        raw, raw_episodes = cmp0.evaluate(raw_dir / "world_model_best.pt", test_raw, device)
        pairs.append({"seed": z_seed, "z_dir": z_dir.as_posix(), "raw_dir": raw_dir.as_posix(),
                      "z": {str(h): v for h, v in z.items()}, "raw": {str(h): v for h, v in raw.items()},
                      f"z_h{MAX_HORIZON}_per_episode": z_episodes, f"raw_h{MAX_HORIZON}_per_episode": raw_episodes})
    horizons = [str(h) for h in range(1, MAX_HORIZON + 1)]
    summary, all_reductions = {}, []
    for h in horizons:
        z_mse = np.array([p["z"][h]["model_reward_mse"] for p in pairs])
        raw_mse = np.array([p["raw"][h]["model_reward_mse"] for p in pairs])
        reductions = (raw_mse - z_mse) / raw_mse
        all_reductions.extend(reductions.tolist())
        summary[h] = {"z_wins": int((z_mse < raw_mse).sum()), "pairs": len(pairs),
                      "median_reduction": float(np.median(reductions)), "reductions": reductions.tolist(),
                      "median_z_reward_mse": float(np.median(z_mse)), "median_raw_reward_mse": float(np.median(raw_mse))}
    per_seed = []
    for p in pairs:
        r = np.array([(p["raw"][h]["model_reward_mse"] - p["z"][h]["model_reward_mse"]) / p["raw"][h]["model_reward_mse"]
                      for h in horizons])
        per_seed.append({"seed": p["seed"], "z_wins_horizons": int((r > 0).sum()), "median_reduction": float(np.median(r))})
    episode_summary = []
    for p in pairs:
        ze, re_ = p[f"z_h{MAX_HORIZON}_per_episode"], p[f"raw_h{MAX_HORIZON}_per_episode"]
        episode_summary.append({"seed": p["seed"], "z_wins_episodes": int(sum(ze[e] < re_[e] for e in ze)),
                                "episodes": len(ze), "median_episode_z": float(np.median(list(ze.values()))),
                                "median_episode_raw": float(np.median(list(re_.values())))})
    report = {"pairs": pairs, "per_horizon": summary, "z_wins_total": sum(s["z_wins"] for s in summary.values()),
              "median_of_horizon_median_reductions": float(np.median([s["median_reduction"] for s in summary.values()])),
              "median_reduction_all_pairs": float(np.median(all_reductions)),
              f"h{MAX_HORIZON}_per_episode_summary": episode_summary, "per_seed": per_seed}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for s in per_seed:
        print(f"seed {s['seed']}: z gana {s['z_wins_horizons']}/10 horizontes, reduccion mediana {100 * s['median_reduction']:+.1f}%")
    print(f"z gana {report['z_wins_total']}/{len(pairs) * MAX_HORIZON} pares; mediana de medianas por horizonte "
          f"{100 * report['median_of_horizon_median_reductions']:+.1f}%; mediana de las medianas por semilla "
          f"{100 * float(np.median([s['median_reduction'] for s in per_seed])):+.1f}% -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-data")
    a = sub.add_parser("train-ae")
    a.add_argument("--latent", type=int, required=True)
    a.add_argument("--data-dir", type=Path, default=DATA_DIR)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--seed", type=int, default=0, help="Autoencoder seed (0 = the official protocol)")
    b = sub.add_parser("train-lstm")
    b.add_argument("--train", type=Path, required=True)
    b.add_argument("--val", type=Path, required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--z-dirs", type=Path, nargs="+", required=True)
    c.add_argument("--raw-dirs", type=Path, nargs="+", required=True)
    c.add_argument("--test-z", type=Path, required=True)
    c.add_argument("--test-raw", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "make-data":
        make_data()
    elif args.cmd == "train-ae":
        train_ae(args.latent, args.data_dir, args.out, args.seed)
    elif args.cmd == "train-lstm":
        train_lstm(args.train, args.val, args.seed, args.out)
    else:
        compare(args.z_dirs, args.raw_dirs, args.test_z, args.test_raw, args.output)


if __name__ == "__main__":
    main()
