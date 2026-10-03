"""v2 Phase 2: does compressing the 104-dim corridor state with an Autoencoder help the LSTM?

Pre-registered in docs/v2/ADDENDUM_AUTOENCODER.md. Two branches, ONE training path:

- branch "z":   Autoencoder (seed a, latent size k) -> z -> LSTM (seed s)
- branch "raw": normalized 104-dim state            -> LSTM (seed s)

Both branches train the LSTM through the same function, ``train_lstm``, with the same
``LSTM_PROTOCOL`` object; the only thing that differs is the input dimension (k or 104).
tests/test_v2_compression_protocol.py checks this. The per-epoch loop, the validation
loss, the reward scaler (train split only) and the seeding are v1's own functions from
training/train_world_model.py, imported, not copied (the v1 raw branch once lost weight
decay and early stopping because its loop was a copy).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.latent_sequence_dataset import LatentSequenceDataset
from datasets.transition_dataset import TransitionDataset
from evaluation.world_model_evaluation import load_episodes, rollout_episode
from models.representation import Autoencoder
from models.world_model import LatentDynamicsLSTM
from training import train_autoencoder as v1_ae
from training import train_world_model as v1_lstm

STATE_DIM = 104
ACTION_DIM = 8          # per-signal one-hot of the 4 keep/switch decisions (encode_actions)
MAX_HORIZON = 10


@dataclass(frozen=True)
class LSTMProtocol:
    """Everything about LSTM training except the input dimension. Shared by both branches."""
    optimizer: str = "adam"
    learning_rate: float = v1_lstm.LEARNING_RATE           # 1e-3
    weight_decay: float = v1_lstm.WEIGHT_DECAY             # 1e-4
    batch_size: int = v1_lstm.BATCH_SIZE                   # 32
    max_epochs: int = 300                                  # v1 Experiment 0 redo: cap that early stopping never reaches
    early_stopping_patience: int = v1_lstm.EARLY_STOPPING_PATIENCE  # 15
    reward_loss_weight: float = v1_lstm.REWARD_LOSS_WEIGHT  # 1.0 (applied inside v1's train_one_epoch/validate)
    sequence_length: int = 16
    hidden_dim: int = 128
    action_dim: int = ACTION_DIM


@dataclass(frozen=True)
class AutoencoderProtocol:
    hidden_dim: int = STATE_DIM        # as wide as the input: the latent layer is the only bottleneck
    activation: str = "relu"
    learning_rate: float = 1e-3        # v1 RepresentationConfig defaults
    batch_size: int = 32
    epochs: int = 100                  # v1: fixed epochs, keep the best validation epoch


LSTM_PROTOCOL = LSTMProtocol()
AE_PROTOCOL = AutoencoderProtocol()


# --------------------------------------------------------------------------- Autoencoder
def train_autoencoder(train_npz: Path, val_npz: Path, latent_dim: int, seed: int, out_dir: Path,
                      protocol: AutoencoderProtocol = AE_PROTOCOL) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_lstm._set_seeds(seed)
    model = Autoencoder(input_dim=STATE_DIM, hidden_dim=protocol.hidden_dim, latent_dim=latent_dim,
                        activation=protocol.activation)
    train_loader = DataLoader(TransitionDataset(train_npz), batch_size=protocol.batch_size, shuffle=True)
    val_loader = DataLoader(TransitionDataset(val_npz), batch_size=protocol.batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=protocol.learning_rate)
    criterion, device = nn.MSELoss(), torch.device("cpu")
    best, best_epoch = float("inf"), 0
    for epoch in range(1, protocol.epochs + 1):
        v1_ae.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = v1_ae.validate(model, val_loader, criterion, device)
        if val < best:
            best, best_epoch = val, epoch
            torch.save(model.state_dict(), out_dir / "autoencoder_best.pt")
    meta = {"input_dim": STATE_DIM, "latent_dim": latent_dim, "seed": seed, "best_epoch": best_epoch,
            "best_validation_mse": best, "protocol": asdict(protocol)}
    (out_dir / "autoencoder_best.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_autoencoder(ae_dir: Path) -> Autoencoder:
    meta = json.loads((ae_dir / "autoencoder_best.json").read_text(encoding="utf-8"))
    p = meta["protocol"]
    model = Autoencoder(input_dim=meta["input_dim"], hidden_dim=p["hidden_dim"], latent_dim=meta["latent_dim"],
                        activation=p["activation"])
    model.load_state_dict(torch.load(ae_dir / "autoencoder_best.pt", map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def encode_split(ae: Autoencoder | None, split_npz: Path, out_npz: Path) -> None:
    """Sequence-ready split: z/next_z from the frozen encoder, or the normalized state itself (ae=None)."""
    with np.load(split_npz) as d:
        data = {k: d[k] for k in d.files}
    if ae is None:
        z, next_z = data["states"], data["next_states"]
    else:
        z = ae.encoder(torch.from_numpy(data["states"])).numpy()
        next_z = ae.encoder(torch.from_numpy(data["next_states"])).numpy()
    np.savez(out_npz, z=z.astype(np.float32), next_z=next_z.astype(np.float32), actions=data["actions"],
             rewards=data["rewards"], episode_id=data["episode_id"], time_step=data["time_step"])


@torch.no_grad()
def reconstruction_report(ae: Autoencoder, val_npz: Path) -> dict:
    """Per-column validation reconstruction MSE (informative; includes the 8 constant columns)."""
    with np.load(val_npz) as d:
        x = torch.from_numpy(d["states"])
    _, recon = ae(x)
    per_col = ((recon - x) ** 2).mean(0).numpy()
    return {"mse": float(per_col.mean()), "per_column": per_col.round(6).tolist()}


# --------------------------------------------------------------------------- LSTM (both branches)
def train_lstm(train_seq: Path, val_seq: Path, input_dim: int, seed: int, out_dir: Path,
               protocol: LSTMProtocol = LSTM_PROTOCOL) -> dict:
    """The single LSTM training path of both branches. ``input_dim`` is k (branch z) or 104 (raw)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_lstm._set_seeds(seed)
    device = torch.device("cpu")
    scaler = v1_lstm.compute_reward_scaler(train_seq)  # train split only
    (out_dir / "reward_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    reward_mean, reward_std = torch.tensor(scaler["reward_mean"]), torch.tensor(scaler["reward_std"])
    train_loader = DataLoader(LatentSequenceDataset(train_seq, protocol.sequence_length, protocol.action_dim),
                              batch_size=protocol.batch_size, shuffle=True)
    val_loader = DataLoader(LatentSequenceDataset(val_seq, protocol.sequence_length, protocol.action_dim),
                            batch_size=protocol.batch_size, shuffle=False)
    model = LatentDynamicsLSTM(latent_dim=input_dim, action_dim=protocol.action_dim, hidden_dim=protocol.hidden_dim,
                               sequence_length=protocol.sequence_length)
    if protocol.optimizer != "adam":
        raise ValueError(f"unsupported optimizer {protocol.optimizer!r}")
    optimizer = optim.Adam(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    hparams = {"architecture": "lstm", "latent_dim": input_dim, "action_dim": protocol.action_dim,
               "sequence_length": protocol.sequence_length, "hidden_dim": protocol.hidden_dim, "seed": seed,
               "protocol": asdict(protocol)}
    best, best_epoch, since, history = float("inf"), 0, 0, []
    for epoch in range(1, protocol.max_epochs + 1):
        train_loss = v1_lstm.train_one_epoch(model, train_loader, optimizer, device, reward_mean, reward_std)
        val_loss = v1_lstm.validate(model, val_loader, device, reward_mean, reward_std)
        history.append((train_loss, val_loss))
        if val_loss < best:
            best, best_epoch, since = val_loss, epoch, 0
            torch.save(model.state_dict(), out_dir / "world_model_best.pt")
        else:
            since += 1
        if since >= protocol.early_stopping_patience:
            break
    hparams |= {"best_epoch": best_epoch, "stopped_epoch": epoch, "best_validation_loss": best,
                "hit_max_epochs": since < protocol.early_stopping_patience}
    (out_dir / "world_model_best.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return hparams


def train_branch(branch: str, train_seq: Path, val_seq: Path, seed: int, out_dir: Path,
                 latent_dim: int | None = None, protocol: LSTMProtocol = LSTM_PROTOCOL) -> dict:
    """Entry point of each branch. The ONLY difference between branches is input_dim."""
    if branch == "z":
        if latent_dim is None:
            raise ValueError("branch 'z' needs latent_dim")
        input_dim = latent_dim
    elif branch == "raw":
        if latent_dim is not None:
            raise ValueError("branch 'raw' takes no latent_dim")
        input_dim = STATE_DIM
    else:
        raise ValueError(f"unknown branch {branch!r}")
    return train_lstm(train_seq, val_seq, input_dim, seed, out_dir, protocol)


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model_dir: Path, test_seq: Path) -> dict:
    """reward_mse per horizon 1..10 on the test split, autoregressive rollouts with the true
    actions (v1's rollout_episode), reward de-normalized with this run's scaler. Also the
    per-episode reward MSE at every horizon, and the persistence baseline."""
    hp = json.loads((model_dir / "world_model_best.json").read_text(encoding="utf-8"))
    scaler = json.loads((model_dir / "reward_scaler.json").read_text(encoding="utf-8"))
    model = LatentDynamicsLSTM(latent_dim=hp["latent_dim"], action_dim=hp["action_dim"], hidden_dim=hp["hidden_dim"],
                               sequence_length=hp["sequence_length"])
    model.load_state_dict(torch.load(model_dir / "world_model_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    per_episode, pooled = {}, {h: [] for h in range(1, MAX_HORIZON + 1)}
    baseline = {h: [] for h in range(1, MAX_HORIZON + 1)}
    for ep_id, ep in load_episodes(test_seq).items():
        records = rollout_episode(model, ep, hp["sequence_length"], hp["action_dim"], MAX_HORIZON,
                                  torch.device("cpu"), scaler["reward_mean"], scaler["reward_std"])
        per_h = {h: [] for h in range(1, MAX_HORIZON + 1)}
        for r in records:
            per_h[r["horizon"]].append(r["model_reward_se"])
            pooled[r["horizon"]].append(r["model_reward_se"])
            baseline[r["horizon"]].append(r["baseline_reward_se"])
        per_episode[int(ep_id)] = {h: float(np.mean(v)) for h, v in per_h.items()}
    reward_mse = {h: float(np.mean(v)) for h, v in pooled.items()}
    return {"reward_mse": reward_mse,
            "gm_reward_mse": float(np.exp(np.mean(np.log(list(reward_mse.values()))))),
            "baseline_reward_mse": {h: float(np.mean(v)) for h, v in baseline.items()},
            "per_episode_reward_mse": per_episode}
