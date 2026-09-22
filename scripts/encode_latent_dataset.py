"""Encode a processed dataset split into latent space using the frozen encoder.

This is the bridge between the Autoencoder and the temporal model: it never
trains anything, it only runs the already-trained encoder over states and
next_states, producing a new .npz with z instead of raw state vectors, while
preserving every field the temporal-model dataset needs (episode_id,
time_step, actions, rewards, terminated, truncated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.autoencoder_evaluation import load_autoencoder

PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
CHECKPOINT_PATH = ROOT_DIR / "models" / "checkpoints" / "autoencoder_best.pt"


def encode_split(split_path: Path, encoder_checkpoint: Path, device: torch.device) -> Path:
    """Encode one split file (train.npz/validation.npz/test.npz) to latent space."""
    with np.load(split_path) as data:
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]
        rewards = data["rewards"]
        episode_id = data["episode_id"]
        time_step = data["time_step"]
        terminated = data["terminated"]
        truncated = data["truncated"]

    input_dim = states.shape[1]
    autoencoder = load_autoencoder(
        checkpoint_path=encoder_checkpoint,
        input_dim=input_dim,
        device=device,
    )
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        states_tensor = torch.from_numpy(states.astype(np.float32)).to(device)
        next_states_tensor = torch.from_numpy(next_states.astype(np.float32)).to(device)
        z = autoencoder.encode(states_tensor).cpu().numpy()
        next_z = autoencoder.encode(next_states_tensor).cpu().numpy()

    output_path = split_path.with_name(split_path.stem + "_latent.npz")
    np.savez(
        output_path,
        z=z.astype(np.float32),
        next_z=next_z.astype(np.float32),
        actions=actions.astype(np.int64),
        rewards=rewards.astype(np.float32),
        episode_id=episode_id.astype(np.int64),
        time_step=time_step.astype(np.int64),
        terminated=terminated.astype(bool),
        truncated=truncated.astype(bool),
    )
    return output_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for split_name in ("train", "validation", "test"):
        split_path = PROCESSED_DIR / f"{split_name}.npz"
        if not split_path.exists():
            print(f"Skipping {split_name}: {split_path} not found.")
            continue
        output_path = encode_split(split_path, CHECKPOINT_PATH, device)
        print(f"Encoded {split_name} -> {output_path}")
