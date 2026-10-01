"""Train one Dream-Environment PPO seed with the OFFICIAL training code, into a new folder.

training/train_controller.py takes its seed from ControllerConfig and writes into the official
models/checkpoints/controller/. This wrapper changes only those two things, from outside and
without editing the official file: it points the module's CONTROLLER_DIR at --output-dir and makes
its ControllerConfig() return ControllerConfig(seed=--seed). Everything else (Dream Environment,
official LSTM, PPO hyperparameters, real-SUMO checkpoint selection) is the official main().

Refuses to write into the official folder or into a folder that already holds files.

    python docs/results/ppo_10_seeds/train_dream_seed.py --seed 3 --output-dir models/checkpoints/controller_10seeds/seed3
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import training.train_controller as tc  # noqa: E402
from configs import ControllerConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if out == tc.CONTROLLER_DIR.resolve():
        raise SystemExit(f"{out} is the official controller folder: refusing to write there.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already holds files: choose an empty folder.")
    out.mkdir(parents=True, exist_ok=True)

    tc.CONTROLLER_DIR = out
    official_config = tc.ControllerConfig
    tc.ControllerConfig = lambda: replace(official_config(), seed=args.seed)
    print(f"Dream PPO, official training code -- seed {args.seed} -> {out}")
    tc.main()


if __name__ == "__main__":
    assert ControllerConfig().seed == 2  # the official default, unchanged
    main()
