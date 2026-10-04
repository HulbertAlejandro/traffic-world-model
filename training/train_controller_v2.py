"""Train one PPO controller of the v2 control phase (docs/v2/ADDENDUM_CONTROL.md).

One script for the three arms, so they cannot drift apart:

- ``dream_lstm`` / ``dream_transformer``: PPO trained in CorridorDreamEnvironment, imagining with
  the world model of the SAME seed and architecture (models/checkpoints/v2/arch_comparison/
  <arch>_raw_s<seed>/). Zero real steps to train; checkpoint selection runs on real SUMO (v1's
  lesson: imagined reward did not predict real reward, Pearson 0.08), through
  ScaledCorridorEnvironment, which normalizes with the dataset's scaler.pkl like the dream.
- ``direct``: PPO trained directly on real SUMO (raw 104-dim state, VecNormalize as in v1), with
  training scenarios 30000, 30001, ... (disjoint from every split).

The PPO configuration of every arm comes from ``ppo_config``, which starts from the same
``ControllerConfig()`` and changes only the fields the addendum lists (section 3). Periodic
evaluation uses validation seeds 21000-21004 (never the train seeds 20000-20111).
Every real SUMO step (training and periodic evaluation) is counted by a wrapper and written to
run_info.json, with the wall-clock times; nothing is deduced.

    python training/train_controller_v2.py --arm dream_lstm --seed 0 --output-dir models/checkpoints/v2/control/dream_lstm_s0
    python training/train_controller_v2.py --arm direct --seed 0 --total-timesteps 10000 --output-dir <folder>

Refuses to write into a folder that already holds any file it would produce (--overwrite), and into
v1's official/archived controller folders or anywhere under the v2 world models
(--overwrite-official). Reuses v1's training helpers, imported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs import ControllerConfig  # noqa: E402
from environments.reseeding_wrapper import ReseedingWrapper  # noqa: E402
from training.output_guard import check_output_dir  # noqa: E402
from training.train_controller import (  # noqa: E402
    SaveVecNormalizeOnBest,
    _set_seeds,
    build_normalized_envs,
    save_hyperparameters,
    vecnormalize_path,
)

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
CONTROL_DIR = CHECKPOINT_DIR / "v2" / "control"
DREAM_ARMS = {"dream_lstm": "lstm", "dream_transformer": "transformer"}
ARMS = (*DREAM_ARMS, "direct")

# docs/v2/ADDENDUM_CONTROL.md, section 2.
TRAIN_SPLIT_SEEDS = range(20000, 20112)                  # dataset only: never evaluated on
SELECTION_SEEDS = (21000, 21001, 21002, 21003, 21004)    # validation, periodic evaluation
DIRECT_TRAINING_SEED_START = 30000                       # direct RL scenarios: 30000, 30001, ...
DREAM_EVAL_FREQ = 5000                                   # v1's value: 10 evaluations in 50,000 steps
N_EVAL_EPISODES = 5

# v1's official / archived controller folders, and the v2 world models: never written without
# --overwrite-official. PROTECTED_ROOTS also covers every folder inside them.
PROTECTED_DIRS = tuple(CHECKPOINT_DIR / name for name in (
    "controller", "controller_direct", "controller_direct_30k", "controller_direct_prefix_bug",
    "controller_direct_30k_prefix_bug", "controller_10seeds", "controller_direct_10seeds",
    "controller_direct_30k_10seeds"))
PROTECTED_ROOTS = (CHECKPOINT_DIR / "v2" / "arch_comparison", CHECKPOINT_DIR / "v2" / "compression")
OUTPUT_FILES = (
    "best_model.zip", "best_model.json", "best_model_vecnormalize.pkl", "evaluations.npz",
    "ppo_controller_final.zip", "ppo_controller_final.json", "ppo_controller_final_vecnormalize.pkl",
    "run_info.json",
)


def ppo_config(arm: str, seed: int, total_timesteps: int | None = None) -> ControllerConfig:
    """The PPO configuration of an arm: ControllerConfig() with only the addendum's per-arm fields."""
    base = ControllerConfig()
    if arm in DREAM_ARMS:
        return replace(base, seed=seed, total_timesteps=total_timesteps or base.total_timesteps)
    if arm == "direct":
        if total_timesteps is None:
            raise ValueError("the direct arm needs an explicit --total-timesteps (10000 or 30000)")
        # normalize_obs: the direct policy sees the raw state (it must not use the dataset's scaler).
        return replace(base, seed=seed, total_timesteps=total_timesteps, normalize_obs=True)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def ppo_kwargs(config: ControllerConfig) -> dict:
    """The PPO constructor arguments, the same for every arm (v1's set)."""
    return {"learning_rate": config.learning_rate, "n_steps": config.n_steps, "batch_size": config.batch_size,
            "n_epochs": config.n_epochs, "gamma": config.gamma, "seed": config.seed}


def eval_freq(arm: str, config: ControllerConfig) -> int:
    return DREAM_EVAL_FREQ if arm in DREAM_ARMS else max(config.n_steps, 1000)


class RealStepCounter(gym.Wrapper):
    """Counts the real SUMO steps and episodes taken through it, and the time spent in them."""

    def __init__(self, env: gym.Env, counter: dict) -> None:
        super().__init__(env)
        self.counter = counter
        counter.update(steps=0, episodes=0, seconds=0.0)

    def reset(self, **kwargs):
        t0 = time.perf_counter()
        result = self.env.reset(**kwargs)
        self.counter["episodes"] += 1
        self.counter["seconds"] += time.perf_counter() - t0
        return result

    def step(self, action):
        t0 = time.perf_counter()
        result = self.env.step(action)
        self.counter["steps"] += 1
        self.counter["seconds"] += time.perf_counter() - t0
        return result


def check_v2_output_dir(output_dir: Path, overwrite: bool, overwrite_official: bool) -> Path:
    """v1's guard (output_guard.check_output_dir) plus: nothing inside the v2 world-model folders."""
    resolved = Path(output_dir).resolve()
    inside = [root for root in PROTECTED_ROOTS if resolved.is_relative_to(root.resolve())]
    if inside and not overwrite_official:
        raise SystemExit(f"{resolved} is inside {inside[0]}, which holds the v2 world models. "
                         "Choose another --output-dir, or pass --overwrite-official if explicitly approved.")
    return check_output_dir(resolved, PROTECTED_DIRS, OUTPUT_FILES, overwrite, overwrite_official)


def make_envs(arm: str, seed: int, config: ControllerConfig, train_counter: dict, eval_counter: dict):
    """(make_train_env, make_eval_env) for build_normalized_envs."""
    from environments.corridor_environment import CorridorTrafficEnvironment
    from environments.scaled_corridor_environment import ScaledCorridorEnvironment

    assert not set(SELECTION_SEEDS) & set(TRAIN_SPLIT_SEEDS)
    eval_seeds = lambda: ReseedingWrapper.fixed_eval_seeds(list(SELECTION_SEEDS))  # noqa: E731
    if arm in DREAM_ARMS:
        from environments.corridor_dream_environment import CorridorDreamEnvironment, world_model_dir

        model_dir = world_model_dir(DREAM_ARMS[arm], seed)
        return (lambda: CorridorDreamEnvironment(model_dir, max_dream_steps=config.dream_max_steps),
                lambda: RealStepCounter(ReseedingWrapper(ScaledCorridorEnvironment(), eval_seeds()), eval_counter))
    return (lambda: RealStepCounter(ReseedingWrapper(CorridorTrafficEnvironment(),
                                                     ReseedingWrapper.training_seeds(DIRECT_TRAINING_SEED_START)),
                                    train_counter),
            lambda: RealStepCounter(ReseedingWrapper(CorridorTrafficEnvironment(), eval_seeds()), eval_counter))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True, help="controller seed (= world model seed in the dream arms)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="imagined steps (dream, default ControllerConfig's 50,000) or real steps (direct, required)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing files already in --output-dir")
    parser.add_argument("--overwrite-official", action="store_true",
                        help="allow writing into v1's official controller folders or the v2 world-model folders")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = check_v2_output_dir(args.output_dir, args.overwrite, args.overwrite_official)
    config = ppo_config(args.arm, args.seed, args.total_timesteps)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds(config.seed)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    train_counter, eval_counter = {}, {}
    train_env, eval_env = build_normalized_envs(*make_envs(args.arm, args.seed, config, train_counter, eval_counter),
                                                config)
    model = PPO("MlpPolicy", train_env, verbose=0, **ppo_kwargs(config))
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(output_dir), log_path=str(output_dir),
        eval_freq=eval_freq(args.arm, config), n_eval_episodes=N_EVAL_EPISODES, deterministic=True,
        callback_on_new_best=SaveVecNormalizeOnBest(output_dir / "best_model.zip") if config.normalize_obs else None,
    )
    print(f"{args.arm} seed {config.seed}: {config.total_timesteps} steps -> {output_dir}", flush=True)
    t0 = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)
    wall = time.perf_counter() - t0

    final_path = output_dir / "ppo_controller_final.zip"
    model.save(final_path)
    save_hyperparameters(final_path, config)
    save_hyperparameters(output_dir / "best_model.zip", config)
    train_env.save(str(vecnormalize_path(final_path)))
    train_env.close()
    eval_env.close()

    dream = args.arm in DREAM_ARMS
    real_train = 0 if dream else train_counter["steps"]
    run_info = {
        "arm": args.arm, "seed": config.seed, "config": asdict(config),
        "world_model_dir": (Path("models/checkpoints/v2/arch_comparison") / f"{DREAM_ARMS[args.arm]}_raw_s{args.seed}").as_posix()
        if dream else None,
        "observation": "scaler.pkl-normalized state (ScaledCorridorEnvironment)" if dream else "raw state + VecNormalize",
        "selection_seeds": list(SELECTION_SEEDS),
        "direct_training_seed_start": None if dream else DIRECT_TRAINING_SEED_START,
        "timesteps_done": int(model.num_timesteps),
        "imagined_steps": int(model.num_timesteps) if dream else 0,
        "real_steps_train": real_train,
        "real_episodes_train": 0 if dream else train_counter["episodes"],
        "real_steps_eval": eval_counter["steps"],
        "real_episodes_eval": eval_counter["episodes"],
        "real_steps_total": real_train + eval_counter["steps"],
        "wall_seconds": wall,
        "eval_sumo_seconds": eval_counter["seconds"],
        "train_sumo_seconds": None if dream else train_counter["seconds"],
        # Wall time outside the periodic evaluation, per training step (imagined or real), PPO updates included.
        "seconds_per_training_step": (wall - eval_counter["seconds"]) / max(model.num_timesteps, 1),
        "best_mean_reward": float(eval_callback.best_mean_reward),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"done in {wall:.0f} s; real steps {run_info['real_steps_total']}; best real eval {run_info['best_mean_reward']:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
