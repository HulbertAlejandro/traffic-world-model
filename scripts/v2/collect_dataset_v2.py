"""Collect the v2 corridor dataset as pre-registered in docs/v2/ADDENDUM_DATASET.md.

One ``episode_<seed>.npz`` per episode under ``datasets/v2/raw/<split>/``, with v1's
eight keys (states, actions, rewards, next_states, terminated, truncated,
episode_id, time_step), so scripts/merge_dataset.merge_files and
scripts/normalize_dataset.normalize_dataset work unchanged. Differences in shape
only: states/next_states are (60, 104) and actions (60, 4) keep/switch int64;
episode_id is the episode's seed (unique across splits).

The .npz files are not committed (CLAUDE.md). What is versioned is the manifest,
docs/results/v2/dataset/<manifest>.json: seed, split, policy, pulse offset,
return, measured time and SHA-256 of every file, so the dataset can be checked
and regenerated (SUMO, the random policy and the pulse offset are all seeded).

    python scripts/v2/collect_dataset_v2.py --pilot        # pilot sample, seeds 19000-19017
    python scripts/v2/collect_dataset_v2.py                # the pre-registered dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_environment import CorridorTrafficEnvironment  # noqa: E402
from environments.four_intersections import DELTA_TIME  # noqa: E402
from scripts.v2.corridor_policies import COLLECTION_POLICIES, make_collection_policy  # noqa: E402

# Pre-registered (docs/v2/ADDENDUM_DATASET.md). Disjoint from each other, from the
# Phase 0 calibration seeds (11000-11019) and from the pilot seeds.
SPLITS = {
    "train": range(20000, 20112),        # 112
    "validation": range(21000, 21024),   # 24
    "test": range(22000, 22024),         # 24
    "ood": range(23000, 23030),          # 30, reserved: never used for any design decision
}
PILOT_SEEDS = range(19000, 19018)        # 6 per policy, discarded after inspection
CALIBRATION_SEEDS = range(11000, 11020)

RAW_DIR = ROOT_DIR / "datasets" / "v2" / "raw"
MANIFEST_DIR = ROOT_DIR / "docs" / "results" / "v2" / "dataset"


def policy_for(seeds: range, seed: int) -> str:
    """Round robin by position inside the split: every split gets the same policy mix."""
    return COLLECTION_POLICIES[(seed - seeds.start) % len(COLLECTION_POLICIES)]


def collect_episode(env: CorridorTrafficEnvironment, seed: int, policy_name: str) -> tuple[dict, dict]:
    policy = make_collection_policy(policy_name, seed)
    t0 = time.perf_counter()
    state, info = env.reset(seed=seed)
    rows = {k: [] for k in ("states", "actions", "rewards", "next_states", "terminated", "truncated")}
    switches = np.zeros(4)
    arrivals = 0.0
    max_pending = 0.0
    max_pending_after_warmup = 0.0  # after the first 30 s, as in the Phase 0 criterion
    step, done = 0, False
    while not done:
        action = policy(env, step)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        rows["states"].append(state)
        rows["actions"].append(action)
        rows["rewards"].append(reward)
        rows["next_states"].append(next_state)
        rows["terminated"].append(terminated)
        rows["truncated"].append(truncated)
        switches += step_info["phase_switched"]
        arrivals += step_info["arrivals_total"]
        max_pending = max(max_pending, step_info["pending_vehicles"])
        if (step + 1) * DELTA_TIME > 30:
            max_pending_after_warmup = max(max_pending_after_warmup, step_info["pending_vehicles"])
        state, step, done = next_state, step + 1, terminated or truncated
    n = len(rows["rewards"])
    data = {
        "states": np.asarray(rows["states"], dtype=np.float32),
        "actions": np.asarray(rows["actions"], dtype=np.int64),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "next_states": np.asarray(rows["next_states"], dtype=np.float32),
        "terminated": np.asarray(rows["terminated"], dtype=bool),
        "truncated": np.asarray(rows["truncated"], dtype=bool),
        "episode_id": np.full(n, seed, dtype=np.int64),
        "time_step": np.arange(n, dtype=np.int64),
    }
    meta = {
        "seed": seed, "policy": policy_name, "pulse_offset": info["pulse_offset"], "transitions": n,
        "return": float(data["rewards"].sum()), "switches_per_signal": switches.tolist(),
        "arrivals": arrivals, "max_pending": max_pending, "max_pending_after_warmup": max_pending_after_warmup,
        "seconds": time.perf_counter() - t0,
    }
    return data, meta


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(plan: list[tuple[str, int, str]], out_dir: Path, manifest_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    t_start = time.perf_counter()
    t_env = time.perf_counter()
    env = CorridorTrafficEnvironment()
    env_creation = time.perf_counter() - t_env
    try:
        for i, (split, seed, policy_name) in enumerate(plan):
            data, meta = collect_episode(env, seed, policy_name)
            path = out_dir / split / f"episode_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **data)
            meta |= {"split": split, "file": path.relative_to(ROOT_DIR).as_posix(), "sha256": _sha256(path)}
            episodes.append(meta)
            print(f"[{i + 1:3d}/{len(plan)}] {split:10s} seed {seed} {policy_name:15s} offset {meta['pulse_offset']:3d} "
                  f"return {meta['return']:9.1f} {meta['seconds']:.2f} s", flush=True)
    finally:
        env.close()
    total = time.perf_counter() - t_start
    manifest = {
        "pre_registration": "docs/v2/ADDENDUM_DATASET.md",
        "splits": {k: [v.start, v.stop - 1] for k, v in SPLITS.items()},
        "environment_creation_seconds": env_creation,
        "total_seconds": total,
        "mean_episode_seconds": float(np.mean([e["seconds"] for e in episodes])),
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(episodes)} episodes in {total:.0f} s "
          f"(mean {manifest['mean_episode_seconds']:.2f} s per episode) -> {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true", help="pilot sample only (seeds 19000-19017)")
    args = parser.parse_args()

    all_seeds = [s for r in (*SPLITS.values(), PILOT_SEEDS, CALIBRATION_SEEDS) for s in r]
    assert len(all_seeds) == len(set(all_seeds)), "seed ranges overlap"

    if args.pilot:
        plan = [("pilot", s, policy_for(PILOT_SEEDS, s)) for s in PILOT_SEEDS]
        collect(plan, ROOT_DIR / "datasets" / "v2", MANIFEST_DIR / "pilot_manifest.json")
    else:
        plan = [(split, s, policy_for(seeds, s)) for split, seeds in SPLITS.items() for s in seeds]
        collect(plan, RAW_DIR, MANIFEST_DIR / "manifest.json")


if __name__ == "__main__":
    main()
