"""Fidelity of the imagined 7-step return, before training any controller
(docs/v2/ADDENDUM_CONTROL.md, section 8.2). Descriptive: decides nothing.

For each of the 20 world models going to control (LSTM and Transformer, seeds 0-9) and every
window of every VALIDATION episode: seed CorridorDreamEnvironment -- the same class PPO trains in --
with the real 16-step window, step it 7 times with the actions actually recorded, and compare the
imagined return (clipped, as PPO sees it, and unclipped) with the real return of those 7 steps.

Per model: Pearson and bias (mean imagined - real), with 95% CIs from a bootstrap over episodes
(2,000 resamples, seed 0; windows of one episode overlap, so the episode is the unit). Per
architecture: the mean over its 10 models, bootstrapped with the same episode resamples.

    python scripts/v2/control_fidelity.py     # -> docs/results/v2/control/fidelity_validation.json
    python scripts/v2/control_fidelity.py --window-alignment aligned --output docs/results/v2/dream_alignment/fidelity_validation_aligned.json

--window-alignment picks CorridorDreamEnvironment's window alignment (default "legacy", the published
run; docs/v2/ADDENDUM_PLANIFICACION.md, section 11). An existing --output is never replaced.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_dream_environment import (  # noqa: E402
    SEQUENCE_DATA_DIR,
    CorridorDreamEnvironment,
    world_model_dir,
)

HORIZON = 7               # dream_max_steps (ADDENDUM_CONTROL.md, section 3)
ARCHITECTURES = ("lstm", "transformer")
SEEDS = tuple(range(10))
N_BOOT = 2000
VALIDATION_SEQ = SEQUENCE_DATA_DIR / "validation_seq.npz"
OUTPUT = ROOT_DIR / "docs" / "results" / "v2" / "control" / "fidelity_validation.json"


def imagined_vs_real(dream: CorridorDreamEnvironment) -> dict[int, np.ndarray]:
    """Per episode, an array (n_windows, 3): imagined clipped, imagined raw, real 7-step return."""
    out = {}
    L = dream.sequence_length
    for ep_id in dream._episode_ids:
        ep = dream._seed_episodes[ep_id]
        rows = []
        # The window ends at state index start+L-1; its 7 transitions use actions and rewards
        # start+L-1 .. start+L+5, so start <= T - L - HORIZON + 1.
        for start in range(len(ep["z"]) - L - HORIZON + 2):
            dream.reset(options={"episode_id": ep_id, "start": start})
            clipped = raw = 0.0
            for k in range(HORIZON):
                _, reward, _, _, info = dream.step(ep["actions"][start + L - 1 + k])
                clipped += reward
                raw += info["raw_predicted_reward"]
            real = float(ep["rewards"][start + L - 1: start + L - 1 + HORIZON].sum())
            rows.append((clipped, raw, real))
        out[ep_id] = np.array(rows)
    return out


def rollout_episode_returns(model, reward_scaler: dict, episode: dict, sequence_length: int,
                            horizon: int = HORIZON, clip: tuple[float, float] | None = None) -> np.ndarray:
    """Reference imagined returns from Experiment 1's own rollout_episode (actions aligned by time
    index), one per window start 0..T-L-horizon: the per-step predicted rewards are captured from
    the predict_next_step calls rollout_episode makes, clipped like the dream, and summed.
    ``clip`` defaults to the v2 dream's [REWARD_CLIP_MIN, REWARD_CLIP_MAX] (v1 passes its own)."""
    from evaluation import world_model_evaluation as wme
    from environments.corridor_dream_environment import REWARD_CLIP_MAX, REWARD_CLIP_MIN

    low, high = clip if clip is not None else (REWARD_CLIP_MIN, REWARD_CLIP_MAX)
    captured = []
    original = wme.predict_next_step

    def capture(*args, **kwargs):
        pred_z, pred_r = original(*args, **kwargs)
        captured.append(float(np.clip(pred_r.item(), low, high)))
        return pred_z, pred_r

    wme.predict_next_step = capture
    try:
        wme.rollout_episode(model, episode, sequence_length, model.action_dim,
                            horizon, torch.device("cpu"), reward_scaler["reward_mean"], reward_scaler["reward_std"])
    finally:
        wme.predict_next_step = original
    return np.array(captured).reshape(-1, horizon).sum(axis=1)


def _stats(rows: np.ndarray, col: int) -> tuple[float, float]:
    imagined, real = rows[:, col], rows[:, 2]
    return float(np.corrcoef(imagined, real)[0, 1]), float((imagined - real).mean())


def parse_args(default_output: Path, argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--window-alignment", choices=("legacy", "aligned"), default="legacy")
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists: choose another --output (published results are never replaced)")
    return args


def main() -> None:
    args = parse_args(OUTPUT)
    torch.set_num_threads(1)
    episodes = None
    per_model = {}
    t0 = time.perf_counter()
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            dream = CorridorDreamEnvironment(world_model_dir(arch, seed), seed_episodes_path=VALIDATION_SEQ,
                                             max_dream_steps=HORIZON, window_alignment=args.window_alignment)
            data = imagined_vs_real(dream)
            episodes = episodes or sorted(data)
            per_model[(arch, seed)] = data
            print(f"{arch} s{seed}: {sum(len(v) for v in data.values())} windows ({time.perf_counter() - t0:.0f} s)",
                  flush=True)

    rng = np.random.default_rng(0)
    resamples = [rng.integers(0, len(episodes), len(episodes)) for _ in range(N_BOOT)]
    result = {"split": "validation", "window_alignment": args.window_alignment, "episodes": episodes, "horizon": HORIZON, "n_boot": N_BOOT,
              "windows_per_model": int(sum(len(v) for v in per_model[("lstm", 0)].values())),
              "models": {}, "architectures": {}}
    boot = {}
    for (arch, seed), data in per_model.items():
        pooled = np.concatenate([data[e] for e in episodes])
        entry = {"real_return_mean": float(pooled[:, 2].mean()),
                 "clipped_fraction_of_returns_changed": float((pooled[:, 0] != pooled[:, 1]).mean())}
        for col, label in ((0, "clipped"), (1, "raw")):
            pearson, bias = _stats(pooled, col)
            samples = np.array([_stats(np.concatenate([data[episodes[i]] for i in idx]), col) for idx in resamples])
            boot[(arch, seed, label)] = samples
            entry[label] = {"pearson": pearson, "bias": bias,
                            "pearson_ci95": np.percentile(samples[:, 0], [2.5, 97.5]).tolist(),
                            "bias_ci95": np.percentile(samples[:, 1], [2.5, 97.5]).tolist()}
        result["models"][f"{arch}_s{seed}"] = entry
    for arch in ARCHITECTURES:
        result["architectures"][arch] = {}
        for label in ("clipped", "raw"):
            point = np.array([[result["models"][f"{arch}_s{s}"][label][k] for k in ("pearson", "bias")] for s in SEEDS])
            mean_boot = np.mean([boot[(arch, s, label)] for s in SEEDS], axis=0)  # same resamples for every model
            result["architectures"][arch][label] = {
                "pearson_mean": float(point[:, 0].mean()), "pearson_min": float(point[:, 0].min()),
                "pearson_max": float(point[:, 0].max()),
                "pearson_mean_ci95": np.percentile(mean_boot[:, 0], [2.5, 97.5]).tolist(),
                "bias_mean": float(point[:, 1].mean()), "bias_min": float(point[:, 1].min()),
                "bias_max": float(point[:, 1].max()),
                "bias_mean_ci95": np.percentile(mean_boot[:, 1], [2.5, 97.5]).tolist()}
    result["seconds"] = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")

    for arch in ARCHITECTURES:
        for label in ("clipped", "raw"):
            a = result["architectures"][arch][label]
            print(f"{arch:11s} {label:7s}: Pearson {a['pearson_mean']:.3f} [{a['pearson_mean_ci95'][0]:.3f}, "
                  f"{a['pearson_mean_ci95'][1]:.3f}] (modelos {a['pearson_min']:.3f}..{a['pearson_max']:.3f}); "
                  f"sesgo {a['bias_mean']:+.1f} [{a['bias_mean_ci95'][0]:+.1f}, {a['bias_mean_ci95'][1]:+.1f}] "
                  f"(modelos {a['bias_min']:+.1f}..{a['bias_max']:+.1f})")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
