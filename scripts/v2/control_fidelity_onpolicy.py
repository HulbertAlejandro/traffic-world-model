"""On-policy fidelity of the dream, a DIAGNOSTIC (docs/v2/ADDENDUM_CONTROL.md, section 12.1). Trains nothing.

The stage-1 fidelity (scripts/v2/control_fidelity.py) used the dataset's actions. Here each of the
20 dream controllers drives real SUMO on validation seeds 21000-21005; its states, actions and
rewards are recorded, and for every window the world model of the SAME seed and architecture is
seeded with the real 16-step window and stepped with the 7 actions the controller actually took
(the same CorridorDreamEnvironment PPO trained in, via control_fidelity.imagined_vs_real). The
imagined 7-step return (clipped as PPO saw it, and unclipped) is compared with the real one.

Also records, per window, how long each signal had kept its green phase at the rollout's first
decision: floor(elapsed_phase_time / 5 s) of the real state (column 24 of each signal's 26-dim
block), in control steps. Bias is reported by the longest hold over the four signals (0-5, 6-10,
11-19, >= 20 steps) and, for >= 20, by which signal holds.

    python scripts/v2/control_fidelity_onpolicy.py   # -> docs/results/v2/control/fidelity_onpolicy_validation.json
    python scripts/v2/control_fidelity_onpolicy.py --window-alignment aligned --output docs/results/v2/dream_alignment/fidelity_onpolicy_validation_aligned.json

--window-alignment and --output as in control_fidelity.py (the published run is "legacy").
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.corridor_dream_environment import CorridorDreamEnvironment, world_model_dir  # noqa: E402
from environments.four_intersections import TRAFFIC_SIGNAL_IDS  # noqa: E402
from scripts.v2.control_fidelity import HORIZON, imagined_vs_real, parse_args  # noqa: E402

SCENARIOS = tuple(range(21000, 21006))   # validation only (section 12, Paso 2)
ARCHITECTURES = ("lstm", "transformer")
SEEDS = tuple(range(10))
CONTROL_DIR = ROOT_DIR / "models" / "checkpoints" / "v2" / "control"
OUTPUT = ROOT_DIR / "docs" / "results" / "v2" / "control" / "fidelity_onpolicy_validation.json"
ELAPSED_COLUMN, STATE_BLOCK, STEP_SECONDS = 24, 26, 5
BUCKETS = (("0-5", 0, 6), ("6-10", 6, 11), ("11-19", 11, 20), (">=20", 20, 10**9))
N_BOOT = 2000


def record_trajectories(model, scaled_env, scenarios) -> dict[int, dict]:
    """Run the controller on real SUMO; per scenario: raw and normalized states before each action,
    actions, rewards."""
    raw_env = scaled_env.env
    out = {}
    for scenario in scenarios:
        raw, _ = raw_env.reset(seed=scenario)
        raws, zs, actions, rewards, done = [], [], [], [], False
        while not done:
            z = scaled_env.observation(raw)
            action = np.asarray(model.predict(z, deterministic=True)[0], dtype=np.int64)
            raws.append(raw)
            zs.append(z)
            actions.append(action)
            raw, reward, terminated, truncated, _ = raw_env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        out[scenario] = {"raw": np.array(raws, dtype=np.float32), "z": np.array(zs, dtype=np.float32),
                         "actions": np.array(actions, dtype=np.int64), "rewards": np.array(rewards, dtype=np.float32)}
    return out


def holds(raw_states: np.ndarray) -> np.ndarray:
    """(T, 4) steps each signal has kept its current green phase."""
    cols = [STATE_BLOCK * k + ELAPSED_COLUMN for k in range(len(TRAFFIC_SIGNAL_IDS))]
    return np.floor(raw_states[:, cols] / STEP_SECONDS).astype(int)


def windows_for_controller(arch: str, seed: int, trajectories: dict, tmp: Path,
                           window_alignment: str = "legacy") -> list[dict]:
    path = tmp / f"{arch}_s{seed}.npz"
    ids = sorted(trajectories)
    np.savez(path, z=np.concatenate([trajectories[s]["z"] for s in ids]),
             next_z=np.concatenate([trajectories[s]["z"] for s in ids]),  # unused by load_episodes
             actions=np.concatenate([trajectories[s]["actions"] for s in ids]),
             rewards=np.concatenate([trajectories[s]["rewards"] for s in ids]),
             episode_id=np.concatenate([np.full(len(trajectories[s]["z"]), s) for s in ids]),
             time_step=np.concatenate([np.arange(len(trajectories[s]["z"])) for s in ids]))
    dream = CorridorDreamEnvironment(world_model_dir(arch, seed), seed_episodes_path=path, max_dream_steps=HORIZON,
                                     window_alignment=window_alignment)
    rows = imagined_vs_real(dream)
    L = dream.sequence_length
    records = []
    for scenario, data in rows.items():
        h = holds(trajectories[scenario]["raw"])
        for start, (clipped, raw, real) in enumerate(data):
            t = start + L - 1  # the first imagined decision is taken in state t
            records.append({"arch": arch, "seed": seed, "scenario": int(scenario), "start": start,
                            "imagined_clipped": float(clipped), "imagined_raw": float(raw), "real": float(real),
                            **{f"hold_{ts}": int(h[t, k]) for k, ts in enumerate(TRAFFIC_SIGNAL_IDS)}})
    return records


def _pearson_bias(rows: list[dict], key: str) -> tuple[float, float]:
    im = np.array([r[key] for r in rows])
    re = np.array([r["real"] for r in rows])
    return float(np.corrcoef(im, re)[0, 1]), float((im - re).mean())


def bucket_of(r: dict) -> str:
    longest = max(r[f"hold_{ts}"] for ts in TRAFFIC_SIGNAL_IDS)
    return next(name for name, lo, hi in BUCKETS if lo <= longest < hi)


def summarize(records: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    resamples = [rng.choice(SCENARIOS, len(SCENARIOS)) for _ in range(N_BOOT)]
    out = {}
    for arch in ARCHITECTURES:
        recs = [r for r in records if r["arch"] == arch]
        entry = {"n_windows": len(recs), "real_mean": float(np.mean([r["real"] for r in recs]))}
        for key, label in (("imagined_clipped", "clipped"), ("imagined_raw", "raw")):
            per_ctrl = [_pearson_bias([r for r in recs if r["seed"] == s], key) for s in SEEDS]
            # (imagined, real) per controller and scenario, so a resample only concatenates arrays.
            cells = {(s, sc): np.array([(r[key], r["real"]) for r in recs if r["seed"] == s and r["scenario"] == sc])
                     for s in SEEDS for sc in SCENARIOS}
            boot = []
            for idx in resamples:
                ctrl = []
                for s in SEEDS:
                    xy = np.concatenate([cells[(s, sc)] for sc in idx])
                    ctrl.append((np.corrcoef(xy[:, 0], xy[:, 1])[0, 1] if xy[:, 1].std() > 0 else np.nan,
                                 (xy[:, 0] - xy[:, 1]).mean()))
                boot.append(np.nanmean(np.array(ctrl), axis=0))
            boot = np.array(boot)
            entry[label] = {
                "pearson_mean_over_controllers": float(np.mean([p for p, _ in per_ctrl])),
                "pearson_per_controller": [p for p, _ in per_ctrl],
                "pearson_mean_ci95": np.nanpercentile(boot[:, 0], [2.5, 97.5]).tolist(),
                "bias_mean_over_controllers": float(np.mean([b for _, b in per_ctrl])),
                "bias_per_controller": [b for _, b in per_ctrl],
                "bias_mean_ci95": np.nanpercentile(boot[:, 1], [2.5, 97.5]).tolist(),
                "pearson_pooled": _pearson_bias(recs, key)[0],
            }
        by_bucket = {}
        for name, _, _ in BUCKETS:
            rows = [r for r in recs if bucket_of(r) == name]
            by_bucket[name] = {"n_windows": len(rows)} | ({
                "real_mean": float(np.mean([r["real"] for r in rows])),
                "bias_clipped": float(np.mean([r["imagined_clipped"] - r["real"] for r in rows])),
                "bias_raw": float(np.mean([r["imagined_raw"] - r["real"] for r in rows])),
                "share_positive_bias_raw": float(np.mean([r["imagined_raw"] > r["real"] for r in rows])),
                "controllers": sorted({r["seed"] for r in rows}),
            } if rows else {})
        entry["bias_by_longest_hold"] = by_bucket
        long_by_signal = {}
        for ts in TRAFFIC_SIGNAL_IDS:
            rows = [r for r in recs if r[f"hold_{ts}"] >= 20]
            long_by_signal[ts] = {"n_windows": len(rows)} | ({
                "bias_clipped": float(np.mean([r["imagined_clipped"] - r["real"] for r in rows])),
                "bias_raw": float(np.mean([r["imagined_raw"] - r["real"] for r in rows])),
                "real_mean": float(np.mean([r["real"] for r in rows])),
            } if rows else {})
        entry["hold_ge_20_by_signal"] = long_by_signal
        out[arch] = entry
    return out


def main() -> None:
    args = parse_args(OUTPUT)
    from stable_baselines3 import PPO

    from environments.scaled_corridor_environment import ScaledCorridorEnvironment

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    records, reproduction = [], []
    scaled_env = ScaledCorridorEnvironment()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for arch in ARCHITECTURES:
                for seed in SEEDS:
                    model = PPO.load(CONTROL_DIR / f"dream_{arch}_s{seed}" / "best_model.zip", device="cpu")
                    trajectories = record_trajectories(model, scaled_env, SCENARIOS)
                    records += windows_for_controller(arch, seed, trajectories, Path(tmp), args.window_alignment)
                    # Check: on the selection seeds 21000-21004 the recorded trajectories must give
                    # the best mean real return the training's EvalCallback logged.
                    with np.load(CONTROL_DIR / f"dream_{arch}_s{seed}" / "evaluations.npz") as ev:
                        logged = float(ev["results"].mean(axis=1).max())
                    replayed = float(np.mean([trajectories[s]["rewards"].sum() for s in SCENARIOS[:5]]))
                    reproduction.append({"controller": f"{arch}_s{seed}", "logged": logged, "replayed": replayed})
                    print(f"{arch} s{seed}: real returns {[round(float(t['rewards'].sum())) for t in trajectories.values()]} "
                          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        scaled_env.close()
    result = {"scenarios": list(SCENARIOS), "horizon": HORIZON, "n_boot": N_BOOT,
              "dataset_action_fidelity": "docs/results/v2/control/fidelity_validation.json",
              "summary": summarize(records), "selection_reproduction": reproduction, "windows": records, "seconds": time.perf_counter() - t0}
    result["window_alignment"] = args.window_alignment
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for arch, e in result["summary"].items():
        for label in ("clipped", "raw"):
            s = e[label]
            print(f"{arch:11s} {label:7s}: Pearson {s['pearson_mean_over_controllers']:.3f} {np.round(s['pearson_mean_ci95'], 3)}"
                  f" (agrupado {s['pearson_pooled']:.3f}); sesgo {s['bias_mean_over_controllers']:+.1f} {np.round(s['bias_mean_ci95'], 1)}")
        for name, b in e["bias_by_longest_hold"].items():
            if b["n_windows"]:
                print(f"   retencion {name:5s}: n {b['n_windows']:5d}, real {b['real_mean']:9.1f}, sesgo recortado "
                      f"{b['bias_clipped']:+9.1f}, sin recortar {b['bias_raw']:+9.1f}, imaginado > real {b['share_positive_bias_raw']:.0%}")
        print("   >=20 por semaforo:", {ts: v for ts, v in e["hold_ge_20_by_signal"].items()})
    worst = max(abs(r["logged"] - r["replayed"]) for r in reproduction)
    print(f"reproduccion de la seleccion (21000-21004): diferencia maxima {worst:.4f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
