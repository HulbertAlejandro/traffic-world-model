"""Pre-registered analysis of the compression experiment (docs/v2/ADDENDUM_AUTOENCODER.md, section 3).

    python scripts/v2/analyze_compression_experiment.py --phase selection
    python scripts/v2/analyze_compression_experiment.py --phase confirmation

Per model, the score is log GM: the log of the geometric mean over horizons 1..10 of the test
reward_mse. For a latent size, the score is the mean log GM over its (Autoencoder, LSTM) models;
for the raw branch, the mean over its LSTM seeds.

    delta = exp(mean log GM (z) - mean log GM (raw)) - 1        (< 0: compression helps)

95% CI of delta: two-level cluster bootstrap, 10,000 resamples, numpy seed 0. Each resample draws
Autoencoders with replacement and, inside each drawn Autoencoder, its LSTM seeds with replacement;
the raw branch draws its seeds with replacement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS = ROOT_DIR / "docs" / "results" / "v2" / "autoencoder"
N_BOOT, BOOT_SEED, HORIZONS = 10_000, 0, range(1, 11)
CONSTANT_COLUMNS = [26 * b + j for b in range(4) for j in (22, 23)]  # yellow phase_one_hot positions


def _log_gm(model: dict, horizons=HORIZONS) -> float:
    mse = model["evaluation"]["reward_mse"]
    return float(np.mean([np.log(mse[str(h)]) for h in horizons]))


def cluster_bootstrap(z_by_ae: dict[int, list[float]], raw: list[float], rng) -> np.ndarray:
    aes = list(z_by_ae)
    z_arr = [np.asarray(z_by_ae[a]) for a in aes]
    raw = np.asarray(raw)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.integers(0, len(aes), len(aes))
        z_mean = np.mean([z_arr[i][rng.integers(0, len(z_arr[i]), len(z_arr[i]))].mean() for i in picked])
        raw_mean = raw[rng.integers(0, len(raw), len(raw))].mean()
        out[b] = np.exp(z_mean - raw_mean) - 1
    return out


def summarize_size(models: list[dict], raw_models: list[dict], horizons=HORIZONS) -> dict:
    z_by_ae: dict[int, list[float]] = {}
    for m in models:
        z_by_ae.setdefault(m["ae_seed"], []).append(_log_gm(m, horizons))
    raw = [_log_gm(m, horizons) for m in raw_models]
    z_all = [v for vs in z_by_ae.values() for v in vs]
    delta = float(np.exp(np.mean(z_all) - np.mean(raw)) - 1)
    boot = cluster_bootstrap(z_by_ae, raw, np.random.default_rng(BOOT_SEED))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ae_means = np.array([np.mean(v) for v in z_by_ae.values()])
    within = np.mean([np.var(v, ddof=1) for v in z_by_ae.values()]) if all(len(v) > 1 for v in z_by_ae.values()) else None
    return {"n_ae": len(z_by_ae), "n_z_models": len(z_all), "n_raw": len(raw),
            "mean_log_gm_z": float(np.mean(z_all)), "mean_log_gm_raw": float(np.mean(raw)),
            "delta": delta, "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "variance_between_ae": float(np.var(ae_means, ddof=1)) if len(ae_means) > 1 else None,
            "variance_within_ae": float(within) if within is not None else None,
            "variance_raw_seeds": float(np.var(raw, ddof=1)) if len(raw) > 1 else None,
            "per_ae_delta": {str(a): float(np.exp(np.mean(v) - np.mean(raw)) - 1) for a, v in z_by_ae.items()}}


def per_horizon(models: list[dict], raw_models: list[dict]) -> dict:
    """Per horizon: delta with its cluster-bootstrap CI, and v1's secondary statistics over all
    (z model, raw model) pairs: share of pairs z wins and median relative reduction (raw - z) / raw."""
    out = {}
    for h in HORIZONS:
        s = summarize_size(models, raw_models, horizons=[h])
        z = np.array([m["evaluation"]["reward_mse"][str(h)] for m in models])
        r = np.array([m["evaluation"]["reward_mse"][str(h)] for m in raw_models])
        red = (r[None, :] - z[:, None]) / r[None, :]
        out[str(h)] = {"delta": s["delta"], "ci95": s["ci95"], "pairs_z_wins": float((red > 0).mean()),
                       "median_relative_reduction": float(np.median(red))}
    return out


def episode_concentration(models: list[dict], h: int = 10) -> dict:
    """v1 pattern check: how much of the test error at horizon h a few episodes carry."""
    shares, medians = [], []
    for m in models:
        per_ep = np.array([v[str(h)] for v in m["evaluation"]["per_episode_reward_mse"].values()])
        shares.append(np.sort(per_ep)[-3:].sum() / per_ep.sum())
        medians.append(np.median(per_ep))
    return {"episodes": len(per_ep), "top3_share_mean": float(np.mean(shares)), "top3_share_max": float(np.max(shares)),
            "median_episode_mse_mean": float(np.mean(medians))}


def worst_episodes(models: list[dict], h: int = 10, k: int = 3) -> list[int]:
    totals: dict[str, float] = {}
    for m in models:
        for ep, v in m["evaluation"]["per_episode_reward_mse"].items():
            totals[ep] = totals.get(ep, 0.0) + v[str(h)]
    return [int(e) for e in sorted(totals, key=totals.get, reverse=True)[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=("selection", "confirmation"))
    args = parser.parse_args()
    data = json.loads((RESULTS / f"{args.phase}.json").read_text(encoding="utf-8"))
    raw = [m for m in data["models"] if m["branch"] == "raw"]
    out = {"phase": args.phase, "sizes": {}, "raw": {
        "gm_reward_mse": [m["evaluation"]["gm_reward_mse"] for m in raw],
        "episode_concentration_h10": episode_concentration(raw),
        "persistence_gm_reward_mse": float(np.exp(np.mean(np.log(
            [raw[0]["evaluation"]["baseline_reward_mse"][str(h)] for h in HORIZONS])))),
        "stopped_epochs": [m["training"]["stopped_epoch"] for m in raw],
        "hit_max_epochs": sum(m["training"]["hit_max_epochs"] for m in raw)}}
    for k in data["sizes"]:
        models = [m for m in data["models"] if m["branch"] == "z" and m["latent_dim"] == k]
        aes = [a for a in data["autoencoders"] if a["latent_dim"] == k]
        const_err = [float(np.mean([a["reconstruction"]["per_column"][c] for c in CONSTANT_COLUMNS])) for a in aes]
        out["sizes"][str(k)] = summarize_size(models, raw) | {
            "per_horizon": per_horizon(models, raw),
            "episode_concentration_h10": episode_concentration(models),
            "worst_episodes_h10": worst_episodes(models),
            "stopped_epochs": [m["training"]["stopped_epoch"] for m in models],
            "hit_max_epochs": sum(m["training"]["hit_max_epochs"] for m in models),
            "ae_validation_mse": [a["reconstruction"]["mse"] for a in aes],
            "ae_constant_columns_mse": const_err,
            "ae_best_epochs": [a["best_epoch"] for a in aes]}
    out["raw"]["worst_episodes_h10"] = worst_episodes(raw)
    if args.phase == "selection":
        out["selected_latent_dim"] = int(min(out["sizes"], key=lambda k: out["sizes"][k]["mean_log_gm_z"]))
    (RESULTS / f"{args.phase}_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"== {args.phase}: raw GM reward_mse {np.round(out['raw']['gm_reward_mse'], 1).tolist()}, "
          f"persistence {out['raw']['persistence_gm_reward_mse']:.1f}")
    for k, s in out["sizes"].items():
        print(f"k={k:>3s}: delta {s['delta']:+.1%} CI95 [{s['ci95'][0]:+.1%}, {s['ci95'][1]:+.1%}] "
              f"({s['n_ae']} AE x {s['n_z_models'] // s['n_ae']} LSTM vs {s['n_raw']} raw) | "
              f"var between AE {s['variance_between_ae']:.4f}, within {s['variance_within_ae']}, "
              f"raw {s['variance_raw_seeds']:.4f} | AE val mse {np.mean(s['ae_validation_mse']):.4f}")
    if "selected_latent_dim" in out:
        print("selected latent_dim:", out["selected_latent_dim"])


if __name__ == "__main__":
    main()
