"""Analysis of the pre-registered, parameter-matched LSTM vs Transformer comparison
(docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md): LSTM hidden 137 vs Transformer d_model 80, ~152k parameters each.

    python scripts/v2/analyze_lstm_vs_transformer.py

Per model, the score is log GM: the log of the geometric mean over horizons 1..10 of the test
reward_mse (the Phase 2 metric). The difference is

    d = mean log GM (Transformer) - mean log GM (LSTM)      (< 0: the Transformer predicts better)
    relative = exp(d) - 1

Unpaired: seed N of one architecture shares no source of randomness with seed N of the other.

- Welch t test on the 10 vs 10 log GM, with its two-sided p and 95% CI of d.
- Percentile bootstrap 95% CI of d: 10,000 resamples, numpy seed 0, each arm's seeds drawn with
  replacement independently.
- Decision (section 4 of the addendum): both 95% CIs exclude 0 on the same side -> the architecture
  with the lower error is adopted; otherwise "sin evidencia suficiente".

Secondary (descriptive, no decision): the same per horizon, epochs to convergence and spread
between seeds. Results: docs/results/v2/arch_comparison/analysis.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_lstm_vs_transformer import ARCH_HPARAMS, EXPECTED_PARAMS

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "checkpoints" / "v2"
RESULTS = ROOT / "docs" / "results" / "v2" / "arch_comparison"
HORIZONS = range(1, 11)
SEEDS = range(10)
N_BOOT = 10_000


def run_dir(arch: str, seed: int) -> Path:
    return CKPT / "arch_comparison" / f"{arch}_raw_s{seed}"


# --------------------------------------------------------------------------- Student t (no scipy in the venv)
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta (Numerical Recipes, betacf)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(q: float, df: float) -> float:
    """Upper quantile: the t with P(T > t) = 1 - q, by bisection (q > 0.5)."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) / 2.0 > 1.0 - q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> dict:
    """Welch test of mean(x) - mean(y): t, df, two-sided p and the ``level`` CI (key "ci95" kept for 0.95)."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    d = float(x.mean() - y.mean())
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    t = d / se
    half = t_quantile(round(1.0 - (1.0 - level) / 2.0, 12), df) * se
    return {"d": d, "t": t, "df": df, "p": t_two_sided_p(t, df), "ci95": [d - half, d + half]}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, level: float = 0.95) -> list[float]:
    bx = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    by = y[rng.integers(0, len(y), (N_BOOT, len(y)))].mean(1)
    tail = round(100.0 * (1.0 - level) / 2.0, 10)  # exactly 2.5 at 0.95: analysis.json stays byte-identical
    lo, hi = np.percentile(bx - by, [tail, 100.0 - tail])
    return [float(lo), float(hi)]


def compare(tf: np.ndarray, lstm: np.ndarray, level: float = 0.95) -> dict:
    """Difference mean(tf) - mean(lstm); the argument names are the original comparison's."""
    w = welch(tf, lstm, level)
    boot = bootstrap_ci(tf, lstm, np.random.default_rng(0), level)
    return {**w, "relative": math.exp(w["d"]) - 1, "relative_ci95_welch": [math.exp(v) - 1 for v in w["ci95"]],
            "bootstrap_ci95": boot, "relative_ci95_bootstrap": [math.exp(v) - 1 for v in boot]}


def decide(c: dict) -> str:
    (wl, wh), (bl, bh) = c["ci95"], c["bootstrap_ci95"]
    if wh < 0 and bh < 0:
        return "transformer"
    if wl > 0 and bl > 0:
        return "lstm"
    return "sin evidencia suficiente"


# --------------------------------------------------------------------------- main
def main() -> None:
    runs = {}
    for arch in ("lstm", "transformer"):
        for s in SEEDS:
            d = run_dir(arch, s)
            ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
            hp = json.loads((d / "world_model_best.json").read_text(encoding="utf-8"))
            assert hp["architecture"] == arch and hp["latent_dim"] == 104 and hp["seed"] == s, d
            assert all(hp[k] == v for k, v in ARCH_HPARAMS[arch].items()), f"{d}: not the pre-registered sizes"
            weights = torch.load(d / "world_model_best.pt", map_location="cpu", weights_only=True)
            n_params = sum(t.numel() for t in weights.values())
            assert n_params == EXPECTED_PARAMS[arch], f"{d}: {n_params} parameters"
            runs[(arch, s)] = {"dir": d.relative_to(ROOT).as_posix(), "reward_mse": ev["evaluation"]["reward_mse"],
                               "gm_reward_mse": ev["evaluation"]["gm_reward_mse"], "best_epoch": hp["best_epoch"],
                               "stopped_epoch": hp["stopped_epoch"], "hit_max_epochs": hp["hit_max_epochs"], "n_params": n_params,
                               "protocol": hp["protocol"],
                               "weights_md5": hashlib.md5((d / "world_model_best.pt").read_bytes()).hexdigest()}
    protocols = {json.dumps(r["protocol"], sort_keys=True) for r in runs.values()}
    assert len(protocols) == 1, "the 20 runs must share one training protocol"

    def log_gm(arch, seeds, horizons=HORIZONS):
        return np.array([np.mean([np.log(runs[(arch, s)]["reward_mse"][str(h)]) for h in horizons]) for s in seeds])

    primary = compare(log_gm("transformer", SEEDS), log_gm("lstm", SEEDS))
    out = {
        "primary": {**primary, "decision": decide(primary)},
        "per_horizon": {str(h): compare(log_gm("transformer", SEEDS, [h]), log_gm("lstm", SEEDS, [h]))
                        for h in HORIZONS},
        "per_arch": {},
        "runs": {f"{a}_s{s}": {k: v for k, v in r.items() if k != "protocol"} for (a, s), r in runs.items()},
        "protocol": runs[("lstm", 0)]["protocol"],
        "architecture_hparams": {a: ARCH_HPARAMS[a] for a in ("lstm", "transformer")},
        "n_params": {a: EXPECTED_PARAMS[a] for a in ("lstm", "transformer")},
    }
    for arch in ("lstm", "transformer"):
        lg = log_gm(arch, SEEDS)
        out["per_arch"][arch] = {
            "log_gm": lg.tolist(), "mean_log_gm": float(lg.mean()), "sd_log_gm": float(lg.std(ddof=1)),
            "gm_reward_mse": [runs[(arch, s)]["gm_reward_mse"] for s in SEEDS],
            "best_epochs": [runs[(arch, s)]["best_epoch"] for s in SEEDS],
            "stopped_epochs": [runs[(arch, s)]["stopped_epoch"] for s in SEEDS],
            "hit_max_epochs": sum(runs[(arch, s)]["hit_max_epochs"] for s in SEEDS)}
    # Context outside the comparison (addendum section 6): Phase 2's size-128 models, 3 seeds each.
    sel = CKPT / "compression" / "selection"
    out["context_phase2_size_128"] = {}
    for arch, prefix in (("lstm", "raw_s"), ("transformer", "tf_raw_s")):
        lg = [np.mean([np.log(v) for v in json.loads((sel / f"{prefix}{s}" / "evaluation.json").read_text(
            encoding="utf-8"))["evaluation"]["reward_mse"].values()]) for s in range(3)]
        out["context_phase2_size_128"][arch] = {"log_gm": [float(v) for v in lg], "mean_log_gm": float(np.mean(lg))}
    sd_l, sd_t = out["per_arch"]["lstm"]["sd_log_gm"], out["per_arch"]["transformer"]["sd_log_gm"]
    out["variance_ratio_transformer_over_lstm"] = (sd_t / sd_l) ** 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    p = out["primary"]
    print(f"Transformer - LSTM, mean log GM: d = {p['d']:+.4f} ({p['relative']:+.1%})")
    print(f"  Welch t = {p['t']:.3f}, df = {p['df']:.1f}, p = {p['p']:.3g}, "
          f"95% CI {p['relative_ci95_welch'][0]:+.1%} .. {p['relative_ci95_welch'][1]:+.1%}")
    print(f"  bootstrap 95% CI {p['relative_ci95_bootstrap'][0]:+.1%} .. {p['relative_ci95_bootstrap'][1]:+.1%}")
    print(f"  decision: {p['decision']}")
    for arch, a in out["per_arch"].items():
        print(f"{arch:12s} mean log GM {a['mean_log_gm']:.4f}  sd {a['sd_log_gm']:.4f}  "
              f"stopped {min(a['stopped_epochs'])}-{max(a['stopped_epochs'])}  hit max {a['hit_max_epochs']}")
    print("per horizon:", " ".join(f"h{h} {c['relative']:+.1%} (p {c['p']:.2g})" for h, c in out["per_horizon"].items()))
    print(f"-> {RESULTS / 'analysis.json'}")


if __name__ == "__main__":
    main()
