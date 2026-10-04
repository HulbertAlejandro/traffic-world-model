"""Section 9 of docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md: LSTM, Transformer and TSMixer, parameter-matched
(~152k each), uncompressed 104-dim state, 10 seeds each. Three pairwise comparisons, Bonferroni-corrected.

    python scripts/v2/analyze_three_architectures.py

Score per model: log GM (log of the geometric mean over h = 1..10 of the test reward_mse). For each
pair (a, b), d = mean log GM (a) - mean log GM (b), unpaired (seed N of one architecture shares no
randomness with seed N of another):

- Welch t test, two-sided p, and the Welch CI at level 1 - 0.05/3 (98.33%).
- Percentile bootstrap CI at the same level: 10,000 resamples, numpy seed 0, each arm drawn separately.
- Pair decision: both corrected CIs exclude 0 on the same side -> the lower-error architecture wins the
  pair; otherwise "sin evidencia suficiente".
- Overall: an architecture is adopted as v2's official temporal model only if it wins BOTH of its pairs.
- Advance to the control phase (section 9.4): every architecture that loses none of its pairs. All three
  advance if no pair is resolved. This decides who is tested in control, not which one is official.

The original 95% analysis of LSTM vs Transformer (sections 5 and 8, analysis.json) is not rewritten;
this script reports that pair again at the corrected level. Secondary (no decision): per horizon,
epochs to convergence, spread between seeds. Output: docs/results/v2/arch_comparison/analysis_three_architectures.json
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations

import numpy as np
import torch

from analyze_lstm_vs_transformer import HORIZONS, RESULTS, ROOT, SEEDS, bootstrap_ci, run_dir, welch
from run_lstm_vs_transformer import ARCH_HPARAMS, EXPECTED_PARAMS

ARCHS = ("lstm", "transformer", "tsmixer")
PAIRS = (("transformer", "lstm"), ("tsmixer", "lstm"), ("tsmixer", "transformer"))  # d = first - second
LEVEL = 1.0 - 0.05 / len(PAIRS)


def compare_at(x: np.ndarray, y: np.ndarray, level: float = LEVEL) -> dict:
    w = welch(x, y, level)
    boot = bootstrap_ci(x, y, np.random.default_rng(0), level)
    return {"level": level, "d": w["d"], "relative": math.exp(w["d"]) - 1, "t": w["t"], "df": w["df"], "p": w["p"],
            "welch_ci": w["ci95"], "bootstrap_ci": boot,
            "relative_welch_ci": [math.exp(v) - 1 for v in w["ci95"]],
            "relative_bootstrap_ci": [math.exp(v) - 1 for v in boot]}


def decide_pair(c: dict, a: str, b: str) -> str:
    """Winner of the pair (a, b), with d = a - b; both corrected CIs must exclude 0 on the same side."""
    (wl, wh), (bl, bh) = c["welch_ci"], c["bootstrap_ci"]
    if wh < 0 and bh < 0:
        return a
    if wl > 0 and bl > 0:
        return b
    return "sin evidencia suficiente"


def overall(pair_winners: dict[tuple[str, str], str]) -> str:
    for arch in ARCHS:
        mine = [w for (a, b), w in pair_winners.items() if arch in (a, b)]
        if len(mine) == 2 and all(w == arch for w in mine):
            return arch
    return "sin evidencia suficiente"


def advancing(pair_winners: dict[tuple[str, str], str]) -> list[str]:
    """Architectures not significantly worse than any other (they lose none of their pairs)."""
    losers = {b if w == a else a for (a, b), w in pair_winners.items() if w in (a, b)}
    return [arch for arch in ARCHS if arch not in losers]


def load_runs() -> dict:
    runs = {}
    for arch in ARCHS:
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
                               "stopped_epoch": hp["stopped_epoch"], "hit_max_epochs": hp["hit_max_epochs"],
                               "n_params": n_params, "protocol": hp["protocol"],
                               "weights_md5": hashlib.md5((d / "world_model_best.pt").read_bytes()).hexdigest()}
    assert len({json.dumps(r["protocol"], sort_keys=True) for r in runs.values()}) == 1, "one training protocol"
    return runs


def main() -> None:
    runs = load_runs()

    def log_gm(arch, horizons=HORIZONS):
        return np.array([np.mean([np.log(runs[(arch, s)]["reward_mse"][str(h)]) for h in horizons]) for s in SEEDS])

    pairs, winners = {}, {}
    for a, b in PAIRS:
        c = compare_at(log_gm(a), log_gm(b))
        winners[(a, b)] = decide_pair(c, a, b)
        pairs[f"{a}_minus_{b}"] = {**c, "decision": winners[(a, b)],
                                   "per_horizon": {str(h): compare_at(log_gm(a, [h]), log_gm(b, [h])) for h in HORIZONS}}
    per_arch = {}
    for arch in ARCHS:
        lg = log_gm(arch)
        per_arch[arch] = {"n_params": EXPECTED_PARAMS[arch], "hparams": ARCH_HPARAMS[arch], "log_gm": lg.tolist(),
                          "mean_log_gm": float(lg.mean()), "sd_log_gm": float(lg.std(ddof=1)),
                          "gm_reward_mse": [runs[(arch, s)]["gm_reward_mse"] for s in SEEDS],
                          "mean_log_reward_mse_per_horizon": {
                              str(h): float(np.mean(log_gm(arch, [h]))) for h in HORIZONS},
                          "best_epochs": [runs[(arch, s)]["best_epoch"] for s in SEEDS],
                          "stopped_epochs": [runs[(arch, s)]["stopped_epoch"] for s in SEEDS],
                          "hit_max_epochs": sum(runs[(arch, s)]["hit_max_epochs"] for s in SEEDS)}
    out = {"level": LEVEL, "pairs": pairs, "overall_decision": overall(winners),
           "advance_to_control": advancing(winners), "per_arch": per_arch,
           "variance_ratios": {f"{a}_over_{b}": (per_arch[a]["sd_log_gm"] / per_arch[b]["sd_log_gm"]) ** 2
                               for a, b in combinations(ARCHS, 2)},
           "runs": {f"{a}_s{s}": {k: v for k, v in r.items() if k != "protocol"} for (a, s), r in runs.items()},
           "protocol": runs[("lstm", 0)]["protocol"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "analysis_three_architectures.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"Bonferroni level {LEVEL:.4%}")
    for arch, a in per_arch.items():
        print(f"{arch:12s} {a['n_params']} params  mean log GM {a['mean_log_gm']:.4f}  sd {a['sd_log_gm']:.4f}  "
              f"stopped {min(a['stopped_epochs'])}-{max(a['stopped_epochs'])}  hit max {a['hit_max_epochs']}")
    for name, c in pairs.items():
        print(f"{name:26s} {c['relative']:+.1%}  p = {c['p']:.3g}  Welch CI {c['relative_welch_ci'][0]:+.1%} .. "
              f"{c['relative_welch_ci'][1]:+.1%}  bootstrap {c['relative_bootstrap_ci'][0]:+.1%} .. "
              f"{c['relative_bootstrap_ci'][1]:+.1%}  -> {c['decision']}")
    print("overall:", out["overall_decision"])
    print("advance to control:", out["advance_to_control"])


if __name__ == "__main__":
    main()
