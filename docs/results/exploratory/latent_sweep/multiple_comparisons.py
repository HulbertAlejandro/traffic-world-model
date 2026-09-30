"""REPORT.md section 13, complementary (NOT pre-registered): Bonferroni-adjusted intervals for the
three per-architecture tests and the three between-architecture comparisons of
ADDENDUM_transformer_tsmixer.md. Same bootstrap as grid_arch.py (B = 100,000, seed 12345, same
draw order), only the percentiles change: 100 * (0.05 / 3) / 2 and 100 * (1 - (0.05 / 3) / 2).
Reads the cached per-run curves gridarch_{lstm,transformer,tsmixer}.json; numpy only."""
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARCHS = ("lstm", "transformer", "tsmixer")
B, SEED, M = 100_000, 12345, 3
Q = [100 * 0.05 / M / 2, 100 * (1 - 0.05 / M / 2)]


def boot_delta(G, raw, rng, ai=None):  # copied from grid_arch.py
    ai = rng.integers(0, 5, size=(B, 5)) if ai is None else ai
    si = rng.integers(0, 5, size=(B, 5, 5))
    zb = G[ai[:, :, None], si].mean(axis=(1, 2))
    rb = raw[rng.integers(0, len(raw), size=(B, len(raw)))].mean(axis=1)
    return zb - rb


pct = lambda d: 100 * (math.exp(d) - 1)
GR = {}
for a in ARCHS:
    e = json.loads((HERE / f"gridarch_{a}.json").read_text(encoding="utf-8"))
    GR[a] = (np.log(np.array(e["z_reward_mse"])).mean(-1), np.log(np.array(e["raw_reward_mse"])).mean(-1))
out = {"bonferroni_level": 1 - 0.05 / M, "per_arch": {}, "pairs": {}}
for a, (G, raw) in GR.items():
    db = boot_delta(G, raw, np.random.default_rng(SEED))
    lo95, hi95 = np.percentile(db, [2.5, 97.5]); lo, hi = np.percentile(db, Q)
    print(f"{a:<12} IC95 [{pct(lo95):+.1f}%, {pct(hi95):+.1f}%]  Bonferroni {100 * (1 - 0.05 / M):.2f}% [{pct(lo):+.1f}%, {pct(hi):+.1f}%]")
    out["per_arch"][a] = {"ci95": [lo95, hi95], "ci_bonferroni": [lo, hi]}
for A, Bn in itertools.combinations(ARCHS, 2):
    rng = np.random.default_rng(SEED)
    ai = rng.integers(0, 5, size=(B, 5))
    d = boot_delta(*GR[A], rng, ai) - boot_delta(*GR[Bn], rng, ai)
    lo, hi = np.percentile(d, Q)
    print(f"{A} - {Bn}: Bonferroni [{lo:+.4f}, {hi:+.4f}] -> {'incluye 0' if lo <= 0 <= hi else 'excluye 0'}")
    out["pairs"][f"{A}-{Bn}"] = {"ci_bonferroni": [lo, hi]}
(HERE / "multiple_comparisons.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
