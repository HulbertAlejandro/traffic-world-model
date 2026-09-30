"""Exact sign test and exact Wilcoxon signed-rank (two-sided) on per-seed reductions (MANIFEST.md)."""
import json
import sys
from math import comb

import numpy as np


def sign_test(x):
    n, k = len(x), int((np.asarray(x) > 0).sum())
    tail = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilcoxon_exact(x):
    x = np.asarray(x, dtype=float)
    ranks = np.argsort(np.argsort(np.abs(x))) + 1  # no ties/zeros expected; checked below
    assert len(set(np.abs(x))) == len(x) and np.all(x != 0)
    w = ranks[x > 0].sum(); mu = ranks.sum() / 2; n = len(x)
    null = np.array([sum(ranks[i] for i in range(n) if s >> i & 1) for s in range(2 ** n)])
    return int(w), float(np.mean(np.abs(null - mu) >= abs(w - mu) - 1e-9))


if __name__ == "__main__":
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    seeds = [s for s in report["per_seed"] if len(sys.argv) < 3 or s["seed"] in json.loads(sys.argv[2])]
    r = [s["median_reduction"] for s in seeds]
    w, p = wilcoxon_exact(r)
    print(f"seeds {[s['seed'] for s in seeds]}: positive {sum(v > 0 for v in r)}/{len(r)}, "
          f"sign p={sign_test(r):.3f}, Wilcoxon W+={w} p={p:.3f}, median {100*np.median(r):+.1f}%, mean {100*np.mean(r):+.1f}%")
