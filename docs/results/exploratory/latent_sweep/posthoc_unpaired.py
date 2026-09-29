"""POST HOC (not pre-registered): unpaired comparison of the 10-run groups, since the seed pairing
between branches is nominal (different input sizes -> different initializations and batch orders).
Per run: geometric mean over h=1..10 of reward_mse (each horizon weighs the same).
Exact two-sided permutation test on the difference of group means of log reward_mse (C(20,10))."""
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
H = [str(h) for h in range(1, 11)]
win = json.loads((HERE / "winner_z12_10seeds.json").read_text(encoding="utf-8"))
old = json.loads((HERE.parent / "seed_zero_check/experiment_0_10seeds_combined.json").read_text(encoding="utf-8"))


def runs(report, branch):
    return np.array([[p[branch][h]["model_reward_mse"] for h in H] for p in report["pairs"]])


groups = {"z12 (24 dims)": runs(win, "z"), "crudo 24": runs(win, "raw"),
          "z16 oficial (26 dims)": runs(old, "z"), "crudo 26": runs(old, "raw")}
score = {g: np.log(v).mean(axis=1) for g, v in groups.items()}
print("mediana por horizonte de reward_mse (10 corridas por grupo)")
print("h  | " + " | ".join(f"{g:>22}" for g in groups))
for i, h in enumerate(H):
    print(f"{h:>2} | " + " | ".join(f"{np.median(v[:, i]):22.1f}" for v in groups.values()))
print("\nmedia geometrica sobre h por corrida (mediana del grupo):", {g: round(float(np.exp(np.median(s))), 1) for g, s in score.items()})


def perm(a, b):
    x = np.concatenate([a, b]); d = a.mean() - b.mean()
    null = [x[list(c)].mean() - np.delete(x, list(c)).mean() for c in itertools.combinations(range(len(x)), len(a))]
    return float(d), float(np.mean(np.abs(null) >= abs(d) - 1e-12))


out = {}
for a, b in [("z12 (24 dims)", "crudo 24"), ("z12 (24 dims)", "crudo 26"), ("crudo 24", "crudo 26"),
             ("z16 oficial (26 dims)", "crudo 26"), ("z12 (24 dims)", "z16 oficial (26 dims)")]:
    d, p = perm(score[a], score[b])
    out[f"{a} vs {b}"] = {"ratio_geomean": float(np.exp(d)), "p": p}
    print(f"{a} vs {b}: razon de medias geometricas {np.exp(d):.3f} ({100 * (np.exp(d) - 1):+.1f}%), permutacion exacta p = {p:.4f}")
(HERE / "posthoc_unpaired.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
