"""ADDENDUM.md analysis: Table 3 with 10 seeds per controller, and p-values before (3 / 4 seeds, the
published episodes) and after (10 / 10).

Tests: the published ones, imported unchanged from scripts/evaluate_multiseed_statistical.py
(seed-level Welch, paired t by scenario), plus a paired Wilcoxon signed-rank test by scenario
(exact when there are no ties or zeros; validated here against brute-force enumeration).

Controls: (3) every existing checkpoint re-evaluated in the new run reproduces its published
per-episode rewards exactly (matched by rewards, not by file name); fixed-time and the rule too.
(4) The published episodes reproduce the published means and p-values.

    python docs/results/ppo_10_seeds/analyze.py              # full analysis (needs the new evaluation)
    python docs/results/ppo_10_seeds/analyze.py --before-only  # controls 4 and Wilcoxon validation only
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_multiseed_statistical import CATASTROPHIC_THRESHOLD, paired_t, welch  # noqa: E402

RES = ROOT / "docs" / "results"
SETS = {"official": "Escenarios oficiales (3000-3014, 5000-5014)", "fresh": "Escenarios nuevos (7000-7029)"}
PUBLISHED_FILES = {
    "official": ["multiseed_official_scenarios.json", "direct_10k_fix_seed3_official_scenarios.json",
                 "direct_30k_fix_seed3_official_scenarios.json"],
    "fresh": ["multiseed_fresh_scenarios_7000.json", "direct_10k_fix_seed3_fresh_scenarios_7000.json",
              "direct_30k_fix_seed3_fresh_scenarios_7000.json"],
}
NEW_FILES = {"official": HERE / "eval_official_scenarios.json", "fresh": HERE / "eval_fresh_scenarios_7000.json"}
PUB_NAME = {"sueno": "dream", "directo_10k_fix": "d10k", "directo_10k_fix_s3": "d10k", "directo_30k_fix": "d30k",
            "directo_30k_fix_s3": "d30k", "tiempo_fijo": "fixed", "regla": "rule"}
NEW_NAME = {"sueno": "dream", "directo_10k": "d10k", "directo_30k": "d30k", "tiempo_fijo": "fixed", "regla": "rule"}
LABEL = {"dream": "PPO + World Model", "d10k": "PPO directo, 10.000 pasos", "d30k": "PPO directo, 30.000 pasos",
         "fixed": "Tiempo fijo", "rule": "Regla fase contraria"}
# Published (PROJECT_STATUS.md, "Auditoría técnica y correcciones", punto 2), as printed there.
PUB_MEANS = {"official": {"dream": -326.79, "d10k": -454.51, "d30k": -402.13, "fixed": -411.27},
             "fresh": {"dream": -293.81, "d10k": -439.73, "d30k": -316.59, "fixed": -391.70}}
PUB_P = {("official", "d10k"): ("0.003", "0.0002"), ("fresh", "d10k"): ("0.077", "4.5e-9"),
         ("official", "d30k"): ("0.029", "0.020"), ("fresh", "d30k"): ("0.021", "0.54")}
INTERACTIONS = {"dream": "3,480 amortizado (4,800 / 10 + 3,000); 7,800 sin amortizar", "d10k": "13,240",
                "d30k": "39,208", "fixed": "no aplica"}


# ------------------------------------------------------------------ Wilcoxon signed-rank (paired)
def _ranks(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank on d = a - b. Exact (DP over rank sums) when there are no
    zeros or ties in |d|; otherwise zeros dropped and normal approximation with tie correction."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n_zero = int((d == 0).sum())
    d = d[d != 0]
    n = len(d)
    r = _ranks(np.abs(d))
    w_plus = float(r[d > 0].sum())
    ties = len(np.unique(np.abs(d))) < n
    if n_zero == 0 and not ties:
        counts = np.zeros(n * (n + 1) // 2 + 1)
        counts[0] = 1
        for k in range(1, n + 1):
            counts[k:] = counts[k:] + counts[:-k].copy()
        total = 2.0 ** n
        w = int(round(w_plus))
        p = min(1.0, 2 * min(counts[:w + 1].sum(), counts[w:].sum()) / total)
        return {"W_plus": w_plus, "n": n, "p": float(p), "method": "exacto"}
    mu = n * (n + 1) / 4
    _, cnt = np.unique(np.abs(d), return_counts=True)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - ((cnt ** 3 - cnt).sum()) / 48)
    z = (w_plus - mu) / sigma
    return {"W_plus": w_plus, "n": n, "p": math.erfc(abs(z) / math.sqrt(2)), "method": f"normal (ceros {n_zero}, empates {ties})"}


def validate_wilcoxon():
    rng = np.random.default_rng(0)
    for n in range(1, 13):
        for _ in range(20):
            a, b = rng.normal(size=n), rng.normal(size=n)
            d = a - b
            r = _ranks(np.abs(d))
            obs = r[d > 0].sum()
            sums = [sum(r[i] for i in range(n) if (m >> i) & 1) for m in range(2 ** n)]
            mu = n * (n + 1) / 4
            brute = min(1.0, sum(abs(s - mu) >= abs(obs - mu) - 1e-9 for s in sums) / 2 ** n)
            res = wilcoxon(a, b)
            # brute force uses |W - mu| (symmetric distribution): identical to 2 * min tail
            assert abs(res["p"] - brute) < 1e-12, (n, res["p"], brute)
    # textbook check: n = 10, W+ = 8 -> two-sided p = 2 * 25 / 1024
    d = np.array([1, 2, 5, -3, -4, -6, -7, -8, -9, -10], float)
    res = wilcoxon(d, np.zeros(10))
    assert res["W_plus"] == 8 and abs(res["p"] - 50 / 1024) < 1e-12, res
    return "Wilcoxon exacto validado: coincide con la enumeracion completa en n = 1..12 (240 casos) y con p = 50/1024 para n = 10, W+ = 8"


# ------------------------------------------------------------------ data
def load_runs(files, name_map, dedupe_fixed=True):
    """{method: {seed_label: rewards ordered by scenario}}, scenarios."""
    out, scenarios = {}, None
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        sc = d["arguments"]["scenarios"]
        scenarios = scenarios or sc
        assert sc == scenarios, f
        for e in d["episodes"]:
            m = name_map[e["policy"]]
            out.setdefault(m, {}).setdefault((Path(f).name, e["seed_label"]), {})[e["scenario_seed"]] = e["reward"]
    res = {}
    for m, seeds in out.items():
        arrays = {k: np.array([v[s] for s in scenarios]) for k, v in seeds.items()}
        if m in ("fixed", "rule") and dedupe_fixed:  # the same deterministic policy in several files: must agree
            vals = list(arrays.values())
            assert all(np.array_equal(vals[0], v) for v in vals), f"{m} differs between files"
            arrays = {next(iter(arrays)): vals[0]}
        res[m] = arrays
    return res, scenarios


def stats(runs, a, b):
    A, Bm = np.stack(list(runs[a].values())), np.stack(list(runs[b].values()))  # (seeds, scenarios)
    sa, sb = A.mean(1), Bm.mean(1)
    pa, pb = A.mean(0), Bm.mean(0)
    wins = sum(int(x > y) for j in range(A.shape[1]) for x, y in itertools.product(A[:, j], Bm[:, j]))
    return {"n_seeds": [len(sa), len(sb)], "mean_diff": float(A.mean() - Bm.mean()),
            "seed_level_welch": welch(sa, sb), "paired_t_by_scenario": paired_t(pa, pb),
            "wilcoxon_by_scenario": wilcoxon(pa, pb), "wins": [wins, A.size * Bm.shape[0]]}


def describe(runs, m):
    X = np.stack(list(runs[m].values()))
    seed_means = X.mean(1)
    return {"n_seeds": X.shape[0], "mean": float(X.mean()), "median": float(np.median(X)),
            "catastrophic": int((X < CATASTROPHIC_THRESHOLD).sum()), "episodes": int(X.size), "worst": float(X.min()),
            "seed_means": seed_means.tolist(), "sd_seed_means": float(seed_means.std(ddof=1)) if X.shape[0] > 1 else None}


def same_at_precision(x, published):
    """x rounded to the significant digits printed in `published` equals it ("0.020" has 2, "4.5e-9" has 2)."""
    mant = published.split("e")[0].replace("-", "")
    digits = mant.replace(".", "").lstrip("0") if "e" not in published else mant.replace(".", "")
    return float(f"{x:.{max(1, len(digits))}g}") == float(published)


def fmt_p(p):
    return "n/a" if p is None else (f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-only", action="store_true")
    args = ap.parse_args()
    out = {"wilcoxon_validation": validate_wilcoxon()}
    print(out["wilcoxon_validation"])

    before = {s: load_runs([RES / f for f in PUBLISHED_FILES[s]], PUB_NAME) for s in SETS}
    print("\n==================== Control 4: los episodios publicados reproducen lo publicado ====================")
    ok4 = True
    for s in SETS:
        runs, _ = before[s]
        for m, pub in PUB_MEANS[s].items():
            v = describe(runs, m)["mean"]
            good = abs(round(v, 2) - pub) < 0.006
            ok4 &= good
            print(f"{s:<8} {m:<6} media {v:.2f} (publicada {pub:.2f}) {'OK' if good else 'DIFERENTE'}")
        for m in ("d10k", "d30k"):
            st = stats(runs, "dream", m)
            ps, pp = st["seed_level_welch"]["p"], st["paired_t_by_scenario"]["p"]
            pub_s, pub_p = PUB_P[(s, m)]
            good, good_p = same_at_precision(ps, pub_s), same_at_precision(pp, pub_p)
            ok4 &= good and good_p
            print(f"{s:<8} dream vs {m}: semilla p = {ps:.4g} (publicado {pub_s}) {'OK' if good else 'DIFERENTE'}; "
                  f"pareado p = {pp:.4g} (publicado {pub_p}) {'OK' if good_p else 'DIFERENTE'}")
    print(f"[control 4] {'OK' if ok4 else 'FALLA'}")
    out["control4_ok"] = ok4
    if args.before_only:
        for s in SETS:
            for m in ("d10k", "d30k"):
                w = stats(before[s][0], "dream", m)["wilcoxon_by_scenario"]
                print(f"antes, {s} dream vs {m}: Wilcoxon W+ = {w['W_plus']:.0f}, n = {w['n']}, p = {fmt_p(w['p'])} ({w['method']})")
        return
    if not ok4:
        sys.exit("control 4 falla - se detiene")

    after = {s: load_runs([NEW_FILES[s]], NEW_NAME) for s in SETS}
    print("\n==================== Control 3: los checkpoints existentes reproducen sus episodios publicados ====================")
    ok3 = True
    for s in SETS:
        b_runs, b_sc = before[s]
        a_runs, a_sc = after[s]
        assert a_sc == b_sc
        for m in ("dream", "d10k", "d30k", "fixed", "rule"):
            existing = {k: v for k, v in a_runs[m].items() if "_10seeds" not in k[1]}
            pub = list(b_runs[m].values())
            matched = sum(any(np.array_equal(v, p) for p in pub) for v in existing.values())
            good = matched == len(existing) == len(pub)
            ok3 &= good
            print(f"{s:<8} {m:<6} {matched}/{len(existing)} checkpoints existentes identicos a un registro publicado "
                  f"({len(pub)} publicados) {'OK' if good else 'DIFERENTE'}")
        for m, n in (("dream", 10), ("d10k", 10), ("d30k", 10)):
            assert len(a_runs[m]) == n, (s, m, len(a_runs[m]))
    print(f"[control 3] {'OK' if ok3 else 'FALLA'}")
    out["control3_ok"] = ok3
    if not ok3:
        sys.exit("control 3 falla - se detiene")

    print("\n==================== Tabla 3 con 10 semillas ====================")
    table = {s: {m: describe(after[s][0], m) for m in ("dream", "d10k", "d30k", "fixed", "rule")} for s in SETS}
    out["table"] = table
    for s in SETS:
        print(SETS[s])
        for m in ("dream", "d10k", "d30k", "fixed", "rule"):
            t = table[s][m]
            print(f"  {LABEL[m]:<28} semillas {t['n_seeds']:>2}  media {t['mean']:8.2f}  mediana {t['median']:8.2f}  "
                  f"< -600: {t['catastrophic']}/{t['episodes']}  peor {t['worst']:9.1f}  sd medias por semilla "
                  f"{t['sd_seed_means'] if t['sd_seed_means'] is None else round(t['sd_seed_means'], 1)}")
            if t["n_seeds"] > 1:
                print(f"     medias por semilla: {' / '.join(f'{x:.2f}' for x in t['seed_means'])}")

    print("\n==================== Comparaciones: antes (3 / 4 semillas) y despues (10 / 10) ====================")
    out["comparisons"] = {}
    for s in SETS:
        for m in ("d10k", "d30k", "fixed"):
            b = stats(before[s][0], "dream", m)
            a = stats(after[s][0], "dream", m)
            key = f"{s}:dream-vs-{m}"
            change = {}
            for test in ("paired_t_by_scenario", "wilcoxon_by_scenario"):
                pb, pa = b[test]["p"], a[test]["p"]
                change[test] = (pb < 0.05) != (pa < 0.05)
            out["comparisons"][key] = {"before": b, "after": a, "crosses_0.05": change}
            sb = "n/a" if m == "fixed" else fmt_p(b["seed_level_welch"]["p"])
            print(f"{s:<8} sueno vs {m:<5} diff {b['mean_diff']:+7.2f} -> {a['mean_diff']:+7.2f} | "
                  f"semilla Welch p {sb} -> {fmt_p(a['seed_level_welch']['p'])} | "
                  f"t pareada p {fmt_p(b['paired_t_by_scenario']['p'])} -> {fmt_p(a['paired_t_by_scenario']['p'])} | "
                  f"Wilcoxon p {fmt_p(b['wilcoxon_by_scenario']['p'])} -> {fmt_p(a['wilcoxon_by_scenario']['p'])} "
                  f"({a['wilcoxon_by_scenario']['method']}) | gana {a['wins'][0]}/{a['wins'][1]}"
                  + ("  <- CRUZA 0.05" if any(change.values()) and m != "fixed" else ""))
    (HERE / "analysis.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
