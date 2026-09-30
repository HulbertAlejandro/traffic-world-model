"""ADDENDUM_grid5.md analysis: 5x5 (AE seed x LSTM seed) grid for latent_dim=12 vs raw-24."""
import json
import math
from pathlib import Path

import numpy as np

import sweep

HERE, W = Path(__file__).parent, sweep.WORK
H = [str(h) for h in range(1, 11)]
AE_DIRS = {0: W / "ae12", **{a: W / f"ae12_seed{a}" for a in (1, 2, 3, 4)}}
LSTM_DIRS = {0: W / "z12", **{a: W / f"z12_ae{a}" for a in (1, 2, 3, 4)}}
SEEDS = (0, 1, 2, 3, 4)


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), Numerical Recipes continued fraction."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - betainc(b, a, 1 - x)
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / d; f = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d; d = 1.0 / d if abs(d) > 1e-300 else 1e300
            c = 1.0 + num / c
            f *= d * c
    return math.exp(lbeta) * f / a


def f_sf(F, d1, d2):
    """P(F_{d1,d2} > F)."""
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * F))


def anova2(M):
    r, c = M.shape
    g = M.mean()
    ms_r = c * ((M.mean(1) - g) ** 2).sum() / (r - 1)
    ms_c = r * ((M.mean(0) - g) ** 2).sum() / (c - 1)
    res = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g
    df_res = (r - 1) * (c - 1)
    ms_e = (res ** 2).sum() / df_res
    return {"ms_ae": ms_r, "ms_lstm": ms_c, "ms_resid": ms_e,
            "F_ae": ms_r / ms_e, "p_ae": f_sf(ms_r / ms_e, r - 1, df_res),
            "F_lstm": ms_c / ms_e, "p_lstm": f_sf(ms_c / ms_e, c - 1, df_res),
            "sigma2_ae": max(0.0, (ms_r - ms_e) / c), "sigma2_lstm": max(0.0, (ms_c - ms_e) / r)}


def logg(p, branch):
    return float(np.mean(np.log([p[branch][h]["model_reward_mse"] for h in H])))


if __name__ == "__main__":
    reports = {}
    for a in (0, 1, 2, 3, 4):
        out = HERE / f"grid5_ae{a}_vs_raw24.json"
        sweep.compare([LSTM_DIRS[a] / f"seed{s}" for s in SEEDS], [W / f"raw24/seed{s}" for s in SEEDS],
                      AE_DIRS[a] / "test_latent.npz", W / "data/test_raw_seq.npz", out)
        reports[a] = json.loads(out.read_text(encoding="utf-8"))
    R = np.array([[s["median_reduction"] for s in reports[a]["per_seed"]] for a in range(5)])
    G = np.array([[logg(p, "z") for p in reports[a]["pairs"]] for a in range(5)])
    # consistency with earlier reports
    w = json.loads((HERE / "winner_z12_10seeds.json").read_text(encoding="utf-8"))
    assert np.allclose(R[0], [s["median_reduction"] for s in w["per_seed"][:5]])
    raw = np.array([logg(p, "raw") for p in w["pairs"]])  # raw24 seeds 0-9

    def show(M, label, fmt):
        print(f"\n{label}\n        " + "".join(f"LSTM s{s:<6}" for s in SEEDS) + "  media fila")
        for a in range(5):
            print(f"AE s{a}   " + "".join(fmt(v) for v in M[a]) + "  " + fmt(M[a].mean()))
        print("media   " + "".join(fmt(v) for v in M.mean(0)))

    show(100 * R, "PRINCIPAL (descriptiva): reduccion mediana (crudo-z)/crudo, %", lambda v: f"{v:+11.1f} ")
    show(np.exp(G), "SECUNDARIA: media geometrica del reward_mse de z (exp(g))", lambda v: f"{v:11.1f} ")
    out = {"R": R.tolist(), "G": G.tolist(), "raw24_g": raw.tolist()}
    for name, M, k in (("secondary_g", G, 1.0), ("primary_reduction", R, 1.0)):
        a2 = anova2(M); out[f"anova_{name}"] = a2
        sd = lambda s2: math.sqrt(s2)
        print(f"\nANOVA {name}: CM_AE {a2['ms_ae']:.5f} F={a2['F_ae']:.2f} p={a2['p_ae']:.4f} | CM_LSTM {a2['ms_lstm']:.5f} "
              f"F={a2['F_lstm']:.2f} p={a2['p_lstm']:.4f} | CM_res {a2['ms_resid']:.5f}")
        print(f"   sigma_AE {sd(a2['sigma2_ae']):.4f}  sigma_LSTM {sd(a2['sigma2_lstm']):.4f}  sigma_res {sd(a2['ms_resid']):.4f}"
              + ("   (en g: ~ desviacion relativa)" if name == "secondary_g" else "   (en fraccion de reduccion)"))

    # decision: clustered bootstrap
    delta = G.mean() - raw.mean()
    rng = np.random.default_rng(12345)
    B = 100_000
    ai = rng.integers(0, 5, size=(B, 5))
    si = rng.integers(0, 5, size=(B, 5, 5))
    zb = G[ai[:, :, None], si].mean(axis=(1, 2))
    rb = raw[rng.integers(0, 10, size=(B, 10))].mean(axis=1)
    db = zb - rb
    lo, hi = np.percentile(db, [2.5, 97.5])
    # reported, not decisive: run-level permutation (Monte Carlo)
    x = np.concatenate([G.ravel(), raw])
    rng2 = np.random.default_rng(12345)
    cnt, N, chunk = 0, 1_000_000, 50_000
    for _ in range(N // chunk):
        perm = np.argsort(rng2.random((chunk, 35)), axis=1)
        d = x[perm[:, :25]].mean(1) - x[perm[:, 25:]].mean(1)
        cnt += int((np.abs(d) >= abs(delta) - 1e-12).sum())
    p_perm = (cnt + 1) / (N + 1)
    verdict = ("EVIDENCIA de que z12 predice MEJOR" if hi < 0 else "EVIDENCIA de que z12 predice PEOR" if lo > 0
               else "SIN EVIDENCIA (el IC incluye 0)")
    print(f"\nDelta = media g(z, 25) - media g(crudo24, 10) = {delta:+.4f}  ->  {100 * (math.exp(delta) - 1):+.1f}% de reward_mse geometrico")
    print(f"IC 95% bootstrap por conglomerados (B={B}): [{lo:+.4f}, {hi:+.4f}]  ->  [{100 * (math.exp(lo) - 1):+.1f}%, {100 * (math.exp(hi) - 1):+.1f}%]")
    print(f"Permutacion por corrida (no decisoria, anticonservadora): p = {p_perm:.4f}")
    print(f"CONCLUSION (criterio fijado): {verdict}")
    print(f"media geometrica: z (25) {math.exp(G.mean()):.1f}, crudo24 (10) {math.exp(raw.mean()):.1f}; por AE: "
          + ", ".join(f"AE{a} {math.exp(G[a].mean()):.1f}" for a in range(5)))
    print(f"celdas con reduccion positiva: {(R > 0).sum()}/25")
    for a in range(5):
        s = json.loads((AE_DIRS[a] / "ae_summary.json").read_text(encoding="utf-8"))
        print(f"AE s{a}: reconstruccion val {s['best_validation_loss']:.4f}")
    out.update({"delta": delta, "ci95": [lo, hi], "p_perm_run_level": p_perm, "verdict": verdict})
    (HERE / "grid5.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
