"""ADDENDUM_transformer_tsmixer.md analysis: per-architecture 5x5 grid at latent_dim=16 and the
between-architecture comparison. Usage: grid_arch.py lstm transformer [tsmixer]"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import sweep
from grid5 import anova2

HERE, W = Path(__file__).parent, sweep.WORK
H = list(range(1, 11))
AE = {a: (W / "ae16" if a == 0 else W / f"ae16_seed{a}") for a in range(5)}
LAYOUT = {  # arch -> (latent dir for AE a, raw dir)
    "lstm": (lambda a: W / ("z16" if a == 0 else f"z16_ae{a}"), W / "raw24"),
    "transformer": (lambda a: W / f"tf16_ae{a}", W / "tf_raw24"),
    "tsmixer": (lambda a: W / f"ts16_ae{a}", W / "ts_raw24"),
}
B, SEED = 100_000, 12345


def mse_curve(ckpt, test):
    agg, _ = sweep.cmp0.evaluate(ckpt, test, torch.device("cpu"))
    return [agg[h]["model_reward_mse"] for h in H]


def evaluate_arch(arch):
    cache = HERE / f"gridarch_{arch}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    zdir, rdir = LAYOUT[arch]
    z = [[mse_curve(zdir(a) / f"seed{s}" / "world_model_best.pt", AE[a] / "test_latent.npz") for s in range(5)] for a in range(5)]
    raw = [mse_curve(rdir / f"seed{s}" / "world_model_best.pt", W / "data/test_raw_seq.npz") for s in range(10)]
    out = {"z_reward_mse": z, "raw_reward_mse": raw}
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def boot_delta(G, raw, rng, ai=None):
    ai = rng.integers(0, 5, size=(B, 5)) if ai is None else ai
    si = rng.integers(0, 5, size=(B, 5, 5))
    zb = G[ai[:, :, None], si].mean(axis=(1, 2))
    rb = raw[rng.integers(0, len(raw), size=(B, len(raw)))].mean(axis=1)
    return zb - rb


pct = lambda d: 100 * (math.exp(d) - 1)
archs = sys.argv[1:]
res = {}
for arch in archs:
    e = evaluate_arch(arch)
    Z, R = np.array(e["z_reward_mse"]), np.array(e["raw_reward_mse"])  # (5,5,10), (10,10)
    G, raw = np.log(Z).mean(-1), np.log(R).mean(-1)
    red = np.median((R[None, :5, :] - Z) / R[None, :5, :], axis=-1)  # descriptive, vs raw seed s
    a2 = anova2(G)
    delta = G.mean() - raw.mean()
    db = boot_delta(G, raw, np.random.default_rng(SEED))
    lo, hi = np.percentile(db, [2.5, 97.5])
    x = np.concatenate([G.ravel(), raw]); rng2 = np.random.default_rng(SEED); cnt = 0
    for _ in range(20):
        perm = np.argsort(rng2.random((50_000, 35)), axis=1)
        cnt += int((np.abs(x[perm[:, :25]].mean(1) - x[perm[:, 25:]].mean(1)) >= abs(delta) - 1e-12).sum())
    verdict = "EVIDENCIA: el latente predice MEJOR" if hi < 0 else "EVIDENCIA: el latente predice PEOR" if lo > 0 else "SIN EVIDENCIA (el IC incluye 0)"
    print(f"\n==================== {arch.upper()} (latent_dim=16, estado de 24 dims) ====================")
    print("reduccion mediana (crudo_s - z)/crudo_s, % (descriptiva)")
    for a in range(5):
        print(f"  AE s{a} " + " ".join(f"{100 * v:+8.1f}" for v in red[a]) + f"   | media {100 * red[a].mean():+.1f}")
    print(f"  celdas positivas: {(red > 0).sum()}/25")
    print("media geometrica del reward_mse (exp g); crudo al final")
    for a in range(5):
        print(f"  AE s{a} " + " ".join(f"{math.exp(v):8.1f}" for v in G[a]) + f"   | AE {math.exp(G[a].mean()):.1f}")
    print(f"  crudo (10): " + " ".join(f"{math.exp(v):.1f}" for v in raw) + f"   | {math.exp(raw.mean()):.1f}")
    print(f"ANOVA g: AE F={a2['F_ae']:.2f} p={a2['p_ae']:.4f} sigma={math.sqrt(a2['sigma2_ae']):.3f} | "
          f"semilla F={a2['F_lstm']:.2f} p={a2['p_lstm']:.4f} sigma={math.sqrt(a2['sigma2_lstm']):.3f} | resid sigma={math.sqrt(a2['ms_resid']):.3f}")
    print(f"Delta {pct(delta):+.1f}%  IC95 conglomerados [{pct(lo):+.1f}%, {pct(hi):+.1f}%]  permutacion por corrida p={(cnt + 1) / 1_000_001:.4f}")
    print(f"CONCLUSION (criterio fijado): {verdict}")
    stops = []
    zdir, rdir = LAYOUT[arch]
    for d in [zdir(a) / f"seed{s}" for a in range(5) for s in range(5)] + [rdir / f"seed{s}" for s in range(10)]:
        f = d / ("lstm_summary.json" if arch == "lstm" else "train_summary.json")
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8")); stops.append((j["last_epoch"], j["stopped_early"]))
    print(f"epocas de corte: {min(s[0] for s in stops)}-{max(s[0] for s in stops)}; llegaron al tope sin early stopping: "
          f"{sum(not s[1] for s in stops)}/{len(stops)} (resumenes disponibles)")
    res[arch] = {"G": G.tolist(), "raw_g": raw.tolist(), "reduction": red.tolist(), "anova": a2, "delta": delta,
                 "ci95": [lo, hi], "verdict": verdict, "_G": G, "_raw": raw}

if len(archs) > 1:
    print("\n==================== ENTRE ARQUITECTURAS: Delta_A - Delta_B (bootstrap conjunto) ====================")
    res["pairs"] = {}
    for A, Bn in itertools.combinations(archs, 2):
        rng = np.random.default_rng(SEED)
        ai = rng.integers(0, 5, size=(B, 5))
        d = boot_delta(res[A]["_G"], res[A]["_raw"], rng, ai) - boot_delta(res[Bn]["_G"], res[Bn]["_raw"], rng, ai)
        lo, hi = np.percentile(d, [2.5, 97.5]); point = res[A]["delta"] - res[Bn]["delta"]
        v = "SIN EVIDENCIA de diferencia" if lo <= 0 <= hi else "EVIDENCIA de diferencia"
        print(f"{A} - {Bn}: {point:+.4f} (x100 ~ puntos %)  IC95 [{lo:+.4f}, {hi:+.4f}] -> {v}")
        res["pairs"][f"{A}-{Bn}"] = {"point": point, "ci95": [lo, hi], "verdict": v}
    print("\nTABLA: ventaja del Autoencoder (latent_dim=16) = exp(Delta)-1 de reward_mse geometrico (negativo = z mejor)")
    for arch in archs:
        r = res[arch]; print(f"  {arch:<12} {pct(r['delta']):+6.1f}%  IC95 [{pct(r['ci95'][0]):+.1f}%, {pct(r['ci95'][1]):+.1f}%]  {r['verdict']}")
for arch in archs:
    res[arch].pop("_G"); res[arch].pop("_raw")
(HERE / f"grid_arch_{'_'.join(archs)}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
