"""ADDENDUM_arch_ranking.md analysis: LSTM vs Transformer vs TSMixer on z (latent_dim=16), 25 runs
each, blocked by Autoencoder. No training: reads gridarch_{arch}.json (z_reward_mse, 5 AE x 5 seeds x
10 horizons) and, for the side-by-side, the official Experimento 3 reports in results/. numpy only."""
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[3]
ARCHS = ("lstm", "transformer", "tsmixer")
OFFICIAL = {"lstm": "world_model_evaluation.json", "transformer": "world_model_transformer_evaluation.json",
            "tsmixer": "world_model_tsmixer_evaluation.json"}
PUBLISHED = {"transformer": 0.381, "tsmixer": 0.561}
B, SEED, CHUNK = 100_000, 12345, 10_000
Q_BONF3 = [100 * 0.05 / 3 / 2, 100 * (1 - 0.05 / 3 / 2)]
Q_BONF2 = [100 * 0.05 / 2 / 2, 100 * (1 - 0.05 / 2 / 2)]
Q95 = [2.5, 97.5]

Z = {a: np.array(json.loads((HERE / f"gridarch_{a}.json").read_text(encoding="utf-8"))["z_reward_mse"]) for a in ARCHS}
LOG = {a: np.log(z) for a, z in Z.items()}  # (5 AE, 5 seeds, 10 h)
for a in ARCHS:
    assert Z[a].shape == (5, 5, 10), (a, Z[a].shape)


def boot_horizon_means(A, Bn):
    """Paired-by-AE cluster bootstrap: same AE indices for both, seeds resampled independently.
    Returns per-replicate mean log reward_mse per horizon for A and for B, each (B, 10)."""
    rng = np.random.default_rng(SEED)
    ai = rng.integers(0, 5, size=(B, 5))
    si_a = rng.integers(0, 5, size=(B, 5, 5))
    si_b = rng.integers(0, 5, size=(B, 5, 5))
    ma, mb = np.empty((B, 10)), np.empty((B, 10))
    for s in range(0, B, CHUNK):
        e = slice(s, s + CHUNK)
        ma[e] = LOG[A][ai[e, :, None], si_a[e]].mean(axis=(1, 2))
        mb[e] = LOG[Bn][ai[e, :, None], si_b[e]].mean(axis=(1, 2))
    return ma, mb


pct = lambda d: 100 * (math.exp(d) - 1)
out = {"A_pairs": {}, "B_article_format": {}, "C_single_seed": {}}
boots = {}

print("==================== A. Pares (D = media g_A - media g_B; Bonferroni 3 pruebas, 98.33%) ====================")
for A, Bn in itertools.combinations(ARCHS, 2):
    ma, mb = boot_horizon_means(A, Bn)
    boots[(A, Bn)] = (ma, mb)
    point = LOG[A].mean() - LOG[Bn].mean()
    d = ma.mean(1) - mb.mean(1)
    lo, hi = np.percentile(d, Q_BONF3)
    lo95, hi95 = np.percentile(d, Q95)
    verdict = (f"EVIDENCIA: {A} predice MEJOR" if hi < 0 else f"EVIDENCIA: {Bn} predice MEJOR" if lo > 0
               else "SIN EVIDENCIA de diferencia")
    per_ae = [pct(LOG[A][a].mean() - LOG[Bn][a].mean()) for a in range(5)]
    print(f"{A} - {Bn}: D={point:+.4f} ({pct(point):+.1f}%)  IC98.33 [{pct(lo):+.1f}%, {pct(hi):+.1f}%]  "
          f"(IC95 [{pct(lo95):+.1f}%, {pct(hi95):+.1f}%]) -> {verdict}")
    print("   por Autoencoder, exp(D_a)-1: " + " ".join(f"{v:+.1f}%" for v in per_ae))
    out["A_pairs"][f"{A}-{Bn}"] = {"D": point, "ci_bonf3": [lo, hi], "ci95": [lo95, hi95], "verdict": verdict,
                                   "per_ae_pct": per_ae}
v = out["A_pairs"]
held = [v["lstm-transformer"]["ci_bonf3"][1] < 0, v["lstm-tsmixer"]["ci_bonf3"][1] < 0, v["transformer-tsmixer"]["ci_bonf3"][1] < 0]
out["ranking_sustained"] = all(held)
print(f"Orden LSTM < Transformer < TSMixer (error): LSTM<TF {held[0]}, LSTM<TS {held[1]}, TF<TS {held[2]} -> "
      f"{'SOSTENIDO CON EVIDENCIA' if all(held) else 'NO sostenido completo'}")

off = {a: json.loads((ROOT / "results" / f).read_text(encoding="utf-8")) for a, f in OFFICIAL.items()}
off = {a: [(d.get("per_horizon", d))[str(h)]["model_reward_mse"] for h in range(1, 11)] for a, d in off.items()}

for X in ("transformer", "tsmixer"):
    ma, mb = boots[("lstm", X)]
    r_point = 1 - np.exp(LOG["lstm"].mean(axis=(0, 1)) - LOG[X].mean(axis=(0, 1)))
    r_boot = 1 - np.exp(ma - mb)  # (B, 10)
    R_point, R_boot = float(np.median(r_point)), np.median(r_boot, axis=1)
    h_lo, h_hi = np.percentile(r_boot, Q95, axis=0)
    R_lo, R_hi = np.percentile(R_boot, Q_BONF2)
    r_off = [(off[X][h] - off["lstm"][h]) / off[X][h] for h in range(10)]
    print(f"\n==================== B. LSTM frente a {X.upper()}: reduccion r_h = 1 - GM_LSTM/GM_{X} ====================")
    print("  h |  GM LSTM |  GM X  |  r_h (25 corridas) | IC95 descriptivo   | oficial (1 corrida)")
    gm_l, gm_x = np.exp(LOG["lstm"].mean(axis=(0, 1))), np.exp(LOG[X].mean(axis=(0, 1)))
    for h in range(10):
        print(f" {h + 1:2d} | {gm_l[h]:8.1f} | {gm_x[h]:6.1f} | {100 * r_point[h]:+17.1f}% | "
              f"[{100 * h_lo[h]:+6.1f}%, {100 * h_hi[h]:+6.1f}%] | {100 * r_off[h]:+6.1f}%")
    n_ci = int((h_lo > 0).sum())
    pub = PUBLISHED[X]
    status = "COMPATIBLE" if R_lo <= pub <= R_hi else "FUERA POR ARRIBA (sobreestima)" if pub > R_hi else "FUERA POR ABAJO (subestima)"
    print(f"  R = mediana de r_h: {100 * R_point:+.1f}%  IC97.5 (Bonferroni 2) [{100 * R_lo:+.1f}%, {100 * R_hi:+.1f}%]"
          f"  | publicado {100 * pub:.1f}% -> {status}")
    print(f"  horizontes con IC95 a favor de la LSTM: {n_ci}/10 (publicado: 10/10)")

    # C. single-run pairs within each AE
    meds, wins = [], []
    for a in range(5):
        for i in range(5):
            for j in range(5):
                l, x = Z["lstm"][a, i], Z[X][a, j]
                meds.append(float(np.median((x - l) / x))); wins.append(int((l < x).sum()))
    meds, wins = np.array(meds), np.array(wins)
    pc = np.percentile(meds, [5, 25, 50, 75, 95])
    rank_pub = float((meds <= pub).mean())
    print(f"  C. una corrida de cada lado (125 pares dentro de Autoencoder): mediana {100 * pc[2]:+.1f}%, "
          f"P5 {100 * pc[0]:+.1f}%, P25 {100 * pc[1]:+.1f}%, P75 {100 * pc[3]:+.1f}%, P95 {100 * pc[4]:+.1f}%")
    print(f"     el {100 * pub:.1f}% publicado cae en el percentil {100 * rank_pub:.1f}; "
          f"pares con la LSTM ganando 10/10: {(wins == 10).mean() * 100:.1f}%; "
          f"LSTM gana mayoria (>=6/10): {(wins >= 6).mean() * 100:.1f}%; reduccion mediana < 0: {(meds < 0).mean() * 100:.1f}%")
    out["B_article_format"][X] = {"r_h": r_point.tolist(), "r_h_ci95": [h_lo.tolist(), h_hi.tolist()],
                                  "gm_lstm": gm_l.tolist(), "gm_x": gm_x.tolist(), "R": R_point,
                                  "R_ci_bonf2": [R_lo, R_hi], "published": pub, "published_status": status,
                                  "horizons_ci_favor_lstm": n_ci, "official_r_h": r_off}
    out["C_single_seed"][X] = {"percentiles_5_25_50_75_95": pc.tolist(), "published_percentile": rank_pub,
                               "frac_lstm_wins_10of10": float((wins == 10).mean()),
                               "frac_lstm_wins_majority": float((wins >= 6).mean()),
                               "frac_negative": float((meds < 0).mean())}
(HERE / "arch_ranking.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
