"""ADDENDUM_ae_seeds.md analysis: 3x3 (autoencoder seed x LSTM seed) variance decomposition."""
import json
from pathlib import Path

import numpy as np

import sweep

HERE, W = Path(__file__).parent, sweep.WORK
LSTM_DIRS = {0: W / "z12", 1: W / "z12_ae1", 2: W / "z12_ae2"}
AE_DIRS = {0: W / "ae12", 1: W / "ae12_seed1", 2: W / "ae12_seed2"}
H = [str(h) for h in range(1, 11)]

reports = {}
for a in (0, 1, 2):
    out = HERE / f"ae_seed{a}_z12_vs_raw24.json"
    sweep.compare([LSTM_DIRS[a] / f"seed{s}" for s in (0, 1, 2)], [W / f"raw24/seed{s}" for s in (0, 1, 2)],
                  AE_DIRS[a] / "test_latent.npz", W / "data/test_raw_seq.npz", out)
    reports[a] = json.loads(out.read_text(encoding="utf-8"))

R = np.array([[s["median_reduction"] for s in reports[a]["per_seed"]] for a in (0, 1, 2)])  # rows AE, cols LSTM
Z = np.array([[np.log([p["z"][h]["model_reward_mse"] for h in H]).mean() for p in reports[a]["pairs"]] for a in (0, 1, 2)])

# consistency: AE0 row must equal the seeds 0-2 of the 10-seed report
w = json.loads((HERE / "winner_z12_10seeds.json").read_text(encoding="utf-8"))
assert np.allclose(R[0], [s["median_reduction"] for s in w["per_seed"][:3]])


def decompose(M, label, scale=100.0, unit="%"):
    r, c = M.shape
    grand = M.mean()
    ms_rows = c * ((M.mean(1) - grand) ** 2).sum() / (r - 1)
    ms_cols = r * ((M.mean(0) - grand) ** 2).sum() / (c - 1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + grand
    ms_res = (resid ** 2).sum() / ((r - 1) * (c - 1))
    between_ae = M.var(axis=0, ddof=1)    # per LSTM column, across AEs
    between_lstm = M.var(axis=1, ddof=1)  # per AE row, across LSTM seeds
    s2_ae, s2_lstm = max(0.0, (ms_rows - ms_res) / c), max(0.0, (ms_cols - ms_res) / r)
    rm = M.mean(1)
    others = [(rm[1] + rm[2]) / 2, (rm[0] + rm[2]) / 2, (rm[0] + rm[1]) / 2]
    se = np.sqrt(ms_res / 2)
    atyp0 = bool((rm[0] == rm.max() or rm[0] == rm.min()) and abs(rm[0] - others[0]) > 2 * se)
    k = scale ** 2
    print(f"\n== {label} ==")
    print("      " + "".join(f"LSTM s{s:<8}" for s in range(c)) + "media fila")
    for a in range(r):
        print(f"AE s{a} " + "".join(f"{scale * M[a, s]:+9.1f}{unit:<3}" for s in range(c)) + f"{scale * rm[a]:+9.1f}{unit}")
    print("media col " + "".join(f"{scale * v:+9.1f}{unit:<3}" for v in M.mean(0)))
    print(f"entre AE (por columna)   var: {[round(k * v, 1) for v in between_ae]}  rango: {[round(scale * v, 1) for v in np.ptp(M, 0)]}  media var {k * between_ae.mean():.1f}")
    print(f"entre LSTM (por fila)    var: {[round(k * v, 1) for v in between_lstm]}  rango: {[round(scale * v, 1) for v in np.ptp(M, 1)]}  media var {k * between_lstm.mean():.1f}")
    print(f"ANOVA: CM_AE {k * ms_rows:.1f} (F={ms_rows / ms_res:.2f}), CM_LSTM {k * ms_cols:.1f} (F={ms_cols / ms_res:.2f}), CM_res {k * ms_res:.1f}")
    print(f"componentes: sigma2_AE {k * s2_ae:.1f} (sd {scale * np.sqrt(s2_ae):.1f}), sigma2_LSTM {k * s2_lstm:.1f} (sd {scale * np.sqrt(s2_lstm):.1f})")
    print(f"AE0 fila {scale * rm[0]:+.1f} vs media otras {scale * others[0]:+.1f}; umbral 2*SE = {2 * scale * se:.1f} -> atipico: {atyp0}")
    return {"matrix": M.tolist(), "row_means": rm.tolist(), "col_means": M.mean(0).tolist(),
            "between_ae_var_per_col": between_ae.tolist(), "between_lstm_var_per_row": between_lstm.tolist(),
            "ms_ae": ms_rows, "ms_lstm": ms_cols, "ms_resid": ms_res, "sigma2_ae": s2_ae, "sigma2_lstm": s2_lstm,
            "ae0_atypical": atyp0, "ae0_vs_others": [rm[0], others[0], 2 * se]}


res = {"primary_reduction": decompose(R, "PRINCIPAL: reduccion mediana (crudo-z)/crudo"),
       "secondary_log_geomean_z": decompose(Z, "SECUNDARIO: 100 x log media geometrica reward_mse de z (diferencias ~ %)", scale=100.0, unit="")}
ten = np.array([s["median_reduction"] for s in w["per_seed"]])
print(f"\nAE0, 10 semillas de LSTM: var {1e4 * ten.var(ddof=1):.1f} (sd {100 * ten.std(ddof=1):.1f} puntos), rango {100 * ten.min():+.1f}% a {100 * ten.max():+.1f}%")
print(f"celdas positivas: {(R > 0).sum()}/9")
for a in (0, 1, 2):
    s = json.loads((AE_DIRS[a] / "ae_summary.json").read_text(encoding="utf-8"))
    print(f"AE s{a}: reconstruccion val {s['best_validation_loss']:.4f} (mejor epoca {s['best_epoch']})")
res["ae0_10_lstm_seeds_var"] = float(ten.var(ddof=1))
(HERE / "ae_variance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
