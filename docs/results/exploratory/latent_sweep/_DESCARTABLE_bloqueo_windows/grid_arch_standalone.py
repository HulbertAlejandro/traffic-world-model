"""
############################################################################################
#  DESCARTABLE - NO ES PARTE DEL PIPELINE (ni oficial ni exploratorio).  2026-09-29
#
#  Las funciones de evaluacion de este archivo estan COPIADAS, no importadas, unicamente
#  porque en este equipo un bloqueo de Windows (Smart App Control / Control de aplicaciones,
#  "Una directiva de Control de aplicaciones bloqueo este archivo") impide cargar la DLL de
#  kiwisolver, y por tanto matplotlib. Los modulos oficiales importan matplotlib a nivel de
#  modulo (evaluation/__init__.py -> autoencoder_evaluation.py, y world_model_evaluation.py),
#  asi que no se pueden importar aqui aunque el calculo no dibuje nada.
#
#  Fuente original (la unica de verdad; este archivo no la sustituye):
#    - evaluation/world_model_evaluation.py: build_world_model, load_world_model,
#      load_reward_scaler, load_episodes, predict_next_step, rollout_episode,
#      aggregate_by_horizon
#    - scripts/compare_experiment_0_multiseed.py: evaluate()
#    - docs/results/exploratory/latent_sweep/grid5.py: betainc, f_sf, anova2
#    - docs/results/exploratory/latent_sweep/grid_arch.py: el resto del analisis
#
#  Copias literales salvo los imports. Verificacion exigida antes de usarlo (--verify):
#  reproducir EXACTAMENTE (igualdad de floats) gridarch_lstm.json, gridarch_transformer.json
#  y grid_arch_lstm_transformer.json. Con el bloqueo resuelto, usar grid_arch.py.
############################################################################################

Usage:
    grid_arch_standalone.py --verify
    grid_arch_standalone.py lstm transformer tsmixer
"""
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LATENT_SWEEP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from models.world_model import LatentDynamicsLSTM, LatentDynamicsTransformer, LatentDynamicsTSMixer  # noqa: E402


# ---------------- copia de evaluation/world_model_evaluation.py ----------------
def build_world_model(hparams: dict) -> nn.Module:
    architecture = hparams.get("architecture", "lstm")
    if architecture == "lstm":
        return LatentDynamicsLSTM(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            hidden_dim=hparams["hidden_dim"],
            sequence_length=hparams["sequence_length"],
        )
    if architecture == "transformer":
        return LatentDynamicsTransformer(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            num_layers=hparams["num_layers"],
            dim_feedforward=hparams["dim_feedforward"],
            sequence_length=hparams["sequence_length"],
            dropout=hparams["dropout"],
        )
    if architecture == "tsmixer":
        return LatentDynamicsTSMixer(
            latent_dim=hparams["latent_dim"],
            action_dim=hparams["action_dim"],
            hidden_dim=hparams["hidden_dim"],
            num_blocks=hparams["num_blocks"],
            sequence_length=hparams["sequence_length"],
            dropout=hparams["dropout"],
        )
    raise ValueError(
        f"Unknown world model architecture {architecture!r}; "
        "expected 'lstm', 'transformer' or 'tsmixer'."
    )


def load_world_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    hparams_path = checkpoint_path.with_suffix(".json")
    if not hparams_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file for checkpoint: {hparams_path}")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    model = build_world_model(hparams).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def load_reward_scaler(checkpoint_path):
    scaler_path = Path(checkpoint_path).parent / "reward_scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing reward scaler file: {scaler_path}. Run training/train_world_model.py first.")
    return json.loads(scaler_path.read_text(encoding="utf-8"))


def load_episodes(path):
    with np.load(Path(path)) as data:
        z = data["z"].astype(np.float32)
        actions = data["actions"].astype(np.int64)
        rewards = data["rewards"].astype(np.float32)
        episode_id = data["episode_id"].astype(np.int64)
        time_step = data["time_step"].astype(np.int64)
    episodes = {}
    for ep_id in np.unique(episode_id):
        mask = episode_id == ep_id
        order = np.argsort(time_step[mask])
        episodes[int(ep_id)] = {"z": z[mask][order], "actions": actions[mask][order], "rewards": rewards[mask][order]}
    return episodes


@torch.no_grad()
def predict_next_step(model, window_z, window_actions_onehot, device, reward_mean=0.0, reward_std=1.0):
    pred_z, pred_r = model(window_z.unsqueeze(0).to(device), window_actions_onehot.unsqueeze(0).to(device))
    pred_z = pred_z.squeeze(0).cpu()
    pred_r = pred_r.squeeze(0).cpu()
    pred_r = pred_r * reward_std + reward_mean
    return pred_z, pred_r


@torch.no_grad()
def rollout_episode(model, episode, sequence_length, action_dim, max_horizon, device, reward_mean=0.0, reward_std=1.0):
    z = torch.from_numpy(episode["z"])
    actions = torch.from_numpy(episode["actions"])
    rewards = torch.from_numpy(episode["rewards"])
    actions_onehot = F.one_hot(actions, num_classes=action_dim).float()
    num_steps = z.shape[0]
    last_start = num_steps - sequence_length - max_horizon
    if last_start < 0:
        return []
    records = []
    for start in range(last_start + 1):
        window_z = z[start : start + sequence_length].clone()
        baseline_z = z[start + sequence_length - 1].clone()
        baseline_reward_idx = max(start + sequence_length - 2, 0)
        baseline_reward = rewards[baseline_reward_idx].clone()
        for h in range(1, max_horizon + 1):
            action_window = actions_onehot[start + h - 1 : start + h - 1 + sequence_length]
            pred_z, pred_r = predict_next_step(model, window_z, action_window, device, reward_mean, reward_std)
            true_idx = start + sequence_length + h - 1
            true_z = z[true_idx]
            true_reward = rewards[true_idx - 1]
            records.append(
                {
                    "horizon": h,
                    "model_latent_se": torch.sum((pred_z - true_z) ** 2).item(),
                    "model_reward_se": ((pred_r - true_reward) ** 2).item(),
                    "baseline_latent_se": torch.sum((baseline_z - true_z) ** 2).item(),
                    "baseline_reward_se": ((baseline_reward - true_reward) ** 2).item(),
                }
            )
            window_z = torch.cat([window_z[1:], pred_z.unsqueeze(0)], dim=0)
    return records


def aggregate_by_horizon(records, latent_dim):
    buckets = defaultdict(lambda: {"model_latent": [], "model_reward": [], "baseline_latent": [], "baseline_reward": []})
    for record in records:
        bucket = buckets[record["horizon"]]
        bucket["model_latent"].append(record["model_latent_se"] / latent_dim)
        bucket["model_reward"].append(record["model_reward_se"])
        bucket["baseline_latent"].append(record["baseline_latent_se"] / latent_dim)
        bucket["baseline_reward"].append(record["baseline_reward_se"])
    summary = {}
    for horizon in sorted(buckets):
        bucket = buckets[horizon]
        summary[horizon] = {
            "model_latent_mse": float(np.mean(bucket["model_latent"])),
            "model_reward_mse": float(np.mean(bucket["model_reward"])),
            "baseline_latent_mse": float(np.mean(bucket["baseline_latent"])),
            "baseline_reward_mse": float(np.mean(bucket["baseline_reward"])),
            "n_samples": len(bucket["model_latent"]),
        }
    return summary


# ---------------- copia de scripts/compare_experiment_0_multiseed.py ----------------
MAX_HORIZON = 10


def evaluate(checkpoint, test_path, device):
    model, hparams = load_world_model(checkpoint, device)
    scaler = load_reward_scaler(checkpoint)
    episodes = load_episodes(test_path)
    records = []
    per_episode = {}
    for episode_id, episode in episodes.items():
        episode_records = rollout_episode(
            model, episode, hparams["sequence_length"], hparams["action_dim"], MAX_HORIZON, device,
            reward_mean=scaler["reward_mean"], reward_std=scaler["reward_std"],
        )
        records.extend(episode_records)
        per_episode[str(episode_id)] = float(
            np.mean([r["model_reward_se"] for r in episode_records if r["horizon"] == MAX_HORIZON])
        )
    return aggregate_by_horizon(records, hparams["latent_dim"]), per_episode


# ---------------- copia de grid5.py (betainc, f_sf, anova2) ----------------
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


# ---------------- copia de grid_arch.py (solo cambia de donde vienen evaluate/anova2) ----------------
HERE, W = LATENT_SWEEP, ROOT / "models" / "checkpoints" / "exploratory_latent_sweep"
H = list(range(1, 11))
AE = {a: (W / "ae16" if a == 0 else W / f"ae16_seed{a}") for a in range(5)}
LAYOUT = {  # arch -> (latent dir for AE a, raw dir)
    "lstm": (lambda a: W / ("z16" if a == 0 else f"z16_ae{a}"), W / "raw24"),
    "transformer": (lambda a: W / f"tf16_ae{a}", W / "tf_raw24"),
    "tsmixer": (lambda a: W / f"ts16_ae{a}", W / "ts_raw24"),
}
B, SEED = 100_000, 12345


def mse_curve(ckpt, test):
    agg, _ = evaluate(ckpt, test, torch.device("cpu"))
    return [agg[h]["model_reward_mse"] for h in H]


def compute_arch(arch):
    zdir, rdir = LAYOUT[arch]
    z = [[mse_curve(zdir(a) / f"seed{s}" / "world_model_best.pt", AE[a] / "test_latent.npz") for s in range(5)] for a in range(5)]
    raw = [mse_curve(rdir / f"seed{s}" / "world_model_best.pt", W / "data/test_raw_seq.npz") for s in range(10)]
    return {"z_reward_mse": z, "raw_reward_mse": raw}


def evaluate_arch(arch):
    cache = HERE / f"gridarch_{arch}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = compute_arch(arch)
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def boot_delta(G, raw, rng, ai=None):
    ai = rng.integers(0, 5, size=(B, 5)) if ai is None else ai
    si = rng.integers(0, 5, size=(B, 5, 5))
    zb = G[ai[:, :, None], si].mean(axis=(1, 2))
    rb = raw[rng.integers(0, len(raw), size=(B, len(raw)))].mean(axis=1)
    return zb - rb


pct = lambda d: 100 * (math.exp(d) - 1)


def analyze(archs, evals):
    res = {}
    for arch in archs:
        e = evals[arch]
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
    return json.loads(json.dumps(res))  # same float round-trip as the file grid_arch.py writes


def verify():
    """Recompute LSTM and Transformer from the checkpoints (ignoring the caches) and require
    exact float equality with the files grid_arch.py wrote."""
    ok = True
    evals = {}
    for arch in ("lstm", "transformer"):
        fresh = json.loads(json.dumps(compute_arch(arch)))
        cached = json.loads((HERE / f"gridarch_{arch}.json").read_text(encoding="utf-8"))
        fz, cz = np.array(fresh["z_reward_mse"]), np.array(cached["z_reward_mse"])
        fr, cr = np.array(fresh["raw_reward_mse"]), np.array(cached["raw_reward_mse"])
        same = fresh == cached
        print(f"[verify] gridarch_{arch}.json: {'IDENTICO' if same else 'DIFERENTE'} "
              f"(z {fz.shape}, crudo {fr.shape}; max |dif| z={np.abs(fz - cz).max():.3e}, crudo={np.abs(fr - cr).max():.3e}; "
              f"celdas distintas z={int((fz != cz).sum())}, crudo={int((fr != cr).sum())})")
        ok &= same
        evals[arch] = fresh
    res = analyze(["lstm", "transformer"], evals)
    stored = json.loads((HERE / "grid_arch_lstm_transformer.json").read_text(encoding="utf-8"))
    same = res == stored
    print(f"[verify] grid_arch_lstm_transformer.json (analisis completo): {'IDENTICO' if same else 'DIFERENTE'}")
    ok &= same
    print(f"[verify] RESULTADO: {'REPRODUCCION EXACTA' if ok else 'HAY DIFERENCIAS - NO USAR'}")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify"]:
        sys.exit(0 if verify() else 1)
    archs = sys.argv[1:]
    res = analyze(archs, {a: evaluate_arch(a) for a in archs})
    (HERE / f"grid_arch_{'_'.join(archs)}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
