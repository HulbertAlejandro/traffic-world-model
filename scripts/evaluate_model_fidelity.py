"""
Genera la Tabla 4b del artículo: fidelidad del retorno imaginado, para LSTM, Transformer y
TSMixer. No modifica nada oficial; solo lee checkpoints y el dataset de test ya existentes.

Ejecutar desde la raíz del repositorio:
    python scripts/evaluate_model_fidelity.py --horizon 7

Retorno imaginado vs. real: para cada episodio de test y cada punto de inicio válido, se
siembra el modelo con una ventana real de sequence_length pasos y se avanza H pasos de forma
autorregresiva CON LAS ACCIONES REGISTRADAS (no con una política). Se compara el retorno
acumulado imaginado contra el retorno real del mismo tramo, con la correlación de Pearson y
el sesgo medio sobre todos los tramos de todos los episodios.

El bucle replica exactamente el de evaluation/world_model_evaluation.py::rollout_episode
(Experimento 1): misma ventana de acciones, misma recompensa objetivo de cada horizonte y
la misma predicción (predict_next_step, importada). Un control lo verifica antes de reportar:
el error cuadrático de cada paso debe coincidir exactamente con el model_reward_se de
rollout_episode.

Los tramos de un mismo episodio se solapan, así que no son independientes. El p-valor de
Pearson (t con n − 2 grados de libertad) los trata como independientes y es optimista. Por
eso se reporta también un IC del 95% por bootstrap de episodios (B = 10,000,
default_rng(12345)).

Referencia: el baseline persistente de rollout_episode (la última recompensa conocida antes del
primer paso predicho, mantenida durante los H pasos). Un Pearson alto puede venir en buena parte
de que la ventana real inicial ya indica el régimen de tráfico. Se reporta el del baseline y el
IC por episodios de la diferencia entre ambos.

La parte opcional --with-sumo-reexecution (tau de Kendall del ranking de acciones con
reejecución real en SUMO) no está implementada.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.world_model_evaluation import (  # noqa: E402
    load_episodes, load_reward_scaler, load_world_model, predict_next_step, rollout_episode,
)
from scripts.evaluate_multiseed_statistical import t_two_sided_p  # noqa: E402

CKPTS = {
    "LSTM": ROOT / "models/checkpoints/world_model_best.pt",
    "Transformer": ROOT / "models/checkpoints/world_model_transformer_best.pt",
    "TSMixer": ROOT / "models/checkpoints/world_model_tsmixer_best.pt",
}
TEST_LATENT = ROOT / "datasets/processed/test_latent.npz"
# 36 extra test episodes (SUMO seeds 9000-9035) from docs/results/exploratory/latent_sweep (section 16),
# encoded with the official Autoencoder. The .npz is not versioned; its md5 is in that folder's MANIFEST.json.
NEW_DIR = ROOT / "docs/results/exploratory/latent_sweep/test_episodes_expanded"
NEW_LATENT = NEW_DIR / "test_new_latent.npz"
DEVICE = torch.device("cpu")
B, BOOT_SEED = 10_000, 12345


@torch.no_grad()
def imagined_vs_real(model, hp, sc, episode, horizon):
    """Per start: (imagined H-step return, real H-step return, per-step squared reward errors,
    persistence-baseline return). The baseline is rollout_episode's: the last reward known before the
    first predicted transition, rewards[start + L - 2], held for the H steps."""
    L, action_dim = hp["sequence_length"], hp["action_dim"]
    z = torch.from_numpy(episode["z"])
    rewards = torch.from_numpy(episode["rewards"])
    onehot = F.one_hot(torch.from_numpy(episode["actions"]), num_classes=action_dim).float()
    out = []
    for start in range(z.shape[0] - L - horizon + 1):
        window_z = z[start:start + L].clone()
        r_imag, r_real, se = 0.0, 0.0, []
        for h in range(1, horizon + 1):
            pred_z, pred_r = predict_next_step(model, window_z, onehot[start + h - 1:start + h - 1 + L], DEVICE,
                                               sc["reward_mean"], sc["reward_std"])
            true_r = rewards[start + L + h - 2]
            r_imag += pred_r.item()
            r_real += true_r.item()
            se.append(((pred_r - true_r) ** 2).item())
            window_z = torch.cat([window_z[1:], pred_z.unsqueeze(0)], dim=0)
        out.append((r_imag, r_real, se, horizon * rewards[max(start + L - 2, 0)].item()))
    return out


def pearson(x, y):
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-300, 1 - r * r))
    return r, t_two_sided_p(float(t), n - 2)


def evaluate(model, hp, sc, episodes, horizon):
    per_ep, control_ok = {}, True
    for ep_id, ep in episodes.items():
        recs = imagined_vs_real(model, hp, sc, ep, horizon)
        ref = rollout_episode(model, ep, hp["sequence_length"], hp["action_dim"], horizon, DEVICE,
                              reward_mean=sc["reward_mean"], reward_std=sc["reward_std"])
        mine = [s for _, _, se, _ in recs for s in se]
        control_ok &= mine == [r["model_reward_se"] for r in ref]
        per_ep[ep_id] = (np.array([r[0] for r in recs]), np.array([r[1] for r in recs]), np.array([r[3] for r in recs]))
    imag = np.concatenate([v[0] for v in per_ep.values()])
    real = np.concatenate([v[1] for v in per_ep.values()])
    base = np.concatenate([v[2] for v in per_ep.values()])
    r, p = pearson(imag, real)
    r_base, _ = pearson(base, real)
    ids = list(per_ep)
    ei = np.random.default_rng(BOOT_SEED).integers(0, len(ids), size=(B, len(ids)))
    rb, bb, rbb = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        im = np.concatenate([per_ep[ids[j]][0] for j in ei[b]])
        re = np.concatenate([per_ep[ids[j]][1] for j in ei[b]])
        rb[b] = np.corrcoef(im, re)[0, 1]
        rbb[b] = rb[b] - np.corrcoef(np.concatenate([per_ep[ids[j]][2] for j in ei[b]]), re)[0, 1]
        bb[b] = np.mean(im - re)
    return {
        "n_episodios": len(ids), "n_tramos": int(len(imag)), "pearson": r,
        "pearson_p_tramos_independientes": p, "pearson_ic95_bootstrap_episodios": np.percentile(rb, [2.5, 97.5]).tolist(),
        "sesgo_medio_imaginado_menos_real": float(np.mean(imag - real)),
        "sesgo_ic95_bootstrap_episodios": np.percentile(bb, [2.5, 97.5]).tolist(),
        "pearson_baseline_persistente": r_base, "sesgo_baseline_persistente": float(np.mean(base - real)),
        "pearson_modelo_menos_baseline_ic95_bootstrap": np.percentile(rbb, [2.5, 97.5]).tolist(),
        "retorno_real_medio": float(real.mean()), "retorno_imaginado_medio": float(imag.mean()),
        "control_reproduce_rollout_episode": bool(control_ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--with-sumo-reexecution", action="store_true",
                    help="tau de Kendall con reejecución real en SUMO: no implementado aquí.")
    args = ap.parse_args()
    if args.with_sumo_reexecution:
        raise SystemExit("--with-sumo-reexecution no está implementado (fuera del alcance de esta entrega).")

    sets = {"test_oficial_12": load_episodes(TEST_LATENT)}
    if NEW_LATENT.exists():
        manifest = json.loads((NEW_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        md5 = hashlib.md5(NEW_LATENT.read_bytes()).hexdigest()
        if md5 != manifest["md5"]["test_new_latent.npz"]:
            raise SystemExit(f"{NEW_LATENT} no coincide con el md5 de su MANIFEST.json")
        new = load_episodes(NEW_LATENT)
        sets["test_nuevos_36"] = new
        sets["test_48"] = {**sets["test_oficial_12"], **new}

    rows = {}
    for name, ckpt in CKPTS.items():
        model, hp = load_world_model(ckpt, DEVICE)
        sc = load_reward_scaler(ckpt)
        rows[name] = {"horizonte": args.horizon}
        for set_name, episodes in sets.items():
            res = evaluate(model, hp, sc, episodes, args.horizon)
            if not res["control_reproduce_rollout_episode"]:
                raise SystemExit(f"[control] {name}/{set_name}: no reproduce rollout_episode - se detiene")
            rows[name][set_name] = res
            lo, hi = res["pearson_ic95_bootstrap_episodios"]
            blo, bhi = res["sesgo_ic95_bootstrap_episodios"]
            print(f"{name:12s} {set_name:16s} n={res['n_tramos']:5d}  Pearson={res['pearson']:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"sesgo={res['sesgo_medio_imaginado_menos_real']:+.2f} [{blo:+.2f}, {bhi:+.2f}]  "
                  f"| persistente: Pearson={res['pearson_baseline_persistente']:.3f} sesgo={res['sesgo_baseline_persistente']:+.2f} "
                  f"| modelo - persistente IC95 [{res['pearson_modelo_menos_baseline_ic95_bootstrap'][0]:+.3f}, "
                  f"{res['pearson_modelo_menos_baseline_ic95_bootstrap'][1]:+.3f}]  control OK")

    out = ROOT / "docs/results/model_fidelity.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
