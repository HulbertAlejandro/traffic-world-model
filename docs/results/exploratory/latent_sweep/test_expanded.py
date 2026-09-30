"""
############################################################################################
#  INCOMPLETO - EN PAUSA (2026-09-30). Nunca paso del control 1: Smart App Control bloquea
#  SUMO (sumo.exe no puede cargar fmt.dll; traci "Could not connect."). No se recolecto ningun
#  episodio y 'collect', 'prepare' NO se han ejecutado nunca: codigo sin probar. 'evaluate' y
#  'analyze' (P1-P4 del addendum) no estan escritos. Se conserva solo para retomar la fase si
#  se levanta el bloqueo; ver ADDENDUM_test_expanded.md, seccion "Estado".
############################################################################################

ADDENDUM_test_expanded.md: 36 new TEST episodes (SUMO seeds 9000-9035) with the official collection
protocol, then the section-15 analysis on 12 original / 36 new / 48 episodes. No retraining.

Subcommands (in order):
    replay     control 1: re-simulate original test episodes 8 and 53 with their recorded actions
    collect    36 new episodes via scripts/collect_dataset.collect_dataset (imported)
    prepare    controls 2-4 and all derived test files (official scaler, official + 10 exploratory AEs)
    evaluate   E[e, h] of the 143 checkpoints on the 36 new episodes (official evaluation, imported)
    analyze    P1-P4
"""
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

OUT = HERE / "test_episodes_expanded"
RAW = OUT / "raw"
PROC = ROOT / "datasets" / "processed"
W = ROOT / "models" / "checkpoints" / "exploratory_latent_sweep"
SEED_START, N_NEW = 9000, 36
DROP = (22, 23)
FIELDS = ("states", "actions", "rewards", "next_states", "terminated", "truncated", "episode_id", "time_step")


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- control 1
def replay():
    from environments import TrafficEnvironment
    test_raw = load(PROC / "test_raw.npz")
    env = TrafficEnvironment()
    ok = True
    try:
        for ep in (8, 53):
            m = test_raw["episode_id"] == ep
            order = np.argsort(test_raw["time_step"][m])
            rec = {k: test_raw[k][m][order] for k in ("states", "actions", "rewards", "next_states")}
            state, _ = env.reset(seed=int(ep))
            st, rw, ns = [], [], []
            for a in rec["actions"]:
                nxt, r, term, trunc, _ = env.step(int(a))
                st.append(np.asarray(state, dtype=np.float32)); rw.append(np.float32(r)); ns.append(np.asarray(nxt, dtype=np.float32))
                state = nxt
            st, rw, ns = np.array(st), np.array(rw), np.array(ns)
            same = (np.array_equal(st, rec["states"]) and np.array_equal(rw, rec["rewards"]) and np.array_equal(ns, rec["next_states"]))
            print(f"[control 1] episodio {ep}: {len(rw)} pasos, estados/recompensas/next_states "
                  f"{'IDENTICOS' if same else 'DIFERENTES'} (max |dif| estado {np.abs(st - rec['states']).max():.3e}, "
                  f"recompensa {np.abs(rw - rec['rewards']).max():.3e})")
            ok &= same
    finally:
        env.close()
    print(f"[control 1] {'OK' if ok else 'FALLA - detener'}")
    return ok


# ---------------------------------------------------------------- collection
def collect():
    from scripts.collect_dataset import collect_dataset
    RAW.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED_START)
    paths = collect_dataset(num_episodes=N_NEW, steps_per_episode=60, output_dir=RAW, seed_start=SEED_START)
    for i, p in enumerate(paths):  # episode_id = SUMO seed, as in the official dataset
        d = load(p)
        d["episode_id"] = np.full_like(d["episode_id"], SEED_START + i)
        np.savez(RAW / f"episode_{SEED_START + i}.npz", **d)
        p.unlink()
    print(f"recolectados {len(paths)} episodios en {RAW}")


# ---------------------------------------------------------------- derived files
def encode(ae_ckpt, states, device):
    import torch
    from evaluation.autoencoder_evaluation import load_autoencoder
    ae = load_autoencoder(checkpoint_path=ae_ckpt, input_dim=states.shape[1], device=device)
    ae.eval()
    with torch.no_grad():
        return ae.encode(torch.from_numpy(states.astype(np.float32)).to(device)).cpu().numpy().astype(np.float32)


def seq_payload(d, z, next_z):
    return dict(z=z.astype(np.float32), next_z=next_z.astype(np.float32), actions=d["actions"].astype(np.int64),
                rewards=d["rewards"].astype(np.float32), episode_id=d["episode_id"].astype(np.int64),
                time_step=d["time_step"].astype(np.int64))


def latent_payload(d, z, next_z):
    p = seq_payload(d, z, next_z)
    p.update(terminated=d["terminated"].astype(bool), truncated=d["truncated"].astype(bool))
    return p


AE_DIRS = {**{f"ae12{'' if a == 0 else f'_seed{a}'}": None for a in range(5)}, **{f"ae16{'' if a == 0 else f'_seed{a}'}": None for a in range(5)}}


def prepare():
    import torch
    device = torch.device("cpu")
    with open(PROC / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    norm = lambda x: (x - sc["state_mean"]) / sc["state_std"]
    ok = True

    # control 2: official normalization reproduced exactly
    tr, tn = load(PROC / "test_raw.npz"), load(PROC / "test.npz")
    c2 = np.array_equal(norm(tr["states"]), tn["states"]) and np.array_equal(norm(tr["next_states"]), tn["next_states"])
    print(f"[control 2] scaler.pkl sobre test_raw.npz reproduce test.npz: {'EXACTO' if c2 else 'DIFERENTE'}"); ok &= c2

    # control 3: encodings reproduced exactly on the ORIGINAL test
    zl = load(PROC / "test_latent.npz")
    c3 = np.array_equal(encode(ROOT / "models/checkpoints/autoencoder_best.pt", tn["states"], device), zl["z"])
    print(f"[control 3] AE oficial reproduce test_latent.npz: {'EXACTO' if c3 else 'DIFERENTE'}"); ok &= c3
    t24 = load(W / "data" / "test.npz")
    c3b = np.array_equal(np.delete(tn["states"], list(DROP), axis=1), t24["states"])
    print(f"[control 3] test de 24 dims = oficial sin columnas 22 y 23: {'EXACTO' if c3b else 'DIFERENTE'}"); ok &= c3b
    for name in AE_DIRS:
        ref = load(W / name / "test_latent.npz")
        same = np.array_equal(encode(W / name / "autoencoder_best.pt", t24["states"], device), ref["z"])
        print(f"[control 3] {name} reproduce su test_latent.npz: {'EXACTO' if same else 'DIFERENTE'}"); ok &= same
    if not ok:
        print("[controles 2-3] FALLA - detener"); sys.exit(1)

    # new episodes: merge in seed order
    eps = [load(RAW / f"episode_{SEED_START + i}.npz") for i in range(N_NEW)]
    new = {k: np.concatenate([e[k] for e in eps]) for k in FIELDS}
    n_steps = {len(e["rewards"]) for e in eps}
    finite = all(np.isfinite(new[k]).all() for k in ("states", "next_states", "rewards"))
    zero_cols = bool(np.all(new["states"][:, list(DROP)] == 0) and np.all(new["next_states"][:, list(DROP)] == 0))
    ids = np.unique(new["episode_id"]).tolist()
    c4 = n_steps == {60} and finite and zero_cols and ids == list(range(SEED_START, SEED_START + N_NEW))
    print(f"[control 4] {len(eps)} episodios, pasos {n_steps}, finitos {finite}, columnas 22-23 en cero {zero_cols}, ids {ids[0]}..{ids[-1]}: "
          f"{'OK' if c4 else 'FALLA'}")
    if not c4:
        sys.exit(1)
    np.savez(OUT / "test_new_raw.npz", **new)
    nn = dict(new, states=norm(new["states"]), next_states=norm(new["next_states"]))
    np.savez(OUT / "test_new.npz", **nn)
    np.savez(OUT / "test_new_raw_seq.npz", **seq_payload(nn, nn["states"], nn["next_states"]))
    np.savez(OUT / "test_new_latent.npz", **latent_payload(
        nn, encode(ROOT / "models/checkpoints/autoencoder_best.pt", nn["states"], device),
        encode(ROOT / "models/checkpoints/autoencoder_best.pt", nn["next_states"], device)))
    n24 = dict(nn, states=np.delete(nn["states"], list(DROP), axis=1), next_states=np.delete(nn["next_states"], list(DROP), axis=1))
    np.savez(OUT / "test_new24_raw_seq.npz", **seq_payload(n24, n24["states"], n24["next_states"]))
    for name in AE_DIRS:
        (OUT / name).mkdir(exist_ok=True)
        ck = W / name / "autoencoder_best.pt"
        np.savez(OUT / name / "test_new_latent.npz", **latent_payload(n24, encode(ck, n24["states"], device), encode(ck, n24["next_states"], device)))
    files = sorted(p for p in OUT.rglob("*.npz"))
    manifest = {"sumo_seeds": [SEED_START, SEED_START + N_NEW - 1], "action_seed": "np.random.seed(9000) once before collect_dataset",
                "episodes": N_NEW, "steps_per_episode": 60, "transitions": int(len(new["rewards"])),
                "scaler": "datasets/processed/scaler.pkl (official, not refitted)",
                "md5": {p.relative_to(OUT).as_posix(): hashlib.md5(p.read_bytes()).hexdigest() for p in files}}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"archivos derivados: {len(files)}; MANIFEST.json escrito")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "replay":
        sys.exit(0 if replay() else 1)
    elif cmd == "collect":
        collect()
    elif cmd == "prepare":
        prepare()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
