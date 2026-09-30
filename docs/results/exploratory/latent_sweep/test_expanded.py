"""ADDENDUM_test_expanded.md: 36 new TEST episodes (SUMO seeds 9000-9035) with the official collection
protocol, then the section-15 analysis on 12 original / 36 new / 48 episodes. No retraining.

Subcommands (in order):
    replay     control 1: re-simulate original test episodes 8 and 53 with their recorded actions
    collect    36 new episodes via scripts/collect_dataset.collect_dataset (imported)
    prepare    controls 2-4 and all derived test files (official scaler, official + 10 exploratory AEs)
    evaluate   control 5 (section-15 E[e, h] reproduced on the original test), then E[e, h] of the 143
               checkpoints on the 36 new episodes (official evaluation, imported) -> episode_errors_new.json
    analyze    P1-P4 on 12 original / 36 new / 48 -> test_expanded.out, test_expanded.json
Results: REPORT.md, section 16.
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


# ---------------------------------------------------------------- evaluation
CACHE_NEW = HERE / "episode_errors_new.json"
H, MAX_H = list(range(1, 11)), 10


def new_test_file(test):
    """Original test file of a run -> its counterpart built from the 36 new episodes."""
    test = Path(test)
    if test == PROC / "test_latent.npz":
        return OUT / "test_new_latent.npz"
    if test == PROC / "test_raw_seq.npz":
        return OUT / "test_new_raw_seq.npz"
    if test == W / "data" / "test_raw_seq.npz":
        return OUT / "test_new24_raw_seq.npz"
    if test.name == "test_latent.npz" and test.parent.name in AE_DIRS:
        return OUT / test.parent.name / "test_new_latent.npz"
    raise ValueError(f"no counterpart for {test}")


def episode_errors(ckpt, test, device):
    from evaluation.world_model_evaluation import load_episodes, load_reward_scaler, load_world_model, rollout_episode
    model, hp = load_world_model(ckpt, device)
    sc = load_reward_scaler(ckpt)
    ids, E = [], []
    for ep_id, ep in load_episodes(test).items():
        rec = rollout_episode(model, ep, hp["sequence_length"], hp["action_dim"], MAX_H, device,
                              reward_mean=sc["reward_mean"], reward_std=sc["reward_std"])
        ids.append(int(ep_id))
        E.append([float(np.mean([r["model_reward_se"] for r in rec if r["horizon"] == h])) for h in H])
    return ids, E


def evaluate():
    import torch
    from episode_bootstrap import run_list
    device = torch.device("cpu")
    runs = run_list()
    assert len(runs) == 143, len(runs)
    old = json.loads((HERE / "episode_errors.json").read_text(encoding="utf-8"))

    # control 5: the imported evaluation still reproduces the section-15 E[e, h] on the ORIGINAL test
    ok = True
    for key, ckpt, test, _ in runs[:3]:
        ids, E = episode_errors(ckpt, test, device)
        same = ids == old[key]["episode_ids"] and E == old[key]["E"]
        print(f"[control 5] {key} sobre los 12 originales reproduce episode_errors.json: {'EXACTO' if same else 'DIFERENTE'}", flush=True)
        ok &= same
    if not ok:
        print("[control 5] FALLA - detener"); sys.exit(1)

    cache = json.loads(CACHE_NEW.read_text(encoding="utf-8")) if CACHE_NEW.exists() else {}
    for i, (key, ckpt, test, _) in enumerate(runs):
        if key in cache:
            continue
        ids, E = episode_errors(ckpt, new_test_file(test), device)
        assert ids == list(range(SEED_START, SEED_START + N_NEW)), (key, ids)
        assert np.isfinite(E).all(), key
        cache[key] = {"test_file": new_test_file(test).relative_to(HERE).as_posix(), "episode_ids": ids, "E": E}
        CACHE_NEW.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"[{i + 1}/{len(runs)}] {key}: h1 {np.mean([e[0] for e in E]):.1f}  h10 {np.mean([e[9] for e in E]):.1f}", flush=True)
    print(f"episode_errors_new.json: {len(cache)} checkpoints x {N_NEW} episodios")


# ---------------------------------------------------------------- analysis (P1-P4)
B, BOOT_SEED = 10_000, 12345
ARCHS = ("lstm", "transformer", "tsmixer")
PAIRS = (("lstm", "transformer"), ("lstm", "tsmixer"), ("transformer", "tsmixer"))
PROBLEM_EPS = (53, 47, 12, 77)
pct = lambda d: 100 * (math.exp(d) - 1)
q95 = lambda x: np.percentile(x, [2.5, 97.5])
QB = [100 * 0.05 / 3 / 2, 100 * (1 - 0.05 / 3 / 2)]
excl = lambda lo, hi: bool(lo > 0 or hi < 0)


def section15_stats(E):
    """Same bootstraps as episode_bootstrap.py (section 15), on whatever episode set E holds: {key: (n, 10)}."""
    n = next(iter(E.values())).shape[0]
    G0 = {k: float(np.mean(np.log(e.mean(axis=0)))) for k, e in E.items()}
    ei = np.random.default_rng(BOOT_SEED).integers(0, n, size=(B, n))
    Gep = {k: np.log(e[ei].mean(axis=1)).mean(axis=1) for k, e in E.items()}
    ar = np.arange(B)
    res = {}

    def pack(point, ep_b, seed_b=None, joint_b=None, bonf=False):
        r = {"point": point, "ci_episodes": q95(ep_b).tolist()}
        r["width_episodes"] = r["ci_episodes"][1] - r["ci_episodes"][0]
        if seed_b is not None:
            r["ci_seed"], r["ci_joint"] = q95(seed_b).tolist(), q95(joint_b).tolist()
            r["width_seed"], r["width_joint"] = r["ci_seed"][1] - r["ci_seed"][0], r["ci_joint"][1] - r["ci_joint"][0]
            r["joint_excludes_0"] = excl(*r["ci_joint"])
        if bonf:
            r["ci_joint_bonf3"] = np.percentile(joint_b, QB).tolist()
            r["joint_bonf3_excludes_0"] = excl(*r["ci_joint_bonf3"])
        return r

    for arch in ARCHS:  # levels of the official single checkpoints
        k = f"off_{arch}"
        res[f"level_off_{arch}"] = pack(G0[k], Gep[k])

    gz = np.array([G0[f"exp0_z_s{s}"] for s in range(5)]); gz_e = np.stack([Gep[f"exp0_z_s{s}"] for s in range(5)], 1)
    gr = np.array([G0[f"exp0_raw_s{s}"] for s in range(5)]); gr_e = np.stack([Gep[f"exp0_raw_s{s}"] for s in range(5)], 1)
    rng = np.random.default_rng(BOOT_SEED); zi = rng.integers(0, 5, (B, 5)); ri = rng.integers(0, 5, (B, 5))
    res["E2_exp0_z_vs_raw"] = pack(gz.mean() - gr.mean(), gz_e.mean(1) - gr_e.mean(1), gz[zi].mean(1) - gr[ri].mean(1),
                                   gz_e[ar[:, None], zi].mean(1) - gr_e[ar[:, None], ri].mean(1))

    def grid(zkeys):
        return (np.array([[G0[k] for k in row] for row in zkeys]),
                np.stack([np.stack([Gep[k] for k in row], 1) for row in zkeys], 1))  # (5,5), (B,5,5)

    def cluster_delta(zkeys, rkeys):
        gz, gz_e = grid(zkeys)
        gr, gr_e = np.array([G0[k] for k in rkeys]), np.stack([Gep[k] for k in rkeys], 1)
        rng = np.random.default_rng(BOOT_SEED)
        ai = rng.integers(0, 5, (B, 5)); si = rng.integers(0, 5, (B, 5, 5)); ri = rng.integers(0, len(rkeys), (B, len(rkeys)))
        return pack(gz.mean() - gr.mean(), gz_e.mean((1, 2)) - gr_e.mean(1),
                    gz[ai[:, :, None], si].mean((1, 2)) - gr[ri].mean(1),
                    gz_e[ar[:, None, None], ai[:, :, None], si].mean((1, 2)) - gr_e[ar[:, None], ri].mean(1))

    res["E3_z12_vs_raw24"] = cluster_delta([[f"z12_ae{a}_s{s}" for s in range(5)] for a in range(5)], [f"lstm_raw24_s{s}" for s in range(10)])
    for arch in ARCHS:
        res[f"E4_{arch}_z16_vs_raw24"] = cluster_delta([[f"{arch}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)],
                                                       [f"{arch}_raw24_s{s}" for s in range(10)])
    for A, Bn in PAIRS:
        ga, ga_e = grid([[f"{A}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)])
        gb, gb_e = grid([[f"{Bn}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)])
        rng = np.random.default_rng(BOOT_SEED)
        ai = rng.integers(0, 5, (B, 5)); sa = rng.integers(0, 5, (B, 5, 5)); sb = rng.integers(0, 5, (B, 5, 5))
        res[f"E4b_{A}-{Bn}"] = pack(ga.mean() - gb.mean(), ga_e.mean((1, 2)) - gb_e.mean((1, 2)),
                                    ga[ai[:, :, None], sa].mean((1, 2)) - gb[ai[:, :, None], sb].mean((1, 2)),
                                    ga_e[ar[:, None, None], ai[:, :, None], sa].mean((1, 2)) - gb_e[ar[:, None, None], ai[:, :, None], sb].mean((1, 2)),
                                    bonf=True)
    return res


def jackknife(E, ids, zkeys, rkeys):
    """Change of Delta = mean g(zkeys) - mean g(rkeys) when each episode is dropped (log scale)."""
    g = lambda keys, keep: np.mean([np.log(E[k][keep].mean(0)).mean() for k in keys])
    allk = np.arange(len(ids))
    full = g(zkeys, allk) - g(rkeys, allk)
    return full, {ids[e]: g(zkeys, allk[allk != e]) - g(rkeys, allk[allk != e]) - full for e in allk}


def analyze():
    old = json.loads((HERE / "episode_errors.json").read_text(encoding="utf-8"))
    new = json.loads(CACHE_NEW.read_text(encoding="utf-8"))
    assert set(old) == set(new) and len(old) == 143
    ids12, ids36 = old["off_lstm"]["episode_ids"], new["off_lstm"]["episode_ids"]
    assert all(old[k]["episode_ids"] == ids12 and new[k]["episode_ids"] == ids36 for k in old)
    ids48 = ids12 + ids36
    E48 = {k: np.vstack([np.array(old[k]["E"]), np.array(new[k]["E"])]) for k in old}  # (48, 10)
    sets = {"12": np.arange(12), "36": np.arange(12, 48), "48": np.arange(48)}
    S = {name: section15_stats({k: e[idx] for k, e in E48.items()}) for name, idx in sets.items()}
    names = list(S["12"])
    out = {"episode_ids": {"12": ids12, "36": ids36}, "B": B, "stats": S}

    # consistency with section 15 (same E, same bootstrap indices for the 12 original)
    ref = json.loads((HERE / "episode_bootstrap.json").read_text(encoding="utf-8"))
    refmap = {n: (None if n.startswith("level") else ref["decomp"][n]) for n in names}
    dev = 0.0
    for n in names:
        if refmap[n] is None:
            gm = ref["E1"][n.replace("level_off_", "")]["ci95_gm"]
            dev = max(dev, max(abs(math.log(g) - c) for g, c in zip(gm, S["12"][n]["ci_episodes"])))
        else:
            for f in ("ci_seed", "ci_episodes", "ci_joint"):
                dev = max(dev, max(abs(a - b) for a, b in zip(refmap[n][f], S["12"][n][f])))
    print(f"[consistencia] set de 12 frente a episode_bootstrap.json (sec. 15): max |dif| de los limites de IC (log) = {dev:.2e}")
    out["consistency_max_abs_diff_vs_section15"] = dev

    # descriptive comparability
    def ep_returns(path):
        d = load(path)
        return np.array([d["rewards"][d["episode_id"] == e].sum() for e in np.unique(d["episode_id"])])
    desc = {"test12": ep_returns(PROC / "test_raw.npz"), "new36": ep_returns(OUT / "test_new_raw.npz"), "train": ep_returns(PROC / "train_raw.npz")}
    print("\n==================== Comparabilidad: recompensa total por episodio ====================")
    out["descriptive_episode_return"] = {}
    for k, v in desc.items():
        print(f"{k:<7} n={len(v):>2}  media {v.mean():8.1f}  sd {v.std(ddof=1):7.1f}  min {v.min():8.1f}  mediana {np.median(v):8.1f}")
        out["descriptive_episode_return"][k] = {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "median": float(np.median(v))}

    # table of every statistic in the three sets
    print("\n==================== Estadisticos en 12 / 36 / 48 episodios (IC95; niveles en GM, deltas en %) ====================")
    for n in names:
        lvl = n.startswith("level")
        f = (lambda x: f"{math.exp(x):.1f}") if lvl else (lambda x: f"{pct(x):+.1f}%")
        for s in ("12", "36", "48"):
            r = S[s][n]
            line = f"{n:<28} n={s:<3} {f(r['point']):>8} | episodios [{f(r['ci_episodes'][0])}, {f(r['ci_episodes'][1])}] ancho {r['width_episodes']:.3f}"
            if not lvl:
                line += (f" | semillas ancho {r['width_seed']:.3f} | conjunto [{f(r['ci_joint'][0])}, {f(r['ci_joint'][1])}] "
                         f"{'excluye 0' if r['joint_excludes_0'] else 'incluye 0'}")
            if "ci_joint_bonf3" in r:
                line += f" | Bonf. [{f(r['ci_joint_bonf3'][0])}, {f(r['ci_joint_bonf3'][1])}] {'excluye 0' if r['joint_bonf3_excludes_0'] else 'incluye 0'}"
            print(line)

    # P1
    print("\n==================== P1. Razon de anchos del IC solo por episodios (48 / 12; sqrt predice 0.50) ====================")
    r48 = {n: S["48"][n]["width_episodes"] / S["12"][n]["width_episodes"] for n in names}
    r36 = {n: S["36"][n]["width_episodes"] / S["12"][n]["width_episodes"] for n in names}
    for n in names:
        print(f"{n:<28} 48/12 {r48[n]:.2f}   (36/12 {r36[n]:.2f}; sqrt predice 0.58)")
    med = float(np.median(list(r48.values())))
    p1 = "se cumple aproximadamente" if 0.40 <= med <= 0.60 else "se reduce menos de lo esperado" if med > 0.60 else "se reduce mas"
    print(f"mediana 48/12 = {med:.2f} -> {p1}   (mediana 36/12 = {np.median(list(r36.values())):.2f})")
    out["P1"] = {"ratio_48_12": r48, "ratio_36_12": r36, "median_48_12": med, "median_36_12": float(np.median(list(r36.values()))), "reading": p1}

    # P2
    print("\n==================== P2. La estimacion con los 36 nuevos, dentro del IC solo por episodios de los 12? ====================")
    inside = {}
    for n in names:
        lo, hi = S["12"][n]["ci_episodes"]; p = S["36"][n]["point"]
        inside[n] = bool(lo <= p <= hi)
        pos = (p - lo) / (hi - lo)
        print(f"{n:<28} 12: [{lo:+.3f}, {hi:+.3f}]  36: {p:+.3f}  {'dentro' if inside[n] else 'FUERA'} (posicion relativa {pos:.2f})")
    k_in = sum(inside.values())
    p2 = "el IC de 12 subestimaba la incertidumbre por episodios" if k_in < len(names) / 2 else "compatible con un bootstrap razonablemente calibrado"
    print(f"dentro: {k_in}/{len(names)} -> {p2}")
    out["P2"] = {"inside": inside, "n_inside": k_in, "n": len(names), "reading": p2}

    # P3
    print("\n==================== P3. Cambia alguna conclusion? (IC conjunto excluye 0, 12 -> 48) ====================")
    p3 = {}
    for n in names:
        if n.startswith("level"):
            continue
        a, b, c = (S[s][n]["joint_excludes_0"] for s in ("12", "36", "48"))
        entry = {"12": a, "36": b, "48": c, "changes_12_to_48": a != c}
        msg = f"{n:<28} 95%: 12 {'excluye' if a else 'incluye'} -> 48 {'excluye' if c else 'incluye'} (36 solos: {'excluye' if b else 'incluye'})"
        if "ci_joint_bonf3" in S["12"][n]:
            ab, bb, cb = (S[s][n]["joint_bonf3_excludes_0"] for s in ("12", "36", "48"))
            entry.update(bonf3_12=ab, bonf3_36=bb, bonf3_48=cb, bonf3_changes_12_to_48=ab != cb)
            msg += f" | Bonferroni: 12 {'excluye' if ab else 'incluye'} -> 48 {'excluye' if cb else 'incluye'} (36: {'excluye' if bb else 'incluye'})"
        if a != c or entry.get("bonf3_changes_12_to_48"):
            msg += "  <- CAMBIA"
        print(msg)
        p3[n] = entry
    out["P3"] = p3

    # P4
    print("\n==================== P4. Siguen dominando ep53, ep47, ep12 y ep77? (descriptivo) ====================")
    h10 = E48["off_lstm"][:, 9]
    order = np.argsort(-h10)
    rank = {ids48[j]: r + 1 for r, j in enumerate(order)}
    pos = [ids48.index(e) for e in PROBLEM_EPS]
    share4_48 = float(h10[pos].sum() / h10.sum())
    share4_12 = float(h10[pos].sum() / h10[:12].sum())
    min4 = h10[pos].min()
    n_new_above = int((h10[12:] > min4).sum())
    print(f"fraccion del error h=10 de la LSTM oficial en los 4: {100 * share4_48:.1f}% de 48 (era {100 * share4_12:.1f}% de 12)")
    print("rango entre 48 (1 = mayor error): " + ", ".join(f"ep{e}: {rank[e]}" for e in PROBLEM_EPS))
    print(f"episodios nuevos con error h=10 mayor que el menor de los 4 (ep{PROBLEM_EPS[int(np.argmin(h10[pos]))]}, {min4:.0f}): {n_new_above}/36")
    print("top-10 por error h=10: " + " ".join(f"ep{ids48[j]}:{h10[j]:.0f}" for j in order[:10]))
    jk = {}
    for label, zk, rk in (("Delta LSTM z16-crudo24", [f"lstm_z16_ae{a}_s{s}" for a in range(5) for s in range(5)], [f"lstm_raw24_s{s}" for s in range(10)]),
                          ("LSTM-Transformer z16", [f"lstm_z16_ae{a}_s{s}" for a in range(5) for s in range(5)], [f"transformer_z16_ae{a}_s{s}" for a in range(5) for s in range(5)])):
        full, ch = jackknife(E48, ids48, zk, rk)
        top = sorted(ch, key=lambda e: -abs(ch[e]))[:6]
        print(f"jackknife 48, {label} (Delta {pct(full):+.1f}%): mayores cambios " + " ".join(f"ep{e}:{ch[e]:+.3f}" for e in top)
              + " | los 4: " + " ".join(f"ep{e}:{ch[e]:+.3f}" for e in PROBLEM_EPS))
        jk[label] = {"delta_log": full, "change_when_dropped": {str(e): v for e, v in ch.items()}}
    out["P4"] = {"share_h10_four_of_48": share4_48, "share_h10_four_of_12": share4_12, "rank_of_48": {str(e): rank[e] for e in PROBLEM_EPS},
                 "n_new_above_min_of_four": n_new_above, "h10_off_lstm": dict(zip(map(str, ids48), h10.tolist())), "jackknife_48": jk}
    (HERE / "test_expanded.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "replay":
        sys.exit(0 if replay() else 1)
    elif cmd == "collect":
        collect()
    elif cmd == "prepare":
        prepare()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
