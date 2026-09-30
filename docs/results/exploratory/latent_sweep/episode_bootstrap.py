"""ADDENDUM_test_episodes.md analysis (Phase 1): how much uncertainty comes from the composition of
the 12-episode test split, compared with the seed variance already measured. No training, no
simulation: rollouts of existing checkpoints on the existing test data, with the official evaluation
IMPORTED from evaluation/world_model_evaluation.py.

Stage 1 (cached in episode_errors.json): per checkpoint, E[e, h] = mean squared reward error of test
episode e at horizon h; aggregate_by_horizon over all records must reproduce the stored curves exactly.
Stage 2: paired episode bootstrap, seed-only / episode-only / joint CIs, episode influence."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.world_model_evaluation import (  # noqa: E402
    aggregate_by_horizon, load_episodes, load_reward_scaler, load_world_model, rollout_episode,
)

CK, W, PROC = ROOT / "models" / "checkpoints", ROOT / "models" / "checkpoints" / "exploratory_latent_sweep", ROOT / "datasets" / "processed"
H, MAX_H = list(range(1, 11)), 10
B, SEED = 10_000, 12345
CACHE = HERE / "episode_errors.json"


def js(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def official_curve(name):
    d = js(ROOT / "results" / name)
    d = d.get("per_horizon", d)
    return [d[str(h)]["model_reward_mse"] for h in H]


def run_list():
    """(key, checkpoint, test file, stored reward_mse curve that must be reproduced exactly)."""
    runs = [("off_lstm", CK / "world_model_best.pt", PROC / "test_latent.npz", official_curve("world_model_evaluation.json")),
            ("off_transformer", CK / "world_model_transformer_best.pt", PROC / "test_latent.npz",
             official_curve("world_model_transformer_evaluation.json")),
            ("off_tsmixer", CK / "world_model_tsmixer_best.pt", PROC / "test_latent.npz", official_curve("world_model_tsmixer_evaluation.json"))]
    e0 = js(ROOT / "docs" / "results" / "experiment_0_multiseed_300ep.json")
    for p in e0["pairs"]:
        s = p["seed"]
        runs.append((f"exp0_z_s{s}", ROOT / p["z_dir"] / "world_model_best.pt", PROC / "test_latent.npz", [p["z"][str(h)]["model_reward_mse"] for h in H]))
        runs.append((f"exp0_raw_s{s}", ROOT / p["raw_dir"] / "world_model_raw_best.pt", PROC / "test_raw_seq.npz", [p["raw"][str(h)]["model_reward_mse"] for h in H]))
    ae12 = {0: W / "ae12", **{a: W / f"ae12_seed{a}" for a in (1, 2, 3, 4)}}
    z12 = {0: W / "z12", **{a: W / f"z12_ae{a}" for a in (1, 2, 3, 4)}}
    for a in range(5):
        g5 = js(HERE / f"grid5_ae{a}_vs_raw24.json")
        for p in g5["pairs"]:
            s = p["seed"]
            runs.append((f"z12_ae{a}_s{s}", z12[a] / f"seed{s}" / "world_model_best.pt", ae12[a] / "test_latent.npz", [p["z"][str(h)]["model_reward_mse"] for h in H]))
    ae16 = {a: (W / "ae16" if a == 0 else W / f"ae16_seed{a}") for a in range(5)}
    layout = {"lstm": (lambda a: W / ("z16" if a == 0 else f"z16_ae{a}"), W / "raw24"),
              "transformer": (lambda a: W / f"tf16_ae{a}", W / "tf_raw24"), "tsmixer": (lambda a: W / f"ts16_ae{a}", W / "ts_raw24")}
    for arch, (zdir, rdir) in layout.items():
        g = js(HERE / f"gridarch_{arch}.json")
        for a in range(5):
            for s in range(5):
                runs.append((f"{arch}_z16_ae{a}_s{s}", zdir(a) / f"seed{s}" / "world_model_best.pt", ae16[a] / "test_latent.npz", g["z_reward_mse"][a][s]))
        for s in range(10):
            runs.append((f"{arch}_raw24_s{s}", rdir / f"seed{s}" / "world_model_best.pt", W / "data" / "test_raw_seq.npz", g["raw_reward_mse"][s]))
    return runs


def stage1():
    cache = js(CACHE) if CACHE.exists() else {}
    device = torch.device("cpu")
    bad = []
    for key, ckpt, test, stored in run_list():
        if key in cache:
            continue
        model, hp = load_world_model(ckpt, device)
        sc = load_reward_scaler(ckpt)
        eps = load_episodes(test)
        all_rec, E, ids = [], [], []
        for ep_id, ep in eps.items():
            rec = rollout_episode(model, ep, hp["sequence_length"], hp["action_dim"], MAX_H, device,
                                  reward_mean=sc["reward_mean"], reward_std=sc["reward_std"])
            all_rec.extend(rec)
            ids.append(int(ep_id))
            E.append([float(np.mean([r["model_reward_se"] for r in rec if r["horizon"] == h])) for h in H])
        agg = aggregate_by_horizon(all_rec, hp["latent_dim"])
        curve = [agg[h]["model_reward_mse"] for h in H]
        exact = curve == list(stored)
        if not exact:
            bad.append((key, max(abs(x - y) for x, y in zip(curve, stored))))
        cache[key] = {"episode_ids": ids, "E": E, "curve": curve, "exact_match_stored": exact}
        print(f"{key}: {'EXACTO' if exact else 'DIFERENTE'}", flush=True)
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    n_exact = sum(v["exact_match_stored"] for v in cache.values())
    print(f"[control] {n_exact}/{len(cache)} curvas reproducen exactamente lo guardado")
    if bad or n_exact != len(cache):
        print("[control] HAY DIFERENCIAS - se detiene:", bad)
        sys.exit(1)
    return cache


pct = lambda d: 100 * (math.exp(d) - 1)
q95 = lambda x: np.percentile(x, [2.5, 97.5])


def main():
    cache = stage1()
    ids = cache["off_lstm"]["episode_ids"]
    assert all(v["episode_ids"] == ids for v in cache.values()) and len(ids) == 12
    E = {k: np.array(v["E"]) for k, v in cache.items()}  # (12, 10)
    G0 = {k: float(np.mean(np.log(v["curve"]))) for k, v in cache.items()}
    ei = np.random.default_rng(SEED).integers(0, 12, size=(B, 12))
    Gep = {k: np.log(e[ei].mean(axis=1)).mean(axis=1) for k, e in E.items()}  # (B,)
    out = {"episode_ids": ids, "B": B}

    # ---------------- E1: official single checkpoints ----------------
    print("\n==================== E1. Checkpoints oficiales (una corrida), IC95 solo por episodios ====================")
    out["E1"] = {}
    ref_sd = {"lstm": [G0[f"exp0_z_s{s}"] for s in range(5)]}
    for arch in ("lstm", "transformer", "tsmixer"):
        ref_sd[f"{arch}_z16"] = [G0[f"{arch}_z16_ae{a}_s{s}"] for a in range(5) for s in range(5)]
    for arch in ("lstm", "transformer", "tsmixer"):
        k = f"off_{arch}"
        lo, hi = q95(Gep[k])
        mse_b = E[k][ei].mean(axis=1)  # (B, 10)
        hlo, hhi = np.percentile(mse_b, [2.5, 97.5], axis=0)
        sd_ep = float(Gep[k].std(ddof=1))
        sd_z16 = float(np.std(ref_sd[f"{arch}_z16"], ddof=1))
        extra = f", entre 5 semillas z del Exp. 0 {np.std(ref_sd['lstm'], ddof=1):.3f}" if arch == "lstm" else ""
        print(f"{arch:<12} GM {math.exp(G0[k]):7.1f}  IC95 episodios [{math.exp(lo):.1f}, {math.exp(hi):.1f}]  "
              f"({pct(lo - G0[k]):+.1f}% / {pct(hi - G0[k]):+.1f}%)  sd(g) episodios {sd_ep:.3f} | sd(g) entre 25 corridas z16 {sd_z16:.3f}{extra}")
        print("   reward_mse por h: " + "  ".join(f"h{h}:{c:.0f}[{a:.0f},{b:.0f}]" for h, c, a, b in zip(H, cache[k]["curve"], hlo, hhi)))
        out["E1"][arch] = {"g": G0[k], "gm": math.exp(G0[k]), "ci95_gm": [math.exp(lo), math.exp(hi)], "sd_g_episodes": sd_ep,
                           "sd_g_between_z16_runs": sd_z16, "per_h_mse": cache[k]["curve"], "per_h_ci95": [hlo.tolist(), hhi.tolist()]}
    out["E1"]["lstm"]["sd_g_between_exp0_z_seeds"] = float(np.std(ref_sd["lstm"], ddof=1))
    for X in ("transformer", "tsmixer"):
        d0 = G0["off_lstm"] - G0[f"off_{X}"]
        lo, hi = q95(Gep["off_lstm"] - Gep[f"off_{X}"])
        print(f"LSTM - {X} (oficiales): {pct(d0):+.1f}%  IC95 solo episodios [{pct(lo):+.1f}%, {pct(hi):+.1f}%]  ancho log {hi - lo:.3f}")
        out["E1"][f"lstm-{X}"] = {"D": d0, "ci95_episodes": [lo, hi]}

    # ---------------- E2-E4': decomposition ----------------
    def summarize(name, point, seed_b, ep_b, joint_b, published=None, bonf=False):
        s_lo, s_hi = q95(seed_b); e_lo, e_hi = q95(ep_b); j_lo, j_hi = q95(joint_b)
        ws, we, wj = s_hi - s_lo, e_hi - e_lo, j_hi - j_lo
        vs, ve = seed_b.var(ddof=1), ep_b.var(ddof=1)
        ratio = we / ws
        reading = "fuente menor" if ratio < 0.33 else "fuente relevante (menor que semillas)" if ratio < 1 else "fuente dominante o igual"
        excl = lambda lo, hi: "excluye 0" if (lo > 0 or hi < 0) else "incluye 0"
        change = excl(s_lo, s_hi) != excl(j_lo, j_hi)
        line = (f"{name:<28} {pct(point):+6.1f}% | semillas [{pct(s_lo):+.1f}, {pct(s_hi):+.1f}] ({excl(s_lo, s_hi)}) "
                f"| episodios [{pct(e_lo):+.1f}, {pct(e_hi):+.1f}] | conjunto [{pct(j_lo):+.1f}, {pct(j_hi):+.1f}] ({excl(j_lo, j_hi)}) "
                f"| anchos log s {ws:.3f} e {we:.3f} j {wj:.3f} | razon e/s {ratio:.2f} -> {reading} | frac var ep {ve / (ve + vs):.2f}"
                + (" | CAMBIA LA CONCLUSION" if change else ""))
        res = {"point": point, "ci_seed": [s_lo, s_hi], "ci_episodes": [e_lo, e_hi], "ci_joint": [j_lo, j_hi],
               "width_seed": ws, "width_episodes": we, "width_joint": wj, "ratio": ratio, "reading": reading,
               "var_frac_episodes": ve / (ve + vs), "conclusion_changes_95": change, "published_ci": published}
        if bonf:
            q = [100 * 0.05 / 3 / 2, 100 * (1 - 0.05 / 3 / 2)]
            sb, jb = np.percentile(seed_b, q), np.percentile(joint_b, q)
            res["ci_seed_bonf3"], res["ci_joint_bonf3"] = sb.tolist(), jb.tolist()
            res["conclusion_changes_bonf3"] = excl(*sb) != excl(*jb)
            line += f"\n{'':<28} Bonferroni 98.33%: semillas [{pct(sb[0]):+.1f}, {pct(sb[1]):+.1f}] -> conjunto [{pct(jb[0]):+.1f}, {pct(jb[1]):+.1f}]" + (
                " | CAMBIA" if res["conclusion_changes_bonf3"] else " | no cambia")
        print(line)
        return res

    def stack(keys):
        return np.array([G0[k] for k in keys]), np.stack([Gep[k] for k in keys], axis=1)  # (n,), (B, n)

    ar = np.arange(B)
    out["decomp"] = {}
    print("\n==================== E2-E4'. Delta con IC95: solo semillas / solo episodios / conjunto ====================")
    # E2: Experimento 0 oficial, 5 z vs 5 raw
    gz, gz_e = stack([f"exp0_z_s{s}" for s in range(5)]); gr, gr_e = stack([f"exp0_raw_s{s}" for s in range(5)])
    rng = np.random.default_rng(SEED); zi = rng.integers(0, 5, (B, 5)); ri = rng.integers(0, 5, (B, 5))
    out["decomp"]["E2_exp0_z_vs_raw"] = summarize(
        "E2 Exp.0 oficial z-crudo", gz.mean() - gr.mean(), gz[zi].mean(1) - gr[ri].mean(1),
        gz_e.mean(1) - gr_e.mean(1), gz_e[ar[:, None], zi].mean(1) - gr_e[ar[:, None], ri].mean(1))

    def grid(zkeys, rkeys):  # zkeys[a][s]
        gz = np.array([[G0[k] for k in row] for row in zkeys]); gz_e = np.stack([np.stack([Gep[k] for k in row], 1) for row in zkeys], 1)  # (B,5,5)
        gr, gr_e = stack(rkeys)
        return gz, gz_e, gr, gr_e

    def cluster_delta(label, zkeys, rkeys, published):
        gz, gz_e, gr, gr_e = grid(zkeys, rkeys)
        rng = np.random.default_rng(SEED)
        ai = rng.integers(0, 5, (B, 5)); si = rng.integers(0, 5, (B, 5, 5)); ri = rng.integers(0, len(rkeys), (B, len(rkeys)))
        seed_b = gz[ai[:, :, None], si].mean((1, 2)) - gr[ri].mean(1)
        ep_b = gz_e.mean((1, 2)) - gr_e.mean(1)
        joint_b = gz_e[ar[:, None, None], ai[:, :, None], si].mean((1, 2)) - gr_e[ar[:, None], ri].mean(1)
        return summarize(label, gz.mean() - gr.mean(), seed_b, ep_b, joint_b, published)

    raw24 = [f"lstm_raw24_s{s}" for s in range(10)]
    out["decomp"]["E3_z12_vs_raw24"] = cluster_delta("E3 sec.11 z12-crudo24", [[f"z12_ae{a}_s{s}" for s in range(5)] for a in range(5)], raw24, [-21.7, 12.0])
    pubs = {"lstm": [-21.3, 2.8], "transformer": [9.8, 60.4], "tsmixer": [21.1, 65.0]}
    for arch in ("lstm", "transformer", "tsmixer"):
        out["decomp"][f"E4_{arch}_z16_vs_raw24"] = cluster_delta(
            f"E4 sec.13 {arch} z16-crudo", [[f"{arch}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)],
            [f"{arch}_raw24_s{s}" for s in range(10)], pubs[arch])
    pubs14 = {("lstm", "transformer"): [-48.6, -26.4], ("lstm", "tsmixer"): [-66.0, -52.0], ("transformer", "tsmixer"): [-45.9, -24.5]}
    for (A, Bn), pub in pubs14.items():
        ga, ga_e, _, _ = grid([[f"{A}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)], raw24)
        gb, gb_e, _, _ = grid([[f"{Bn}_z16_ae{a}_s{s}" for s in range(5)] for a in range(5)], raw24)
        rng = np.random.default_rng(SEED)
        ai = rng.integers(0, 5, (B, 5)); sa = rng.integers(0, 5, (B, 5, 5)); sb = rng.integers(0, 5, (B, 5, 5))
        seed_b = ga[ai[:, :, None], sa].mean((1, 2)) - gb[ai[:, :, None], sb].mean((1, 2))
        ep_b = ga_e.mean((1, 2)) - gb_e.mean((1, 2))
        joint_b = ga_e[ar[:, None, None], ai[:, :, None], sa].mean((1, 2)) - gb_e[ar[:, None, None], ai[:, :, None], sb].mean((1, 2))
        out["decomp"][f"E4b_{A}-{Bn}"] = summarize(f"E4' sec.14 {A}-{Bn}", ga.mean() - gb.mean(), seed_b, ep_b, joint_b, pub, bonf=True)

    # ---------------- influence ----------------
    print("\n==================== Influencia de episodios (descriptivo) ====================")
    counts = np.stack([np.bincount(r, minlength=12) for r in ei])  # (B, 12)
    stats = {
        "g LSTM oficial": (Gep["off_lstm"], lambda keep: np.log(E["off_lstm"][keep].mean(0)).mean()),
        "Delta Exp.0 z-crudo": (np.stack([Gep[f"exp0_z_s{s}"] for s in range(5)], 1).mean(1) - np.stack([Gep[f"exp0_raw_s{s}"] for s in range(5)], 1).mean(1),
                                lambda keep: np.mean([np.log(E[f"exp0_z_s{s}"][keep].mean(0)).mean() for s in range(5)]) - np.mean([np.log(E[f"exp0_raw_s{s}"][keep].mean(0)).mean() for s in range(5)])),
        "Delta LSTM z16-crudo24": (np.stack([Gep[f"lstm_z16_ae{a}_s{s}"] for a in range(5) for s in range(5)], 1).mean(1) - np.stack([Gep[k] for k in raw24], 1).mean(1),
                                   lambda keep: np.mean([np.log(E[f"lstm_z16_ae{a}_s{s}"][keep].mean(0)).mean() for a in range(5) for s in range(5)]) - np.mean([np.log(E[k][keep].mean(0)).mean() for k in raw24])),
        "LSTM-Transformer z16": (np.stack([Gep[f"lstm_z16_ae{a}_s{s}"] for a in range(5) for s in range(5)], 1).mean(1) - np.stack([Gep[f"transformer_z16_ae{a}_s{s}"] for a in range(5) for s in range(5)], 1).mean(1),
                                 lambda keep: np.mean([np.log(E[f"lstm_z16_ae{a}_s{s}"][keep].mean(0)).mean() for a in range(5) for s in range(5)]) - np.mean([np.log(E[f"transformer_z16_ae{a}_s{s}"][keep].mean(0)).mean() for a in range(5) for s in range(5)])),
    }
    out["influence"] = {}
    for name, (vals, fn) in stats.items():
        lo_t, hi_t = np.percentile(vals, [2.5, 97.5])
        c_lo, c_hi = counts[vals <= lo_t].mean(0), counts[vals >= hi_t].mean(0)
        full = fn(np.arange(12))
        jack = [fn(np.array([j for j in range(12) if j != e])) - full for e in range(12)]
        top = lambda c: ", ".join(f"ep{ids[j]} x{c[j]:.2f}" for j in np.argsort(-c)[:3])
        print(f"{name}: cola baja {top(c_lo)} | cola alta {top(c_hi)}")
        print("   jackknife (cambio al quitar cada episodio, log): " + " ".join(f"ep{ids[e]}:{jack[e]:+.3f}" for e in range(12)))
        out["influence"][name] = {"tail_low_mean_counts": c_lo.tolist(), "tail_high_mean_counts": c_hi.tolist(), "jackknife": jack}
    share = E["off_lstm"][:, 9] / E["off_lstm"][:, 9].sum()
    print("peso de cada episodio en el error h=10 de la LSTM oficial: " + " ".join(f"ep{ids[e]}:{100 * share[e]:.1f}%" for e in np.argsort(-share)))
    out["influence"]["share_h10_off_lstm"] = dict(zip(map(str, ids), share.tolist()))
    (HERE / "episode_bootstrap.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
