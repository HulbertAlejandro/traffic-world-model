"""Figuras 5 a 11 del artículo (v2), solo a partir de los resultados versionados.

    python scripts/v2/make_article_figures.py [--out docs/figures_v2]

Solo lectura: no simula, no entrena y no modifica ningún resultado. Depende de json, csv, numpy y
matplotlib (sin pandas ni SUMO). PNG a 300 dpi, rótulos en español, sin títulos incrustados (el pie va
en el artículo). Cada serie lleva, además del color, una trama, un marcador o un estilo de línea, para
que se distinga impresa en blanco y negro. Las figuras 1 a 4 las hace el autor aparte.

Fuentes (todas en docs/results/v2/):
  5  demand_calibration/analysis_reference_failures_it5_it6_it7.json (it5, per_seed[].gap_per_signal)
  6  autoencoder/selection_analysis.json, selection_k96_analysis.json, selection_transformer_analysis.json,
     selection_tsmixer_analysis.json (sizes.<k>.delta y ci95)
  7  arch_comparison/analysis_three_architectures.json (per_arch.<arq>.mean_log_reward_mse_per_horizon)
  8  control/fidelity_onpolicy_validation.json (summary.<arq>.bias_by_longest_hold)
  9  planning/validation_analysis.json (planners.<brazo>.<H>.mean y replica_means)
  10 planning/test/*.csv (retorno por episodio), contrastado con planning/test_analysis.json
  11 planning/test/*.csv (medias por réplica e intersección), contrastado con planning/test_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_multiseed_statistical import t_two_sided_p  # noqa: E402  (numpy puro)

R = ROOT / "docs" / "results" / "v2"
SIGNALS = ("A0", "B0", "C0", "D0")
DPI = 300

# Luminosidades muy separadas (se distinguen en gris); el gris neutro se reserva para las reglas.
DARK, ORANGE, LIGHT, GRAY = "#1B3A5C", "#D55E00", "#56B4E9", "#7A7A7A"
INK, MUTED = "#222222", "#666666"
ARCH_STYLE = {  # color, marcador, estilo de línea, trama
    "lstm": (DARK, "o", "-", ""),
    "transformer": (ORANGE, "s", "--", "///"),
    "tsmixer": (LIGHT, "^", ":", "..."),
}
ARCH_LABEL = {"lstm": "LSTM", "transformer": "Transformer", "tsmixer": "TSMixer"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E3E3E3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "hatch.linewidth": 0.6,
    "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def load(rel: str):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def t_quantile(level: float, df: float) -> float:
    """Cuantil bilateral de la t de Student por bisección sobre el p-valor del proyecto (sin scipy)."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > 1 - level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), t_quantile(0.95, len(x) - 1) * float(x.std(ddof=1)) / math.sqrt(len(x))


def thousands(x, _pos=None) -> str:
    return f"{x:,.0f}".replace(",", " ").replace("-", "−")


def save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 5
def fig5(out: Path) -> Path:
    seeds = load("demand_calibration/analysis_reference_failures_it5_it6_it7.json")["it5"]["per_seed"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, s in enumerate(SIGNALS):
        g = np.array([p["gap_per_signal"][s] for p in seeds], dtype=float)
        m, h = mean_ci95(g)
        ax.bar(i, m, width=0.6, color=DARK if m > 0 else ORANGE, hatch="" if m > 0 else "///",
               edgecolor=INK, linewidth=0.6)
        ax.errorbar(i, m, yerr=h, color=INK, capsize=3, linewidth=0.9)
        wins = int((g > 0).sum())
        # semillas ganadas: encima del IC si la barra es positiva, sobre la línea del 0 si es negativa
        ax.annotate(f"{wins}/{len(g)}", (i, m + h if m > 0 else 0), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(4), SIGNALS)
    ax.set_xlabel("Intersección")
    ax.set_ylabel("Ventaja de cola_mas_larga sobre\nmin_verde_y_cambiar (recompensa)")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig5_ventaja_por_interseccion.png")


# ---------------------------------------------------------------- 6
def fig6(out: Path) -> Path:
    files = {"lstm": ["selection_analysis.json", "selection_k96_analysis.json"],
             "transformer": ["selection_transformer_analysis.json"],
             "tsmixer": ["selection_tsmixer_analysis.json"]}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ks = [16, 32, 48, 64, 80, 96]
    for j, (arch, fns) in enumerate(files.items()):
        sizes = {}
        for fn in fns:
            sizes.update(load(f"autoencoder/{fn}")["sizes"])
        assert sorted(map(int, sizes)) == ks, (arch, sorted(sizes))
        c, mk, ls, _ = ARCH_STYLE[arch]
        x = np.arange(len(ks)) + (j - 1) * 0.22
        d = np.array([sizes[str(k)]["delta"] for k in ks]) * 100
        lo = np.array([sizes[str(k)]["ci95"][0] for k in ks]) * 100
        hi = np.array([sizes[str(k)]["ci95"][1] for k in ks]) * 100
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt=mk, color=c, mfc=c if arch != "tsmixer" else "white",
                    mec=c if arch != "tsmixer" else INK, ms=5, capsize=2.5, linewidth=0.9, linestyle="none",
                    label=ARCH_LABEL[arch])
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("Tamaño del latente k")
    ax.set_ylabel("Δ relativo del error de recompensa,\ncomprimido frente a sin comprimir (%)")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:+.0f}".replace("-", "−"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return save(fig, out, "fig6_diferencia_relativa_por_k.png")


# ---------------------------------------------------------------- 7
def fig7(out: Path) -> Path:
    A = load("arch_comparison/analysis_three_architectures.json")
    hs = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for arch, v in A["per_arch"].items():
        c, mk, ls, _ = ARCH_STYLE[arch]
        gm = np.exp([v["mean_log_reward_mse_per_horizon"][str(h)] for h in hs])
        runs = [r for k, r in A["runs"].items() if k.startswith(arch + "_s")]
        assert len(runs) == 10
        ax.plot(hs, gm, color=c, marker=mk, linestyle=ls, linewidth=1.6, ms=4.5,
                mfc=c if arch != "tsmixer" else "white", mec=c if arch != "tsmixer" else INK,
                label=f"{ARCH_LABEL[arch]} ({v['n_params']:,} parámetros)".replace(",", " "))
    ax.set_xticks(hs)
    ax.set_xlabel("Horizonte h (pasos de 5 s)")
    ax.set_ylabel("Error cuadrático medio de la recompensa\n(media geométrica de 10 semillas)")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    return save(fig, out, "fig7_error_por_horizonte.png")


# ---------------------------------------------------------------- 8
def fig8(out: Path) -> Path:
    S = load("control/fidelity_onpolicy_validation.json")["summary"]
    buckets = ["0-5", "6-10", "11-19", ">=20"]
    labels = ["0–5", "6–10", "11–19", "≥ 20"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    for ax, arch in zip(axes, ("lstm", "transformer")):
        c, _, _, hatch = ARCH_STYLE[arch]
        b = [S[arch]["bias_by_longest_hold"][k]["bias_clipped"] for k in buckets]
        n = [S[arch]["bias_by_longest_hold"][k]["n_windows"] for k in buckets]
        ax.bar(range(4), b, width=0.6, color=c, hatch=hatch, edgecolor=INK, linewidth=0.6)
        for i, (bi, ni) in enumerate(zip(b, n)):
            ax.annotate(f"+{bi:,.0f}".replace(",", " "), (i, bi), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
            ax.annotate(f"n = {ni:,}".replace(",", " "), (i, 0), xytext=(0, -22), textcoords="offset points",
                        ha="center", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(4), labels)
        ax.text(0.03, 0.97, ARCH_LABEL[arch], transform=ax.transAxes, ha="left", va="top", fontsize=9,
                color=INK, fontweight="bold")
        ax.yaxis.set_major_formatter(thousands)
        ax.set_ylim(0, max(b) * 1.15)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Sesgo: retorno imaginado − real\n(7 pasos, recompensa recortada)")
    fig.supxlabel("Fase mantenida más larga entre los 4 semáforos al empezar a imaginar (pasos)", fontsize=9,
                  color=INK)
    fig.tight_layout(w_pad=2.0)
    return save(fig, out, "fig8_sesgo_por_retencion.png")


# ---------------------------------------------------------------- 9
def fig9(out: Path) -> Path:
    V = load("planning/validation_analysis.json")
    style = {"plan_ppo_lstm": (DARK, "o", "-", "Planificador + PPO, LSTM"),
             "plan_solo_lstm": (DARK, "o", ":", "Planificador solo, LSTM"),
             "plan_ppo_transformer": (ORANGE, "s", "--", "Planificador + PPO, Transformer"),
             "plan_solo_transformer": (ORANGE, "s", "-.", "Planificador solo, Transformer")}
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for j, (arm, (c, mk, ls, lab)) in enumerate(style.items()):
        byh = V["planners"][arm]
        hs = sorted(int(h) for h in byh)
        means = [byh[str(h)]["mean"] for h in hs]
        filled = arm.startswith("plan_ppo")
        ax.plot(hs, means, color=c, marker=mk, linestyle=ls, linewidth=1.5, ms=5,
                mfc=c if filled else "white", mec=c, label=lab)
        for h in hs:  # medias de las 3 réplicas
            reps = byh[str(h)]["replica_means"]
            assert len(reps) == 3 and abs(np.mean(reps) - byh[str(h)]["mean"]) < 0.06, (arm, h)
            ax.scatter(np.full(len(reps), h + (j - 1.5) * 0.06), reps, s=7, color=c, alpha=0.55, linewidths=0)
    ax.set_xticks([3, 5, 7])
    ax.set_xlabel("Horizonte de planificación H")
    ax.set_ylabel("Retorno medio por episodio\n(validación 24000–24023)")
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="lower left", fontsize=7.5)
    return save(fig, out, "fig9_validacion_por_H.png")


# ---------------------------------------------------------------- 10 y 11: episodios de la prueba final
def test_episodes() -> dict[str, list[dict]]:
    eps: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((R / "planning" / "test").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                eps[row["policy"]].append(row)
    T = load("planning/test_analysis.json")["arms"]
    assert set(eps) == set(T), (sorted(eps), sorted(T))
    for name, rows in eps.items():  # los CSV reproducen el análisis publicado
        r = np.array([float(x["reward"]) for x in rows])
        assert len(r) == T[name]["n_episodes"] and abs(r.mean() - T[name]["mean"]) < 0.06, name
    return eps


METHODS = [  # nombre, rótulo (los nombres del artículo), familia
    ("plan_ppo_transformer", "Planificador (plan_ppo), Transformer", "plan"),
    ("plan_ppo_lstm", "Planificador (plan_ppo), LSTM", "plan"),
    ("plan_solo_transformer", "Planificador (plan_solo), Transformer", "plan"),
    ("plan_solo_lstm", "Planificador (plan_solo), LSTM", "plan"),
    ("sueno_transformer", "PPO en imaginación sin planificar, Transformer", "sueno"),
    ("sueno_lstm", "PPO en imaginación sin planificar, LSTM", "sueno"),
    ("directo_30k", "PPO directo, 30.000 pasos", "directo"),
    ("directo_10k", "PPO directo, 10.000 pasos", "directo"),
    ("espera_mas_larga", "espera_mas_larga", "regla"),
    ("cola_mas_larga", "cola_mas_larga", "regla"),
    ("min_verde_y_cambiar", "min_verde_y_cambiar", "regla"),
    ("fijo_2_3", "Tiempo fijo", "regla"),
    ("max_presion", "Presión máxima", "regla"),
]
FAMILY = {"plan": (ORANGE, "///", "Planificador"), "sueno": (LIGHT, "...", "PPO en imaginación sin planificar"),
          "directo": (DARK, "", "PPO directo"), "regla": (GRAY, "xx", "Regla sin aprendizaje")}
# Estilo sencillo de la v1: colores planos, sin tramas (morado para el PPO directo, la cuarta familia).
FAMILY_V1 = {"plan": "#E67E22", "sueno": "#2E86AB", "directo": "#8E44AD", "regla": "#95A5A6"}
CLIP = -8000.0


def fig10(out: Path, eps, style: str = "article") -> Path:
    """Cajas del retorno por episodio en la prueba final. style: "article" (300 dpi, tramas) o "v1"."""
    v1 = style == "v1"

    def thousands_dot(x, _pos=None) -> str:  # punto como separador de miles, como en el artículo
        return f"{x:,.0f}".replace(",", ".").replace("-", "−")

    threshold = load("planning/test_analysis.json")["threshold"]
    rc = {"axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black",
          "xtick.color": "black", "ytick.color": "black", "grid.color": "#B0B0B0", "grid.alpha": 0.25,
          "grid.linewidth": 0.8} if v1 else {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.3, 4.6))
        n_m = len(METHODS)
        for i, (name, _lab, fam) in enumerate(METHODS):
            r = np.array([float(x["reward"]) for x in eps[name]])
            y = n_m - 1 - i
            c, hatch, _ = FAMILY[fam]
            if v1:
                c, hatch = FAMILY_V1[fam], ""
            bp = ax.boxplot(r, positions=[y], orientation="horizontal", widths=0.6, patch_artist=True, whis=1.5,
                            medianprops={"color": INK, "linewidth": 1.4},
                            whiskerprops={"color": INK, "linewidth": 0.8}, capprops={"color": INK, "linewidth": 0.8},
                            flierprops={"marker": "o", "ms": 2.2, "mfc": "none", "mec": MUTED, "mew": 0.5})
            for b in bp["boxes"]:
                b.set(facecolor=c, edgecolor=INK, linewidth=0.7, hatch=hatch)
            below = int((r < CLIP).sum())  # columna a la derecha del eje: episodios fuera del recorte
            ax.annotate(f"{below}/{len(r)}" if below else "—", (1.01, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=7, color=INK if below else MUTED)
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=0.9)
        ax.annotate(f"umbral de catastrófico ({thousands_dot(threshold)})", (threshold, n_m - 0.45),
                    xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK, va="bottom")
        ax.annotate(f"< {thousands_dot(CLIP)}", (1.01, n_m - 0.45), xycoords=("axes fraction", "data"),
                    va="bottom", ha="left", fontsize=7, color=MUTED)
        ax.set_yticks(range(n_m), [lab for _n, lab, _f in reversed(METHODS)])
        ax.set_xlim(CLIP, 0)
        ax.set_ylim(-0.6, n_m + 0.25 if v1 else n_m - 0.1)  # en la v1 hay borde arriba: espacio para el umbral
        ax.xaxis.set_major_formatter(thousands_dot)
        ax.set_xlabel(f"Retorno por episodio (prueba 25000–25047; eje recortado en {thousands_dot(CLIP)})")
        if v1:
            ax.grid(True)
        ax.grid(axis="y", visible=False)
        handles = [matplotlib.patches.Patch(facecolor=FAMILY_V1[k] if v1 else c, hatch="" if v1 else h,
                                            edgecolor=INK, linewidth=0.6, label=lab)
                   for k, (c, h, lab) in FAMILY.items()]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, fontsize=7.5)
        if v1:
            path = out / "fig10_cajas_prueba_final_v1.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            return path
        return save(fig, out, "fig10_cajas_prueba_final.png")


def fig11(out: Path, eps) -> Path:
    methods = [("plan_ppo_transformer", "Planificador + PPO, Transformer", ORANGE, "///"),
               ("directo_30k", "RL directo 30k", DARK, ""),
               ("espera_mas_larga", "espera_mas_larga (mejor regla en el total)", GRAY, "xx")]
    T = load("planning/test_analysis.json")["arms"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for j, (name, lab, c, hatch) in enumerate(methods):
        x = np.arange(4) + (j - 1) * 0.26
        means, errs = [], []
        for s in SIGNALS:
            by_rep: dict[str, list[float]] = defaultdict(list)
            for row in eps[name]:
                by_rep[row["seed_label"]].append(float(row[f"reward_{s}"]))
            rep_means = [np.mean(v) for v in by_rep.values()]
            m = float(np.mean([float(r[f"reward_{s}"]) for r in eps[name]]))
            assert abs(m - T[name]["per_signal"][s]) < 0.06, (name, s)
            means.append(m)
            errs.append(mean_ci95(rep_means)[1] if len(rep_means) > 1 else 0.0)
        ax.bar(x, means, width=0.26, color=c, hatch=hatch, edgecolor=INK, linewidth=0.6, label=lab)
        if any(errs):
            ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=INK, capsize=2, linewidth=0.8)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(4), SIGNALS)
    ax.set_xlabel("Intersección")
    ax.set_ylabel("Retorno medio por episodio\n(prueba 25000–25047)")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=7.5)
    return save(fig, out, "fig11_retorno_por_interseccion.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures_v2")
    parser.add_argument("--only", type=int, choices=[10], help="regenera solo esa figura (10: sus dos estilos)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eps = test_episodes()
    if args.only == 10:
        makers = (lambda: fig10(args.out, eps), lambda: fig10(args.out, eps, style="v1"))
    else:
        makers = (lambda: fig5(args.out), lambda: fig6(args.out), lambda: fig7(args.out), lambda: fig8(args.out),
                  lambda: fig9(args.out), lambda: fig10(args.out, eps), lambda: fig10(args.out, eps, style="v1"),
                  lambda: fig11(args.out, eps))
    for make in makers:
        path = make()
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
