"""Figuras de la v2 en el estilo simple de la v1, solo a partir de resultados guardados.

    python scripts/v2/make_figures_estilo_v1.py [--out docs/figures_v2/estilo_v1]

Solo lectura: no simula, no entrena y no modifica ningún resultado. Depende de json, numpy y matplotlib.
Estilo de la v1 (`evaluation/world_model_evaluation.py::plot_compounding_error`): líneas con marcadores,
cuadrícula suave, título por panel, LSTM azul, Transformer naranja, TSMixer morado. PNG a 170 dpi, sin
título de figura (el pie va en el artículo).

Fuentes (todas versionadas, en models/checkpoints/v2/arch_comparison/<arq>_raw_s<i>/):
  - history.json: (pérdida de entrenamiento, pérdida de validación) por época. Es la pérdida de la v1:
    MSE del estado normalizado + MSE de la recompensa normalizada (`training/v2_compression_experiment.py`,
    `train_temporal_model`).
  - world_model_best.json: best_epoch y stopped_epoch.
  - evaluation.json: reward_mse y baseline_reward_mse (baseline persistente de la v1, `rollout_episode`)
    por horizonte, en el test del dataset 22000-22023.

No hay figura de pérdidas del Autoencoder: en la v2 su entrenamiento solo guarda la mejor época y su MSE
(`train_autoencoder` escribe `autoencoder_best.json`), no la curva por época.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "models" / "checkpoints" / "v2" / "arch_comparison"
SEEDS = range(10)
HORIZONS = range(1, 11)
DPI = 170
ARCHS = {  # nombre, color, marcador
    "lstm": ("LSTM", "tab:blue", "o"),
    "transformer": ("Transformer", "tab:orange", "s"),
    "tsmixer": ("TSMixer", "tab:purple", "^"),
}


def load(arch: str, seed: int, name: str):
    return json.loads((RUNS / f"{arch}_raw_s{seed}" / name).read_text(encoding="utf-8"))


def fig_perdidas(out: Path) -> Path:
    """(a)-(c): pérdida por época de la semilla 0; en tenue, la validación de las semillas 1-9."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, letter, (arch, (label, color, _mk)) in zip(axes, "abc", ARCHS.items()):
        for seed in SEEDS:
            hist = np.asarray(load(arch, seed, "history.json"), dtype=float)
            meta = load(arch, seed, "world_model_best.json")
            assert len(hist) == meta["stopped_epoch"], (arch, seed)
            assert int(np.argmin(hist[:, 1])) + 1 == meta["best_epoch"], (arch, seed)
            epochs = np.arange(1, len(hist) + 1)
            if seed == 0:
                ax.plot(epochs, hist[:, 0], color=color, label="Entrenamiento (semilla 0)")
                ax.plot(epochs, hist[:, 1], color=color, linestyle="--", label="Validación (semilla 0)")
                ax.axvline(meta["best_epoch"], color="gray", linestyle=":", linewidth=1,
                           label=f"Mejor época (semilla 0): {meta['best_epoch']}")
            else:
                ax.plot(epochs, hist[:, 1], color=color, alpha=0.18, linewidth=0.8,
                        label="Validación (semillas 1–9)" if seed == 1 else None)
        ax.set_title(f"({letter}) {label}")
        ax.set_xlabel("Época")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pérdida MSE (estado + recompensa, normalizados)")
    fig.tight_layout()
    path = out / "perdidas_modelos_temporales.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_horizonte(out: Path) -> Path:
    """(a) error de recompensa por horizonte frente al baseline persistente; (b) reducción frente a él."""
    baseline = load("lstm", 0, "evaluation.json")["evaluation"]["baseline_reward_mse"]
    for arch in ARCHS:  # el baseline no depende del modelo: el mismo en las 30 corridas
        for seed in SEEDS:
            assert load(arch, seed, "evaluation.json")["evaluation"]["baseline_reward_mse"] == baseline
    hs = list(HORIZONS)
    base = np.array([baseline[str(h)] for h in hs])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for arch, (label, color, mk) in ARCHS.items():
        mse = np.array([[load(arch, s, "evaluation.json")["evaluation"]["reward_mse"][str(h)] for h in hs]
                        for s in SEEDS])
        gm = np.exp(np.log(mse).mean(axis=0))  # media geométrica entre semillas, la escala de la métrica
        axes[0].plot(hs, gm, marker=mk, color=color, label=f"{label} (10 semillas)")
        axes[1].plot(hs, 100 * (1 - gm / base), marker=mk, color=color, label=label)
    axes[0].plot(hs, base, marker="x", linestyle="--", color="gray", label="Baseline persistente")
    axes[0].set_title("(a) Error de predicción de recompensa vs. horizonte")
    axes[0].set_ylabel("MSE de recompensa")
    axes[1].set_title("(b) Reducción del error frente al baseline persistente")
    axes[1].set_ylabel("Reducción del MSE de recompensa (%)")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Horizonte (pasos)")
        ax.set_xticks(hs)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    path = out / "error_recompensa_vs_baseline.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures_v2" / "estilo_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in (fig_perdidas(args.out), fig_horizonte(args.out)):
        print(path)


if __name__ == "__main__":
    main()
