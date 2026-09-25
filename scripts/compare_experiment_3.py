"""Compare Experimento 3: LSTM vs. Transformer vs. TSMixer as the temporal model.

Same protocol as Experimento 0 (scripts/compare_experiment_0.py): the three
models were trained on the same latent dataset, with the same seeds, optimizer,
early stopping and shared reward_scaler.json, and evaluated on the same test
split and horizons. reward_mse is the deciding metric; latent_mse is shown only
as context (here it IS comparable, all three predict the same z, but the
reward is what the Dream Environment feeds to the controller).

Decision criterion (fixed before seeing any result):

- Per horizon, the architecture with the lowest reward_mse wins.
- A candidate (Transformer or TSMixer) is recommended to replace the LSTM ONLY
  if BOTH hold:
    1. it wins a strict majority of the horizons, and
    2. its median relative reduction of reward_mse vs. the LSTM is of a
       magnitude similar to the one that justified keeping the Autoencoder in
       Experimento 0 (at least SIMILAR_MAGNITUDE_FRACTION of it).
- If instead the LSTM itself wins a strict majority of the horizons against
  both candidates, it is kept because it predicts better; its integration in
  the pipeline is reported as an additional argument, not the main one.
- Otherwise (genuinely split result, no clear winner) the recommendation is to
  keep the LSTM, because it is already integrated and validated across the
  whole pipeline (Dream Environment, checkpoint selection in real SUMO, PPO
  verified with 3 seeds): replacing it means repeating that whole verification
  chain for the new candidate, not just one more training run. It is not
  "the simplest" by parameter count -- TSMixer is smaller.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.world_model_evaluation import build_world_model

RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"

# (name, evaluation report, checkpoint hyperparameters). LSTM first: it is the
# incumbent every candidate is measured against.
ARCHITECTURES = [
    ("LSTM", RESULTS_DIR / "world_model_evaluation.json", CHECKPOINT_DIR / "world_model_best.json"),
    (
        "Transformer",
        RESULTS_DIR / "world_model_transformer_evaluation.json",
        CHECKPOINT_DIR / "world_model_transformer_best.json",
    ),
    (
        "TSMixer",
        RESULTS_DIR / "world_model_tsmixer_evaluation.json",
        CHECKPOINT_DIR / "world_model_tsmixer_best.json",
    ),
]
INCUMBENT = "LSTM"

# Median relative reduction of reward_mse of the Autoencoder (z) vs. the raw
# state in Experimento 0, second asymmetric round (PROJECT_STATUS.md, "Experimento
# 0 repetido con el dataset nuevo"): per-horizon reductions 10.1, 8.9, 13.2,
# 16.5, 16.2, 19.1, 20.5, 17.0, 6.2, -6.0 % -> median 14.7 %. Frozen here as a
# constant because re-running the LSTM evaluation would otherwise move the
# reference this experiment is judged against.
EXPERIMENT_0_MEDIAN_REDUCTION = 0.147
SIMILAR_MAGNITUDE_FRACTION = 0.5
MIN_REDUCTION = EXPERIMENT_0_MEDIAN_REDUCTION * SIMILAR_MAGNITUDE_FRACTION


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding evaluate_world_model*.py script first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(hparams_path: Path) -> int:
    if not hparams_path.exists():
        raise FileNotFoundError(
            f"Missing {hparams_path}. Run the corresponding train_world_model*.py script first."
        )
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    model = build_world_model(hparams)
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    names = [name for name, _, _ in ARCHITECTURES]
    reports = {name: load_report(report_path) for name, report_path, _ in ARCHITECTURES}
    parameters = {name: count_parameters(hparams_path) for name, _, hparams_path in ARCHITECTURES}

    # Guard: the comparison is only fair if all three reports come from the
    # same test split -- same horizons and same number of scored windows.
    reference = reports[INCUMBENT]
    for name in names:
        if reports[name].keys() != reference.keys():
            raise ValueError(f"{name} report has different horizons than the {INCUMBENT} report.")
        for h in reference:
            if reports[name][h]["n_samples"] != reference[h]["n_samples"]:
                raise ValueError(
                    f"{name} report has n_samples={reports[name][h]['n_samples']} at h={h}, "
                    f"{INCUMBENT} has {reference[h]['n_samples']}: not the same test set."
                )

    horizons = sorted(int(h) for h in reference.keys())

    header = f"{'h':>3} | " + " | ".join(f"{f'reward_mse ({n})':>24}" for n in names) + f" | {'gana':>11}"
    print(header)
    print("-" * len(header))

    wins = {name: 0 for name in names}
    reductions = {name: [] for name in names if name != INCUMBENT}
    # The reverse view, used only when the LSTM wins: how much lower the LSTM's
    # reward_mse is, relative to each candidate's (bounded by 100%).
    incumbent_reductions = {name: [] for name in reductions}
    for h in horizons:
        row = {name: reports[name][str(h)]["model_reward_mse"] for name in names}
        winner = min(names, key=lambda name: row[name])
        wins[winner] += 1
        for name in reductions:
            reductions[name].append((row[INCUMBENT] - row[name]) / row[INCUMBENT])
            incumbent_reductions[name].append((row[name] - row[INCUMBENT]) / row[name])

        print(f"{h:3d} | " + " | ".join(f"{row[n]:24.3f}" for n in names) + f" | {winner:>11}")

    print()
    summary_header = (
        f"{'arquitectura':>12} | {'parametros':>10} | {'horizontes ganados':>18} | "
        f"{'reduccion mediana vs LSTM':>25} | {'latent_mse h=1*':>15}"
    )
    print(summary_header)
    print("-" * len(summary_header))
    median_reduction = {name: statistics.median(values) for name, values in reductions.items()}
    for name in names:
        reduction_text = "--" if name == INCUMBENT else f"{median_reduction[name] * 100:+.1f}%"
        print(
            f"{name:>12} | {parameters[name]:>10,} | {wins[name]:>11}/{len(horizons):<6} | "
            f"{reduction_text:>25} | {reports[name]['1']['model_latent_mse']:15.6f}"
        )

    print()
    print("* latent_mse se muestra solo como contexto; la decision usa reward_mse.")
    print(
        f"Umbral de magnitud: reduccion mediana >= {MIN_REDUCTION * 100:.1f}% "
        f"({SIMILAR_MAGNITUDE_FRACTION:.0%} de la del Experimento 0, "
        f"{EXPERIMENT_0_MEDIAN_REDUCTION * 100:.1f}%)."
    )
    print()

    majority = len(horizons) // 2 + 1
    qualified = [
        name
        for name in reductions
        if wins[name] >= majority and median_reduction[name] >= MIN_REDUCTION
    ]
    if qualified:
        chosen = max(qualified, key=lambda name: median_reduction[name])
        print(
            f"DECISION: {chosen} gana {wins[chosen]}/{len(horizons)} horizontes con una "
            f"reduccion mediana de reward_mse de {median_reduction[chosen] * 100:.1f}% frente a "
            f"la LSTM -- SE RECOMIENDA como reemplazo de la LSTM."
        )
    elif wins[INCUMBENT] >= majority:
        incumbent_median = {
            name: statistics.median(values) for name, values in incumbent_reductions.items()
        }
        versus = " y ".join(
            f"{incumbent_median[name] * 100:.1f}% frente a {name}" for name in incumbent_median
        )
        print(
            f"DECISION: LSTM gana en {wins[INCUMBENT]}/{len(horizons)} horizontes frente a ambas "
            f"alternativas, con reduccion mediana de reward_mse de {versus} -- SE MANTIENE "
            "porque predice mejor, con el argumento adicional de que ya esta integrada y "
            "validada en todo el pipeline (Dream Environment, seleccion de checkpoint por "
            "SUMO real, y PPO verificado con 3 semillas)."
        )
    else:
        for name in reductions:
            reasons = []
            if wins[name] < majority:
                reasons.append(f"gana solo {wins[name]}/{len(horizons)} horizontes (mayoria: {majority})")
            if median_reduction[name] < MIN_REDUCTION:
                reasons.append(
                    f"reduccion mediana {median_reduction[name] * 100:+.1f}% < {MIN_REDUCTION * 100:.1f}%"
                )
            print(f"  {name}: no califica -- " + "; ".join(reasons) + ".")
        print(
            "DECISION: resultado mixto o diferencias marginales -- SE MANTIENE la LSTM, "
            "por estar ya integrada y validada en todo el pipeline (Dream Environment, "
            "seleccion de checkpoint por SUMO real, y PPO verificado con 3 semillas) -- "
            "reemplazarla exige repetir esa cadena completa de verificacion para el "
            "candidato nuevo, no solo un entrenamiento mas."
        )


if __name__ == "__main__":
    main()
