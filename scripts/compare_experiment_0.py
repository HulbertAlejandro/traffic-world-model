"""Compare Experimento 0: LatentDynamicsLSTM on z (Autoencoder) vs. on raw state.

Decision criterion (agreed before implementation): reward_mse is the only metric
directly comparable in absolute magnitude between the two experiments, since the
reward is the same physical quantity regardless of which representation feeds the
LSTM -- the latent/state error, by contrast, lives in two different spaces
(16-dim learned latent vs. 26-dim normalized raw state) with different natural
scales, so comparing their raw magnitude is not apples-to-apples. Latent/state
error is still reported for context, never as the deciding number.

The Autoencoder is kept in the final system ONLY if it measurably and
consistently improves reward_mse across horizons versus the raw-state model --
matching the criterion already written in the project proposal (Sección 24):
"El Autoencoder/VAE debe mantenerse solo si demuestra utilidad para la
representación."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "results"
Z_REPORT_PATH = RESULTS_DIR / "world_model_evaluation.json"
RAW_REPORT_PATH = RESULTS_DIR / "world_model_raw_evaluation.json"


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding evaluate_world_model*.py script first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    z_report = load_report(Z_REPORT_PATH)
    raw_report = load_report(RAW_REPORT_PATH)

    horizons = sorted(int(h) for h in z_report.keys())

    print(f"{'h':>3} | {'reward_mse (z)':>15} | {'reward_mse (raw)':>17} | {'z gana?':>8} | "
          f"{'state_mse (z)*':>15} | {'state_mse (raw)*':>17}")
    print("-" * 92)

    z_wins_reward = 0
    for h in horizons:
        z_row = z_report[str(h)]
        raw_row = raw_report[str(h)]

        z_reward = z_row["model_reward_mse"]
        raw_reward = raw_row["model_reward_mse"]
        z_wins = z_reward < raw_reward
        z_wins_reward += int(z_wins)

        print(f"{h:3d} | {z_reward:15.3f} | {raw_reward:17.3f} | {'z' if z_wins else 'crudo':>8} | "
              f"{z_row['model_latent_mse']:15.3f} | {raw_row['model_latent_mse']:17.3f}")

    print()
    print("* state_mse/latent_mse NO son directamente comparables entre sí (espacios")
    print("  distintos, 16 vs. 26 dimensiones) -- se muestran solo como contexto.")
    print()
    print(f"El Autoencoder (z) tuvo menor reward_mse en {z_wins_reward}/{len(horizons)} horizontes.")

    if z_wins_reward >= len(horizons) * 0.8:
        print("DECISION: el Autoencoder aporta valor consistente -- SE MANTIENE en el sistema final.")
    elif z_wins_reward <= len(horizons) * 0.2:
        print("DECISION: el estado crudo predice la recompensa mejor de forma consistente -- "
              "el Autoencoder SE DESCARTA para esta etapa del pipeline.")
    else:
        print("DECISION: resultado mixto, sin ventaja consistente de ninguno de los dos -- "
              "revisar manualmente antes de decidir (ver Sección 24 de la propuesta: "
              "el Autoencoder se mantiene solo si demuestra utilidad clara).")


if __name__ == "__main__":
    main()
