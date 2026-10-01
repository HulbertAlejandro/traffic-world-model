"""
Genera la Tabla 4a del artículo: costo computacional de LSTM, Transformer y TSMixer,
medido en el mismo equipo. No modifica nada oficial; solo lee checkpoints ya entrenados
y mide tiempos.

Ejecutar desde la raíz del repositorio:
    python scripts/benchmark_temporal_models.py

Requiere que existan los checkpoints oficiales:
    models/checkpoints/world_model_best.pt
    models/checkpoints/world_model_transformer_best.pt
    models/checkpoints/world_model_tsmixer_best.pt

Un "paso" es una llamada al modelo con una ventana de sequence_length pasos que devuelve
(z_{t+1}, r_{t+1}): es lo que el Dream Environment ejecuta por cada paso imaginado.

Medición estable en una CPU híbrida (núcleos P y E): torch usa 1 hilo (fijo y registrado) y las
tres arquitecturas se miden intercaladas en N_ROUNDS rondas, para que todas enfrenten las mismas
condiciones de carga. Se reporta la mediana por ronda y el rango intercuartílico.
"""
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.world_model_evaluation import load_world_model  # noqa: E402

CKPTS = {
    "LSTM": ROOT / "models/checkpoints/world_model_best.pt",
    "Transformer": ROOT / "models/checkpoints/world_model_transformer_best.pt",
    "TSMixer": ROOT / "models/checkpoints/world_model_tsmixer_best.pt",
}
DEVICE = torch.device("cpu")
N_WARMUP, N_TIMED, N_ROUNDS = 20, 200, 15
TORCH_THREADS = 1
MAX_REL_IQR = 0.10  # a measurement whose IQR exceeds 10% of its median is flagged unstable (CPU contention)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_inputs(hp, batch_size):
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(batch_size, hp["sequence_length"], hp["latent_dim"], generator=gen)
    a = torch.randint(0, hp["action_dim"], (batch_size, hp["sequence_length"]), generator=gen)
    return z, torch.nn.functional.one_hot(a, hp["action_dim"]).float()


def time_calls(model, z, a):
    """ms per call over N_TIMED calls."""
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(N_TIMED):
            model(z, a)
        return 1000 * (time.perf_counter() - start) / N_TIMED


def quartiles(x):
    q = statistics.quantiles(x, n=4)
    return statistics.median(x), q[0], q[2]


def main():
    torch.set_num_threads(TORCH_THREADS)
    models = {}
    for name, ckpt in CKPTS.items():
        model, hp = load_world_model(ckpt, DEVICE)
        model.eval()
        inputs = {b: make_inputs(hp, b) for b in (1, 64)}
        with torch.no_grad():
            for b in (1, 64):
                for _ in range(N_WARMUP):
                    model(*inputs[b])
        models[name] = (ckpt, model, inputs)

    ms = {name: {1: [], 64: []} for name in models}
    for _ in range(N_ROUNDS):  # interleaved: every round measures every model under the same load
        for name, (_, model, inputs) in models.items():
            for b in (1, 64):
                ms[name][b].append(time_calls(model, *inputs[b]))

    rows = {}
    for name, (ckpt, model, _) in models.items():
        m1, q1a, q1b = quartiles(ms[name][1])
        m64, q64a, q64b = quartiles(ms[name][64])
        rows[name] = {
            "checkpoint": ckpt.relative_to(ROOT).as_posix(),
            "parametros": count_params(model),
            "latencia_1_paso_lote_1_ms": round(m1, 4), "latencia_1_paso_lote_1_ms_iqr": [round(q1a, 4), round(q1b, 4)],
            "pasos_por_segundo_lote_1": round(1000 / m1, 1),
            "latencia_llamada_lote_64_ms": round(m64, 4), "latencia_llamada_lote_64_ms_iqr": [round(q64a, 4), round(q64b, 4)],
            "pasos_por_segundo_lote_64": round(64 * 1000 / m64, 1),
            "pasos_por_segundo_lote_64_iqr": [round(64 * 1000 / q64b, 1), round(64 * 1000 / q64a, 1)],
        }
        r = rows[name]
        r["estable"] = bool((q1b - q1a) / m1 <= MAX_REL_IQR and (q64b - q64a) / m64 <= MAX_REL_IQR)
        print(f"{name:12s} params={r['parametros']:>7d}  lat(b=1)={m1:.3f} ms [{q1a:.3f}, {q1b:.3f}]  "
              f"llamada(b=64)={m64:.3f} ms [{q64a:.3f}, {q64b:.3f}]  pasos/s(b=64)={r['pasos_por_segundo_lote_64']:.0f}"
              f"  {'estable' if r['estable'] else 'INESTABLE (IQR > 10%): repetir'}")

    meta = {"device": "cpu", "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor(), "python": platform.python_version(),
            "n_warmup": N_WARMUP, "n_timed_calls_per_round": N_TIMED, "n_rounds_interleaved": N_ROUNDS}
    out = ROOT / "docs/results/benchmark_temporal_models.json"
    out.write_text(json.dumps({"meta": meta, "models": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out}")
    print("Nota: mide inferencia (el costo del Dream Environment), no tiempo de entrenamiento: no hay logs")
    print("oficiales con tiempos por epoca (los .json de los checkpoints oficiales no los registran).")


if __name__ == "__main__":
    main()
