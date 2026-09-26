> Documento completo de la propuesta académica disponible en `docs/PROPUESTA.md`.
> Consúltalo para el detalle exacto de cualquier sección citada aquí (por ejemplo, Sección 15, 20, 30).

# CLAUDE.md — Contexto permanente del proyecto

## Objetivo académico

Trabajo de grado de Ingeniería de Sistemas (Universidad del Quindío). Implementar un
**World Model** (Ha & Schmidhuber, 2018) para control inteligente de semáforos en una
intersección simulada con SUMO, comparando este enfoque contra Reinforcement Learning
directo sobre el simulador.

Pregunta de investigación: ¿puede un modelo aprendido de la dinámica del tráfico reducir
las interacciones necesarias con SUMO, sin perder desempeño de control?

## Papers de referencia

- Ha, D., & Schmidhuber, J. (2018). *World Models*. — arquitectura base (V+M+C).
- Hafner, D., et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*.
  — referencia conceptual para el Dream Environment.
- Alegre, L. N. *SUMO-RL* (github.com/LucasAlegre/sumo-rl) — infraestructura de
  simulación reutilizada, no una técnica de modelado propia del proyecto.

Nota: **no se implementa** la Mixture Density Network (MDN-RNN) del paper original, ni
CMA-ES para el controlador — decisión de diseño explícita, documentada en la Sección 30
de la propuesta (auditoría tecnológica curricular).

## Arquitectura general (estado real, no aspiracional)

```
SUMO → TrafficEnvironment → CustomStateBuilder → TrafficState (vector de 26)
     → dataset (collect → split por episodio → normalize) → TransitionDataset
     → Autoencoder (Encoder/Decoder) → z                                        [COMPLETO, entrenado]
     → encode_latent_dataset.py → *_latent.npz                                   [COMPLETO]
     → LatentSequenceDataset (ventanas por episodio)                             [COMPLETO]
     → LatentDynamicsLSTM (recibe z_t + a_t, predice ẑ_{t+1} y r̂_{t+1})         [COMPLETO, entrenado]
     → evaluation/world_model_evaluation.py (Experimento 1: modelo vs. baseline
       persistente, a 1 y varios pasos, con compounding error)                   [COMPLETO]
     → DreamEnvironment (imagina transiciones con el LSTM, sin SUMO)             [COMPLETO]
     → PPO del sueño (selección del checkpoint por evaluación en SUMO real,
       vía EncodedTrafficEnvironment)                                            [COMPLETO, 3 semillas]
     → PPO directo contra SUMO real (baseline de RL directo)                     [COMPLETO, 3 semillas]
     → evaluate_final_comparison.py (sueño vs. directo vs. tiempo fijo vs.
       regla trivial, en SUMO real)                                              [COMPLETO]
```

Experimentos cerrados: 0 (el Autoencoder se mantiene, 9/10 horizontes), 1 (el LSTM supera
al baseline persistente en 10/10), 2 (control en SUMO real, ver el resultado central) y
3 (Transformer y TSMixer, ver abajo).

Transformer y TSMixer (`models/world_model/transformer.py`, `tsmixer.py`) están
**implementados y evaluados** como sustitutos intercambiables del LSTM (misma interfaz,
`models/world_model/base.py::TemporalModel`). **No reemplazan al LSTM**: el LSTM tiene el
menor `reward_mse` en los 10 horizontes (reducción mediana de 38.1% frente al Transformer
y 56.1% frente a TSMixer).

## Resultado central (preciso)

Con el presupuesto original del RL directo (10,000 pasos de entrenamiento, 13,000
interacciones reales por semilla), el World Model controla mejor: -326.79 frente a
-453.74 (media de 3 semillas × 30 episodios), y es mucho más consistente entre semillas.
Ese presupuesto ya era ~2.8 veces el del World Model (~4,600 interacciones por semilla).
Con el triple de presupuesto (30,000 pasos, 39,000 interacciones por semilla), el RL
directo alcanza un desempeño comparable (-335.24; p = 0.71 a nivel de semilla), pero con
**~8.5 veces las interacciones reales del World Model**. La ventaja demostrada del World
Model es de **eficiencia en interacciones reales**, no de mejor control sin límite de
presupuesto. Detalle completo en PROJECT_STATUS.md.

## Decisiones de diseño ya tomadas

- El estado (`TrafficState`) tiene **26 dimensiones**: 4 carriles × 5 variables
  (vehicle_counts, queue_lengths, waiting_times, mean_speeds, occupancies) + 4 de
  one-hot de fase del semáforo + 2 (elapsed/remaining phase time).
- La fase del semáforo y la acción del controlador se codifican **one-hot**
  (`action_dim = 2`).
- La acción es el **índice de la fase verde de destino**, no un interruptor
  mantener/cambiar (`sumo_rl.TrafficSignal.set_next_phase`). Con 2 fases, la acción 1
  equivale a "cambiar" solo cuando la fase actual es la 0, alrededor de la mitad de los
  pasos. Documentado en el código (commit `c253d88`), **sin cambio de comportamiento**:
  todo el pipeline usa esta convención de forma consistente.
- `info["phase_change"]` mide si se **pidió la fase 1** (`action == 1`), no si el
  semáforo cambió de fase de verdad; `info["phase_switched"]` mide el cambio real. La
  recompensa penaliza una u otra según `RewardConfig.phase_penalty`: el valor por
  defecto, `"requested_phase_1"`, es la definición de todos los datos y modelos
  entrenados, y **no debe cambiarse sin decidirlo explícitamente** (invalidaría todo lo
  entrenado). `"actual_switch"` existe como opción; pasar a ella por defecto es la
  opción B pendiente en TODO.md.
- El "throughput" en la recompensa se mide como vehículos que **llegaron a destino**
  (`traci.simulation.getArrivedNumber()`, delta entre pasos) — nunca vehículos
  presentes en el carril.
- Split de datos **por episodio completo**, nunca por transición suelta.
- Normalización de estados ajustada **solo con el split de entrenamiento**.
- `RepresentationConfig.input_dim` **no tiene valor por defecto** — siempre se obtiene
  del tamaño real del estado.
- `WorldModelConfig.latent_dim` deriva de `RepresentationConfig.latent_dim` — única
  fuente de verdad, con validación de consistencia si se pasan ambos.
- Episodios más cortos que `sequence_length` se **descartan**, nunca se rellenan.
- El `LatentDynamicsLSTM` predice **dos salidas**: `(ẑ_{t+1}, r̂_{t+1})` — la predicción
  de recompensa es parte del núcleo según la Sección 15 de la propuesta.
- Cada checkpoint entrenado (`.pt`) guarda junto a sí un `.json` con los hiperparámetros
  exactos usados.
- Semillas fijas (`torch`, `numpy`, `cuda`) en todo entrenamiento.
- La evaluación del World Model (Experimento 1) compara siempre contra un **baseline
  persistente** ("nada cambia respecto al último valor real observado"), usando las
  **acciones reales** del episodio, no acciones imaginadas — evaluar con acciones
  hipotéticas es responsabilidad del Dream Environment, no de esta evaluación.
- La pérdida de entrenamiento del modelo temporal **normaliza la recompensa** (con
  `reward_scaler.json`, ajustado solo con el split de entrenamiento) antes de combinarla
  con la pérdida latente; sin esto, el término de recompensa dominaba el entrenamiento.
  Transformer y TSMixer **leen** ese mismo archivo, nunca lo recalculan.
- Criterio de decisión del Experimento 3, fijado antes de ver resultados: un candidato
  reemplaza al LSTM solo si tiene el menor `reward_mse` en la **mayoría de los
  horizontes** y su **reducción mediana de `reward_mse` es ≥ 7.3%** (la mitad del 14.7%
  que justificó mantener el Autoencoder en el Experimento 0). La decisión se toma por
  `reward_mse` en rollouts, nunca por la pérdida de validación: el Transformer tiene
  mejor pérdida de validación y mejor `latent_mse`, pero peor `reward_mse`.
- Contabilidad de interacciones reales: el dataset del World Model (4,800 transiciones)
  se recolecta una vez y lo comparten las 3 semillas (~4,600 por semilla con la
  selección en SUMO real); la evaluación periódica del RL directo cuenta como
  interacción real (13,000 por semilla con 10k pasos, 39,000 con 30k). No describir la
  comparación con 10k como "a presupuesto comparable": el RL directo ya tenía ~2.8
  veces más interacciones que el World Model, y el World Model ganaba con **menos**.
- Los resultados de controladores se reportan como media de las 3 semillas de
  entrenamiento (90 episodios en SUMO real), nunca con una sola semilla.

## Decisiones que NO deben cambiarse sin consultar primero

- No agregar MDN-RNN ni CMA-ES.
- No usar CNN (el estado es un vector, no una imagen).
- No usar embeddings de tokens discretos para el estado.
- No cambiar el criterio de episodios cortos (descartar, no rellenar) sin discutirlo.
- No commitear datasets generados ni pesos de checkpoint (`.npz`, `.pt`, `.zip`, `.pkl`)
  — sí se permiten los `.json` de hiperparámetros junto a cada checkpoint (son pequeños
  y documentan qué configuración produjo cada resultado).
- No evaluar el modelo temporal (Experimento 1) con acciones imaginadas — eso es del
  Dream Environment, una etapa separada.
- Los checkpoints **oficiales** son los de `models/checkpoints/controller/` (PPO del
  sueño, semilla 1) y `models/checkpoints/controller_direct/` (PPO directo, 10,000 pasos,
  semilla 0). La verificación de 30,000 pasos en `controller_direct_30k/` es una
  **referencia**: no reemplaza a ningún checkpoint oficial sin una decisión explícita.
- El modelo temporal del sistema es el LSTM (`world_model_best.pt`): el Dream
  Environment y los PPO dependen de él. No reentrenarlo ni sobrescribirlo sin consultar.

## Estructura del repositorio

```
configs/        EnvironmentConfig, RewardConfig, RepresentationConfig, WorldModelConfig,
                ControllerConfig
datasets/       transition_dataset.py, latent_sequence_dataset.py, metadata.json, raw/, processed/ (generados)
docs/           PROPUESTA.md (propuesta académica), DOCUMENTACION_PROYECTO.md (documento de estudio)
environments/   TrafficEnvironment, CustomStateBuilder, TrafficState, ProjectActionSpace,
                ProjectRewardFunction, DreamEnvironment, EncodedTrafficEnvironment,
                ReseedingWrapper, single-intersection/ (red SUMO propia, demanda asimétrica)
models/
  representation/  Encoder, Decoder, Autoencoder
  world_model/     base.py (Protocol TemporalModel), lstm.py (LatentDynamicsLSTM, el del sistema),
                   transformer.py, tsmixer.py (Experimento 3)
  checkpoints/     controller/ y controller_direct/ (oficiales), controller_direct_30k/ (referencia),
                   raw_state/ (Experimento 0); pesos no versionados, sus .json sí
training/       train_autoencoder.py, train_world_model.py, train_world_model_raw.py,
                train_world_model_transformer.py, train_world_model_tsmixer.py,
                train_controller.py (PPO del sueño), train_controller_direct.py (RL directo)
evaluation/     autoencoder_evaluation.py, evaluate_autoencoder.py,
                world_model_evaluation.py (incluye build_world_model)
scripts/        datos: collect_dataset.py, split_dataset.py, merge_dataset.py, normalize_dataset.py,
                  visualize_dataset.py, encode_latent_dataset.py, test_environment.py
                Exp. 0/1: prepare_raw_sequence_dataset.py, evaluate_world_model.py,
                  evaluate_world_model_raw.py, compare_experiment_0.py
                Exp. 3: evaluate_world_model_transformer.py, evaluate_world_model_tsmixer.py,
                  compare_experiment_3.py
                control: evaluate_controller.py, evaluate_controller_sumo.py,
                  evaluate_direct_vs_dream.py, evaluate_final_comparison.py,
                  analyze_controller_actions.py
tests/          13 archivos, 67 tests (pytest -v)
ver_controlador.py (demo del PPO del sueño en la GUI de SUMO), CLAUDE.md, PROJECT_STATUS.md,
TODO.md, README.md, pytest.ini, requirements.txt, .gitignore, LICENSE
```

No existen (a propósito): `utils/`, `notebooks/`, `experiments/`, `papers/`.

## Convenciones de trabajo establecidas en este proyecto

- Auditoría antes de avanzar: cada bloque grande se cierra con revisión línea por línea
  contra el repositorio real (`git clone`, no tarball ni narración) antes de pasar al
  siguiente.
- Commits pequeños, una responsabilidad por commit, verificados con `pytest -v` antes
  de subir.
- Diseño explicado y aprobado antes de escribir código.
- Nunca asumir que "corrió sin error" significa "está bien" — se exige inspeccionar los
  números reales de la salida (curvas de pérdida, métricas de evaluación), no solo que
  el script termine.

## Pendiente de verificar (no confirmado, no inventar)

- Nada abierto en este momento.

Resuelto: la cuenta/app desconocida que apareció como colaborador en el repositorio de
GitHub era Codex Connector; su acceso se revocó en ambas capas, y los colaboradores
humanos son legítimos (investigado y confirmado por el autor directamente en GitHub).
