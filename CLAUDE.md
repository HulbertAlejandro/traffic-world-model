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
     → PPO directo contra SUMO real (baseline de RL directo)                     [COMPLETO, 4 semillas
                                                                                  × 10k y 30k pasos]
     → evaluate_multiseed_statistical.py (sueño vs. directo vs. tiempo fijo vs.
       regla trivial, N semillas, escenarios oficiales y nuevos, en SUMO real)   [COMPLETO]
```

Experimentos cerrados: 0 (el Autoencoder mejora la predicción de la recompensa de forma
moderada y mayoritaria: 5 semillas por rama con convergencia igualada, 37/50 pares
semilla × horizonte, reducción mediana +10.9%), 1 (el LSTM supera al baseline persistente
en 10/10), 2 (control en SUMO real, ver el resultado central) y 3 (Transformer y TSMixer,
ver abajo).

Transformer y TSMixer (`models/world_model/transformer.py`, `tsmixer.py`) están
**implementados y evaluados** como sustitutos intercambiables del LSTM (misma interfaz,
`models/world_model/base.py::TemporalModel`). **No reemplazan al LSTM**: el LSTM tiene el
menor `reward_mse` en los 10 horizontes (reducción mediana de 38.1% frente al Transformer
y 56.1% frente a TSMixer).

## Resultado central (preciso)

Medias de todas las semillas de entrenamiento (PPO del sueño 3, RL directo 4), en SUMO
real, con el RL directo reentrenado tras el fix de C1 (conexión TraCI global):

| | Escenarios oficiales (3000–3014, 5000–5014) | Escenarios nuevos (7000–7029) |
|---|---|---|
| PPO del sueño (World Model) | -326.79 | -293.81 |
| RL directo, 10k pasos (13,240 interacciones por semilla) | -454.51 | -439.73 |
| RL directo, 30k pasos (39,208 interacciones por semilla) | -402.13 | -316.59 |
| Tiempo fijo | -411.27 | -391.70 |

- **Frente al RL directo de 10k**, el World Model es mejor en los dos conjuntos (pareado
  por escenario p = 0.0002 y p = 4.5e-9).
- **Frente al de 30k no se detectó una diferencia consistente:** hay brecha a favor del
  World Model en los escenarios oficiales (+75.34; semilla p = 0.029, pareado p = 0.020) y
  ninguna detectable en los nuevos (pareado p = 0.54).
- La ventaja demostrada es de **eficiencia en interacciones reales**: el World Model
  (~4,600 por semilla, 7,800 sin amortizar el dataset) iguala o supera al RL directo con
  entre 2.9x y 8.5x menos interacciones (1.7x a 5.0x sin amortizar). No es una ventaja de
  control cuando el RL directo dispone de presupuesto de sobra, y el punto de equilibrio
  exacto no se midió.
- El World Model es el método con **menos episodios catastróficos** (< -600): 5/90 y 1/90,
  frente a 25/120 y 29/120 (10k) y 24/120 y 9/120 (30k).

Detalle completo, con todas las pruebas, en PROJECT_STATUS.md ("Auditoría técnica y
correcciones").

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
- El término de "throughput" de la recompensa (`info["throughput"]`) **no mide llegadas**:
  `getArrivedNumber()` solo cuenta el último segundo simulado del paso y el código resta
  dos de esos valores (~13% de las llegadas reales). Se mantiene **sin cambios** porque todo
  lo entrenado lo usa y pesa ~0.2 por paso frente a una recompensa media de -43. Para
  cualquier afirmación sobre flujo vehicular se usa `info["arrivals_total"]` (llegadas del
  intervalo completo, verificada contra un conteo segundo a segundo), que no entra en la
  recompensa.
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
  que se atribuía al Autoencoder en el Experimento 0 publicado; con el Experimento 0
  rehecho es +10.9%, y el LSTM gana con reducciones del 38–56%, así que la conclusión no
  cambia). La decisión se toma por `reward_mse` en rollouts, nunca por la pérdida de
  validación: el Transformer tiene mejor pérdida de validación y mejor `latent_mse`, pero
  peor `reward_mse`.
- Experimento 0: las dos ramas (z y estado crudo) **importan el mismo protocolo** de
  `train_world_model.py` (un test lo garantiza), se comparan con **varias semillas
  emparejadas por semilla** (`scripts/compare_experiment_0_multiseed.py`) y se entrenan
  hasta que corte el early stopping (tope de 300 épocas), no hasta un tope que corte solo
  a una rama. En horizontes largos se reporta también la mediana por episodio: con 12
  episodios de test, 2 o 3 congestionados dominan la media.
- Contabilidad de interacciones reales: el dataset del World Model (4,800 transiciones)
  se recolecta una vez y lo comparten las 3 semillas (~4,600 por semilla con la
  selección en SUMO real; 7,800 si no se amortiza). La evaluación periódica del RL
  directo cuenta como interacción real, y SB3 completa el último rollout: 13,240 por
  semilla con 10k pasos y 39,208 con 30k. Reportar la razón como rango (2.9x a 8.5x, o
  1.7x a 5.0x sin amortizar), nunca como un solo número. No describir la comparación con
  10k como "a presupuesto comparable".
- Los resultados de controladores se reportan como media de **todas** las semillas de
  entrenamiento (3 del sueño, 4 del RL directo), nunca con una sola, en los escenarios
  oficiales **y** en los nuevos (7000–7029). Las comparaciones usan el Welch sobre las
  medias por semilla y el pareado por escenario; los tests a nivel de episodio son
  pseudorreplicación. Todo resultado por episodio queda versionado en `docs/results/`.

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
- Los checkpoints **oficiales** son `models/checkpoints/controller/best_model.zip` (PPO del
  sueño, semilla 2), `controller_direct/best_model.zip` (RL directo, 10,000 pasos,
  semilla 0) y `controller_direct_30k/best_model.zip` (RL directo, 30,000 pasos, semilla
  1; verificación de presupuesto). En cada carpeta, el oficial es la semilla con mejor
  media en los escenarios oficiales, una etiqueta descriptiva; las demás semillas
  (`_seedN`) son evidencia de variabilidad. No cambiarlos sin una decisión explícita.
- No escribir en los archivos: `controller_direct_prefix_bug/` y
  `controller_direct_30k_prefix_bug/` (RL directo entrenado con el bug C1) y
  `raw_state_old_protocol/` (Experimento 0 con el protocolo viejo). Los scripts de
  entrenamiento se niegan a escribir en carpetas oficiales o archivadas sin
  `--overwrite-official`.
- El modelo temporal del sistema es el LSTM (`world_model_best.pt`): el Dream
  Environment y los PPO dependen de él. No reentrenarlo ni sobrescribirlo sin consultar.
  Se entrenó hasta el tope de 100 épocas (mejor época 93), no hasta converger; con early
  stopping real su mejor época habría sido la 152.

## Estructura del repositorio

```
configs/        EnvironmentConfig, RewardConfig, RepresentationConfig, WorldModelConfig,
                ControllerConfig
datasets/       transition_dataset.py, latent_sequence_dataset.py, metadata.json, raw/, processed/ (generados)
docs/           PROPUESTA.md (propuesta académica), DOCUMENTACION_PROYECTO.md (documento de estudio),
                results/ (resultados por episodio de cada evaluación, versionados)
environments/   TrafficEnvironment, CustomStateBuilder, TrafficState, ProjectActionSpace,
                ProjectRewardFunction, DreamEnvironment, EncodedTrafficEnvironment,
                ReseedingWrapper, single-intersection/ (red SUMO propia, demanda asimétrica)
models/
  representation/  Encoder, Decoder, Autoencoder
  world_model/     base.py (Protocol TemporalModel), lstm.py (LatentDynamicsLSTM, el del sistema),
                   transformer.py, tsmixer.py (Experimento 3)
  checkpoints/     controller/, controller_direct/ y controller_direct_30k/ (oficiales + _seedN),
                   *_prefix_bug/ (RL directo antes del fix de C1), exp0_multiseed/ y
                   exp0_multiseed_300ep/ (Experimento 0), raw_state_old_protocol/ (archivo);
                   pesos no versionados, sus .json sí
training/       train_autoencoder.py, train_world_model.py, train_world_model_raw.py,
                train_world_model_transformer.py, train_world_model_tsmixer.py,
                train_controller.py (PPO del sueño), train_controller_direct.py (RL directo),
                output_guard.py (no sobrescribir resultados oficiales o archivados)
evaluation/     autoencoder_evaluation.py, evaluate_autoencoder.py,
                world_model_evaluation.py (incluye build_world_model)
scripts/        datos: collect_dataset.py, split_dataset.py, merge_dataset.py, normalize_dataset.py,
                  visualize_dataset.py, encode_latent_dataset.py, test_environment.py
                Exp. 0/1: prepare_raw_sequence_dataset.py, evaluate_world_model.py,
                  evaluate_world_model_raw.py, compare_experiment_0.py,
                  compare_experiment_0_multiseed.py
                Exp. 3: evaluate_world_model_transformer.py, evaluate_world_model_tsmixer.py,
                  compare_experiment_3.py
                control: evaluate_controller.py, evaluate_controller_sumo.py,
                  evaluate_direct_vs_dream.py, evaluate_final_comparison.py,
                  evaluate_multiseed_statistical.py, analyze_controller_actions.py
tests/          16 archivos, 81 tests (pytest -v)
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
