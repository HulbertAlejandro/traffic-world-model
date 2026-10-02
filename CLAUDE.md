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
       vía EncodedTrafficEnvironment)                                            [COMPLETO, 10 semillas]
     → PPO directo contra SUMO real (baseline de RL directo)                     [COMPLETO, 10 semillas
                                                                                  × 10k y 30k pasos]
     → evaluate_multiseed_statistical.py (sueño vs. directo vs. tiempo fijo vs.
       regla trivial, N semillas, escenarios oficiales y nuevos, en SUMO real)   [COMPLETO]
```

Experimentos cerrados: 0 (el Autoencoder mejora la predicción de la recompensa de forma
moderada y mayoritaria: 5 semillas por rama con convergencia igualada, 37/50 pares
semilla × horizonte, reducción mediana +10.9%), 1 (el LSTM supera al baseline persistente
en 10/10), 2 (control en SUMO real, ver el resultado central) y 3 (Transformer y TSMixer,
ver abajo).

Exploración posterior del Experimento 0, **sin cambios en lo oficial** (`latent_dim` sigue en 16
y el estado en 26 dims): `docs/EXPLORACION_LATENT_DIM.md` y `docs/results/exploratory/latent_sweep/`
(rama `exploratory/latent-dim-sweep`, ya fusionada en main).

- **Autoencoder con la LSTM:** con 5 semillas de Autoencoder × 5 de LSTM no hay evidencia de que mejore
  ni empeore la predicción de la recompensa. La varianza dominante es qué Autoencoder se entrena.
- **Transformer y TSMixer** (5×5 a `latent_dim` = 16): el latente es neutro con la LSTM y empeora la
  predicción de la recompensa con el Transformer (+33%) y con TSMixer (+42%).

**Nota sobre el Experimento 0 (réplica con episodios de test nuevos, exploratoria).** El resultado
oficial (+10.9%, 37/50) se midió sobre los 12 episodios del test oficial, y **depende de cuáles son
esos 12**:

- Se re-simularon los episodios de test 8 y 53, y SUMO 1.27.1 los reproduce bit a bit.
- Después se generaron 36 episodios de test nuevos con el protocolo oficial (semillas de SUMO
  9000–9035).
- Las mismas 5 + 5 semillas del Experimento 0 pasan de Δ = −13.0% (z mejor) en los 12 originales a
  **+3.6%** en los 36 nuevos. Sobre los 48, Δ = −0.7%, con IC conjunto [−14.6%, +15.4%].
  - Esta Δ es la diferencia z − crudo de la media geométrica del `reward_mse` sobre h = 1..10, no la
    reducción mediana del resultado oficial.
- Lo mismo con las grillas de la exploración: −7% → −0.4% y −10% → −2.9% con 48 episodios.

En cambio, sí se replican en los episodios nuevos el daño del latente al Transformer y a TSMixer y
el orden LSTM > Transformer > TSMixer. Detalle en la sección 16 de
`docs/results/exploratory/latent_sweep/REPORT.md`.

El resultado oficial **no se reescribe**. Al citarlo, decir que se midió sobre 12 episodios de test
y que en episodios independientes la ventaja del Autoencoder con la LSTM no se replica (efecto
cercano a 0).

Transformer y TSMixer (`models/world_model/transformer.py`, `tsmixer.py`) están
**implementados y evaluados** como sustitutos intercambiables del LSTM (misma interfaz,
`models/world_model/base.py::TemporalModel`). **No reemplazan al LSTM**: el LSTM tiene el
menor `reward_mse` en los 10 horizontes (reducción mediana de 38.1% frente al Transformer
y 56.1% frente a TSMixer).

## Resultado central (preciso)

Medias de **10 semillas de entrenamiento por controlador**, en SUMO real, con el RL directo
reentrenado tras el fix de C1 (conexión TraCI global). La extensión de 3 / 4 a 10 semillas se
pre-registró antes de entrenar (`docs/results/ppo_10_seeds/ADDENDUM.md`):

| | Escenarios oficiales (3000–3014, 5000–5014) | Escenarios nuevos (7000–7029) |
|---|---|---|
| PPO del sueño (World Model) | -326.89 | -305.53 |
| RL directo, 10k pasos (13,240 interacciones por semilla) | -455.02 | -445.39 |
| RL directo, 30k pasos (39,208 interacciones por semilla) | -403.83 | -303.51 |
| Tiempo fijo | -411.27 | -391.70 |

- **Frente al RL directo de 10k**, el World Model es mejor en los dos conjuntos, y con 10
  semillas la evidencia se refuerza (+128.13 y +139.86; t pareada por escenario p = 2.0e-6
  y p = 5.0e-11; Welch por semilla p = 2.3e-5 y p = 6.7e-4).
- **Frente al de 30k no se detecta una diferencia de control en ningún conjunto de
  escenarios.** Hay que distinguir la brecha numérica de la significancia:
  - En los oficiales hay una **brecha de magnitud similar a la publicada** (+76.94, antes
    +75.34), pero con 10 semillas **ya no alcanza significancia estadística** (t pareada
    p = 0.068, Wilcoxon p = 0.11, Welch por semilla p = 0.12; antes pareado p = 0.020).
    Buena parte de esa brecha viene de la semilla 8 del 30k (media -774.03, un episodio de
    -10,913); sin ella, +35.80 (t pareada p = 0.0995).
  - En los nuevos no hay brecha (-2.02; t pareada p = 0.91).
- La ventaja demostrada es de **eficiencia en interacciones reales**: el World Model iguala o
  supera al RL directo con menos interacciones. La razón depende de cómo se cuente el dataset
  de 4,800 transiciones:
  - **Sin compartirlo (comparación conservadora, no cambió con 10 semillas):** 7,800 por
    semilla, **1.7x a 5.0x** menos.
  - **Compartiéndolo entre las 10 semillas:** 3,480 por semilla, **3.8x a 11.3x** menos. Esta
    cifra mejora solo por entrenar más semillas (con 3 era 2.9x a 8.5x).

  No es una ventaja de control cuando el RL directo dispone de presupuesto de sobra, y el
  punto de equilibrio exacto no se midió.
- Entre los controladores aprendidos, el World Model es el que tiene **menos episodios
  catastróficos** (< -600): 19/300 y 13/300, frente a 57/300 y 65/300 (10k) y 38/300 y
  19/300 (30k). Tiempo fijo tiene 0/30 en los dos conjuntos.

Detalle completo en PROJECT_STATUS.md ("Extensión del PPO a 10 semillas por controlador"), y
el resultado anterior con 3 / 4 semillas en "Auditoría técnica y correcciones", punto 2.

## Decisiones de diseño ya tomadas

- El estado (`TrafficState`) tiene **26 dimensiones**: 4 carriles × 5 variables
  (vehicle_counts, queue_lengths, waiting_times, mean_speeds, occupancies) + 4 de
  one-hot de fase del semáforo + 2 (elapsed/remaining phase time). De las 4 posiciones
  de fase solo se activan 2: la lógica que construye sumo-rl tiene 2 fases verdes y 2
  amarillas, y el estado registra siempre la verde. `remaining_phase_time` vale
  `min_green − elapsed`, así que es 0 en todos los pasos salvo al inicio del episodio (5).
  No se cambia: todo lo entrenado usa este vector.
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
- Semillas fijas (`torch`, `numpy`, `cuda`) en todo entrenamiento. La recolección del
  dataset fija la semilla de SUMO de cada episodio, pero **no** la de las acciones
  aleatorias (`np.random` global): una recolección nueva da otro dataset. El oficial está
  respaldado y no se regenera.
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
  se recolecta una vez y lo comparten las 10 semillas del sueño (3,480 por semilla con
  los 3,000 de selección en SUMO real; 7,800 si no se comparte). La evaluación periódica
  del RL directo cuenta como interacción real, y SB3 completa el último rollout: 13,240
  por semilla con 10k pasos y 39,208 con 30k. Reportar la razón como rango y con las dos
  cifras: **1.7x a 5.0x sin compartir el dataset** (la comparación conservadora) y 3.8x a
  11.3x compartiéndolo entre las 10 semillas (esta mejora solo por entrenar más semillas).
  Nunca como un solo número. No describir la comparación con 10k como "a presupuesto
  comparable".
- Los resultados de controladores se reportan como media de **todas** las semillas de
  entrenamiento (10 del sueño, 10 de cada RL directo), nunca con una sola, en los
  escenarios oficiales **y** en los nuevos (7000–7029). Las comparaciones usan el Welch
  sobre las medias por semilla y la t pareada por escenario, y desde la extensión a 10
  semillas también el Wilcoxon pareado por escenario. Los tests a nivel de episodio son
  pseudorreplicación. Una diferencia numérica sin significancia se reporta como "brecha
  no significativa", nunca como diferencia de control. Todo resultado por episodio queda
  versionado en `docs/results/`.

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
  1; verificación de presupuesto). El oficial se eligió como la semilla con mejor media
  en los escenarios oficiales **entre las semillas originales** (0–2 del sueño, 0–3 del
  RL directo); es una etiqueta descriptiva. Con las 10 semillas ya no es la mejor de su
  método (sueño: semilla 6, -287.10; 10k: semilla 8, -325.74; 30k: semilla 6, -283.62), y
  por decisión pre-registrada **no se reetiqueta**: el resultado es la media de todas las
  semillas. Las demás semillas (`_seedN` y las carpetas `*_10seeds/`) son evidencia de
  variabilidad. No cambiarlos sin una decisión explícita.
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
                EXPLORACION_LATENT_DIM.md (exploración del Experimento 0, no oficial),
                results/ (resultados por episodio de cada evaluación, versionados;
                results/exploratory/ = exploraciones, no forman parte del pipeline oficial)
environments/   TrafficEnvironment, CustomStateBuilder, TrafficState, ProjectActionSpace,
                ProjectRewardFunction, DreamEnvironment, EncodedTrafficEnvironment,
                ReseedingWrapper, single-intersection/ (red SUMO propia, demanda asimétrica)
models/
  representation/  Encoder, Decoder, Autoencoder
  world_model/     base.py (Protocol TemporalModel), lstm.py (LatentDynamicsLSTM, el del sistema),
                   transformer.py, tsmixer.py (Experimento 3)
  checkpoints/     controller/, controller_direct/ y controller_direct_30k/ (oficiales + _seedN),
                   controller_10seeds/ (sueño, semillas 3–9), controller_direct_10seeds/ y
                   controller_direct_30k_10seeds/ (semillas 4–9) (extensión a 10 semillas),
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
                Tablas 4a y 4b: benchmark_temporal_models.py (costo de inferencia de LSTM,
                  Transformer y TSMixer), evaluate_model_fidelity.py (fidelidad del retorno
                  imaginado de los tres modelos frente al real)
                control: evaluate_controller.py, evaluate_controller_sumo.py,
                  evaluate_direct_vs_dream.py, evaluate_final_comparison.py,
                  evaluate_multiseed_statistical.py, analyze_controller_actions.py
tests/          16 archivos, 81 tests (pytest -v)
ver_controlador.py (demo del PPO del sueño oficial en la GUI de SUMO, escenario nuevo 7025),
ver_tiempo_fijo.py (el mismo demo con la política de tiempo fijo, para comparar a simple vista),
CLAUDE.md, PROJECT_STATUS.md, TODO.md, README.md, pytest.ini, requirements.txt, .gitignore, LICENSE
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
