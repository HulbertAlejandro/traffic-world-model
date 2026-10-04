# Diseño de la Fase 3 de la v2: control en el corredor de 4 intersecciones

**Escrito el 4 de octubre de 2026, antes de entrenar o evaluar ningún controlador.** Describe qué
se reutiliza de la v1 y qué cambia. El pre-registro (brazos, semillas, métricas, estadística,
hipótesis) está en `ADDENDUM_CONTROL.md`.

## 1. Qué se reutiliza de la v1, importado y sin copiar

| Pieza de la v1 | Uso en la v2 |
|---|---|
| `configs/controller.py::ControllerConfig` | Los mismos hiperparámetros de PPO en todos los brazos (sección 5). |
| `training/train_controller.py::build_normalized_envs` | Envuelve el entorno de entrenamiento y el de evaluación con `VecNormalize` exactamente como en la v1 (`Monitor` por debajo; recompensa normalizada en entrenamiento, real en evaluación). |
| `training/train_controller.py::save_hyperparameters`, `vecnormalize_path`, `SaveVecNormalizeOnBest`, `load_obs_normalizer` | El `.json` junto a cada checkpoint y las estadísticas de `VecNormalize` del RL directo. |
| `environments/reseeding_wrapper.py::ReseedingWrapper` | Semilla nueva por episodio en el RL directo y semillas fijas cíclicas en las evaluaciones periódicas. |
| `training/output_guard.py::check_output_dir` | La protección contra sobrescribir resultados. |
| `evaluation/world_model_evaluation.py::predict_next_step`, `load_episodes` | El paso imaginado y la carga de episodios por `episode_id`, la misma fuente de verdad que la evaluación del modelo. |
| `datasets/latent_sequence_dataset.py::encode_actions` | La acción conjunta pasa al one-hot por semáforo con la misma función con que se entrenaron los modelos. |
| `training/v2_compression_experiment.py::build_temporal_model` | Reconstruye la LSTM o el Transformer desde su `world_model_best.json`. |
| `scripts/evaluate_multiseed_statistical.py::welch`, `paired_t` y el Wilcoxon de `docs/results/ppo_10_seeds/analyze.py` (vía `scripts/v2/validate_corridor_demand.py`) | Estadística de las comparaciones. |
| `scripts/v2/corridor_policies.py` (políticas del validador de la Fase 0) | Las 5 referencias sin aprendizaje, sin reimplementarlas. |

`DreamEnvironment` y `EncodedTrafficEnvironment` de la v1 **no se pueden usar tal cual**: el
primero carga solo la LSTM de la v1 (`load_world_model`) y codifica la acción con
`F.one_hot(acción escalar)`; el segundo corre el Encoder. Las clases de la v2 siguen su mismo
diseño y comparten con ellas las funciones de arriba; lo que cambia es lo de la sección 2.

## 2. Qué cambia en la v2

### 2.1 Sin Encoder: el puente es solo el escalador de estado

Decisión ya tomada (Fase 2): no hay Autoencoder. Los modelos temporales se entrenaron sobre el
estado de 104 dimensiones **normalizado con `datasets/v2/processed/scaler.pkl`** (media y
desviación del split de entrenamiento; las columnas constantes tienen desviación 1, regla de
`scripts/normalize_dataset.py`). Se verificó que la `z` de
`models/checkpoints/v2/compression/selection/raw_data/*_seq.npz`, con la que se entrenaron los 20
modelos, es exactamente ese estado normalizado (diferencia máxima 0.0).

- **`environments/scaled_corridor_environment.py::ScaledCorridorEnvironment`** envuelve
  `CorridorTrafficEnvironment` y normaliza cada observación con ese mismo `scaler.pkl`. Es el
  reemplazo de `EncodedTrafficEnvironment`: la lección del bug de la v1 (el puente usaba otra
  normalización que el entrenamiento) se aplica cargando el **mismo archivo**, y un test reproduce
  un episodio del dataset y compara las observaciones con las del dataset normalizado.
- El controlador del sueño ve, en SUMO real, exactamente la misma escala que vio en el sueño.

### 2.2 Acción conjunta `MultiDiscrete([2, 2, 2, 2])`

PPO elige 4 bits (mantener 0 / cambiar 1, uno por semáforo, A0 a D0), el mismo espacio de acción de
`CorridorTrafficEnvironment`. En el sueño, la acción se traduce al vector de 8 posiciones que
espera el modelo con `encode_actions(acción[None], 8)`: el one-hot de 2 clases de cada semáforo,
concatenado en orden. Es la misma función con que se construyeron las ventanas de entrenamiento,
así que no hay una segunda codificación que pueda divergir (un test lo comprueba para las 16
acciones).

### 2.3 Observación de 104 valores normalizados

Sueño y SUMO real entregan al controlador el estado normalizado de 104 dimensiones. Como en la v1
con `z`, el PPO del sueño usa `normalize_obs = False`: la observación ya está en escala unitaria y
sin estadísticas móviles que guardar junto al checkpoint.

### 2.4 La arquitectura del modelo es un parámetro

`CorridorDreamEnvironment(model_dir=...)` lee `world_model_best.json` y reconstruye la arquitectura
que diga (`lstm` o `transformer`) con `build_temporal_model`. El script de entrenamiento recibe
`--arm dream_lstm | dream_transformer | direct`; el controlador de semilla *i* usa el modelo de
semilla *i* de su arquitectura (`models/checkpoints/v2/arch_comparison/<arq>_raw_s<i>/`).

### 2.5 El Dream Environment de la v2

`environments/corridor_dream_environment.py::CorridorDreamEnvironment`. Igual que el de la v1
salvo lo dicho arriba:

- **Ventanas semilla reales** de 16 pasos (`sequence_length` del modelo), tomadas del split de
  **entrenamiento** (`train_seq.npz`), igual que la v1 con `train_latent.npz`. El inicio de la
  ventana se sortea en `[0, T − 16]`, como en la v1.
- **Solo la última acción de la ventana es hipotética**; las anteriores son las registradas.
- **`max_dream_steps = 7`**, como en la v1 (fijado en el pre-registro, no se ajusta).
- **Recorte de la recompensa imaginada** con la regla de la v1, recalculada para el dataset de la
  v2: percentiles 1 y 99 de las recompensas reales del split de entrenamiento de la v2
  (6,720 transiciones): **[−345.43, 0.0]** (mínimo −777.0, media −45.65). `info` expone,
  como en la v1, si se recortó y el valor sin recortar.
- **Nunca llama a SUMO** (un test lo comprueba bloqueando `traci` y `sumo_rl`).
- `reset(options={"episode_id": e, "start": s})` permite fijar la ventana; lo usa el diagnóstico de
  fidelidad, que así recorre **el mismo código** que ve PPO.

### 2.6 Entrenamiento y evaluación

- **`training/train_controller_v2.py`**: un solo script para los tres brazos, con `--seed`,
  `--output-dir` y la protección de la v1 (no escribe en una carpeta con resultados sin
  `--overwrite`, ni en carpetas oficiales o archivadas sin `--overwrite-official`; las carpetas de
  los modelos del mundo de la v2 quedan protegidas también). La configuración de PPO de los tres
  brazos sale de una sola función (`ppo_config`), que parte del mismo `ControllerConfig()`.
  Guarda `run_info.json` con los pasos reales consumidos (entrenamiento y evaluación periódica,
  contados por un envoltorio, no deducidos) y los tiempos.
- **`scripts/v2/evaluate_control_v2.py`**: evalúa políticas (sueño, directo o referencia) sobre una
  lista de semillas de un split, y guarda por episodio el retorno total y **por intersección**
  (A0, B0, C0, D0), espera, cola, llegadas y cambios de fase por semáforo. Se niega a usar
  semillas de entrenamiento (20000–20111) y exige una confirmación explícita para `test` y `ood`.
- **`scripts/v2/control_fidelity.py`**: el diagnóstico de fidelidad del Paso 4.

### 2.7 Conexiones TraCI concurrentes (C1)

El RL directo abre dos simulaciones en el mismo proceso (entrenamiento y evaluación). En la v2,
`CorridorStateBuilder` lee cada simulación por la conexión de su propio entorno, y
`tests/test_corridor_concurrent_environments.py` ya lo comprueba sin envoltorio; se agrega la
misma comprobación con `ScaledCorridorEnvironment`.

## 3. Limitaciones conocidas del diseño, heredadas de la v1

1. **El controlador del sueño solo ve estados desde el paso 15 de un episodio.** Cada ventana
   semilla tiene 16 pasos reales, así que la primera observación imaginada es la de t ≥ 15
   (≥ 75 s de simulación). En SUMO real el controlador decide desde t = 0, con la red casi vacía.
   La v1 tenía la misma limitación (también `sequence_length = 16`). No se rellena (regla del
   proyecto).
2. **El sueño no sabe en qué segundo del episodio está.** Con el inicio sorteado en `[0, T − 16]`,
   hasta 7 pasos imaginados pueden caer después del segundo 300, fuera de lo que el modelo vio.
   Se conserva el comportamiento de la v1.
3. **PPO no tiene memoria.** `MlpPolicy` ve solo el estado actual; el modelo del mundo usa 16 pasos
   de historia, pero el controlador no. Ver la hipótesis en `ADDENDUM_CONTROL.md`, sección 6.

## 4. Archivos que se agregan

```
environments/corridor_dream_environment.py
environments/scaled_corridor_environment.py
training/train_controller_v2.py
scripts/v2/evaluate_control_v2.py
scripts/v2/control_fidelity.py
scripts/v2/corridor_policies.py        (agrega las 5 referencias, reutilizando el validador)
tests/test_v2_control.py
```

Salidas: `models/checkpoints/v2/control/` (pesos no versionados; sí sus `.json`),
`models/checkpoints/v2/control_pilot/` (piloto de humo) y `docs/results/v2/control/`.

## 5. Configuración de PPO

`ControllerConfig()` de la v1, sin cambios: `learning_rate` 3e-4, `n_steps` 256, `batch_size` 64,
`n_epochs` 10, `gamma` 0.99, `normalize_reward` True, `reward_clip` 10, `dream_max_steps` 7,
`MlpPolicy` por defecto de SB3. Por brazo solo cambian, como en la v1:

| Campo | Sueño (LSTM y Transformer) | RL directo |
|---|---|---|
| `seed` | semilla del controlador (0–9) | semilla del controlador (0–9) |
| `total_timesteps` | 50,000 imaginados (default de la v1) | 10,000 o 30,000 reales |
| `normalize_obs` | False (estado ya normalizado con `scaler.pkl`) | True (estado crudo, `VecNormalize`, como en la v1) |
| Evaluación periódica | cada 5,000 pasos, 5 episodios | cada `max(n_steps, 1000)` pasos, 5 episodios |

El RL directo no usa `scaler.pkl` a propósito: ese escalador sale del dataset, y si el RL directo
dependiera de él habría que sumarle las interacciones del dataset (ver `ADDENDUM_CONTROL.md`).
