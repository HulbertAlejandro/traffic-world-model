# Documentación del Proyecto — Traffic World Model

Control inteligente de semáforos mediante World Models en una intersección simulada de SUMO.

> Este documento describe **todo** el proyecto tal como existe hoy en el repositorio (`https://github.com/HulbertAlejandro/traffic-world-model`, estado hasta el commit `dd0e547`) — archivo por archivo, concepto por concepto, con las cifras finales verificadas contra `PROJECT_STATUS.md`. Está pensado para leerse de principio a fin sin necesitar ningún documento anterior como referencia. No incluye nada que no esté implementado, y marca claramente lo que aún falta.

---

## Índice

1. [Resumen del proyecto](#1-resumen-del-proyecto)
2. [Estado actual del proyecto](#2-estado-actual-del-proyecto)
3. [Estructura completa del repositorio](#3-estructura-completa-del-repositorio)
4. [Explicación archivo por archivo](#4-explicación-archivo-por-archivo)
5. [Flujo completo del proyecto](#5-flujo-completo-del-proyecto)
6. [Configuración](#6-configuración)
7. [Dataset](#7-dataset)
8. [Environments](#8-environments)
9. [Modelos](#9-modelos)
10. [Entrenamiento](#10-entrenamiento)
11. [Evaluación](#11-evaluación)
12. [Tests](#12-tests)
13. [Comandos importantes](#13-comandos-importantes)
14. [Qué mostrar en la sustentación](#14-qué-mostrar-en-la-sustentación)
15. [Preguntas probables de la profesora](#15-preguntas-probables-de-la-profesora)
16. [Qué falta por desarrollar](#16-qué-falta-por-desarrollar)
17. [Glosario completo](#17-glosario-completo)
18. [Resumen final (lectura de 5 minutos)](#18-resumen-final-lectura-de-5-minutos)
19. [Observaciones técnicas y deuda conocida](#19-observaciones-técnicas-y-deuda-conocida)
20. [Historia del proyecto: de la primera intersección al resultado final](#20-historia-del-proyecto-de-la-primera-intersección-al-resultado-final)

---

## 1. Resumen del proyecto

**Objetivo.** Diseñar e implementar un sistema basado en *World Models* (Ha & Schmidhuber, 2018) que aprenda la dinámica del tráfico de una intersección simulada en SUMO, y usar ese aprendizaje para entrenar un controlador de semáforo **sin** interactuar con el simulador durante el entrenamiento — comparando este enfoque contra entrenar un controlador directamente contra SUMO.

**Problema que resuelve.** Entrenar un controlador de semáforo por Reinforcement Learning clásico requiere miles de interacciones con el simulador (lento y costoso). La idea de un World Model es que el agente primero aprenda una *aproximación interna* de cómo se comporta el tráfico, y luego pueda "imaginar" las consecuencias de sus acciones sin tener que probarlas todas en el simulador real.

**Pregunta de investigación.** ¿Puede un modelo aprendido de la dinámica del tráfico igualar o superar a un controlador entrenado directamente contra el simulador, usando muchas menos interacciones reales?

**Arquitectura completa, implementada de punta a punta:**

```
SUMO → Estado del tráfico (26 dims) → Autoencoder → z (16 dims)
     → LatentDynamicsLSTM → predicción de (ẑ_{t+1}, r̂_{t+1})
     → Dream Environment → Controlador PPO (entrenado sin tocar SUMO)
     → Evaluación en SUMO real, comparado contra RL directo y baselines clásicos
```

**Qué está implementado y verificado hoy:**
- El simulador (SUMO) y su conexión con Python vía `TrafficEnvironment`.
- El entorno propio del proyecto, con estado y recompensa diseñados a medida (no la observación nativa de `sumo-rl`).
- El pipeline completo de datos (recolectar con semilla variable → dividir → normalizar).
- El Autoencoder, entrenado y validado mediante un experimento controlado (Experimento 0).
- El modelo temporal `LatentDynamicsLSTM`, que predice tanto el siguiente estado latente como la recompensa.
- El `DreamEnvironment`, que permite entrenar un controlador sin tocar SUMO.
- Un controlador PPO entrenado en el sueño, verificado con 3 semillas de entrenamiento y evaluado en SUMO real.
- Un baseline de RL directo (PPO entrenado sin pasar por el World Model), también con 3 semillas.
- Un escenario de demanda de tráfico asimétrica, diseñado específicamente para que la solución trivial del problema deje de ser óptima y así poder comparar los métodos por calidad de control real.
- La comparación final entre los dos métodos y dos baselines clásicos (tiempo fijo, regla trivial), en SUMO real.
- Transformer y TSMixer como alternativas al LSTM (Experimento 3): implementados, entrenados con el mismo protocolo y evaluados; ninguno mejora al LSTM, que se mantiene.
- Una verificación del RL directo con el triple de presupuesto de entrenamiento: alcanza un desempeño comparable al del World Model, pero con ~8.5 veces más interacciones reales.

**Qué falta, de forma deliberada (no por descuido):**
- Más semillas por método (siguen siendo 3), para poder afirmar o descartar diferencias a nivel de semilla.
- Una curva de desempeño frente a interacciones reales del RL directo: hoy solo hay dos puntos (13,000 y 39,000 interacciones por semilla).
- La causa completa de un número reducido de episodios catastróficos que persisten en ambos métodos.
- Una demanda de tráfico que varíe en el tiempo (hoy es asimétrica pero constante dentro de cada episodio).

---

## 2. Estado actual del proyecto

| Bloque | Estado | Tests |
|---|---|---|
| Entorno SUMO (`TrafficEnvironment`, estado de 26 dims, recompensa) | ✅ Completo | 5 |
| Pipeline de dataset (recolección con semilla por episodio, split, normalización) | ✅ Completo | 3 |
| Autoencoder | ✅ Completo | 2 |
| Modelo temporal LSTM (`LatentDynamicsLSTM`) | ✅ Completo | 10 |
| Experimento 1 (LSTM vs. baseline persistente, horizontes 1-10) | ✅ Completo | 9 (incluye 3 del selector `build_world_model`) |
| Experimento 0 (Autoencoder vs. estado crudo) | ✅ Completo | — |
| Dream Environment | ✅ Completo | 9 |
| Controlador PPO entrenado en el sueño | ✅ Completo, 3 semillas | 9 (compartidos) |
| Baseline de RL directo | ✅ Completo, 3 semillas | 3 (compartidos) |
| Evaluación final comparativa (SUMO real) | ✅ Completo | — |
| Transformer / TSMixer (Experimento 3) | ✅ Completo; se mantiene la LSTM | 14 |
| Verificación del RL directo con 3x presupuesto (30,000 pasos) | ✅ Completo (checkpoints archivados, no oficiales) | — |
| Demanda de tráfico variable en el tiempo | ⚪ No implementado | 0 |

**62 tests automatizados, todos pasando**, distribuidos en 12 archivos dentro de `tests/`.

No existen `utils/`, `notebooks/`, `experiments/`, `papers/` ni ninguna carpeta `data/`: nunca se versionaron, y las carpetas vacías que quedaban en disco se eliminaron.

---

## 3. Estructura completa del repositorio

```text
traffic-world-model/
│
├── configs/                        # Todos los "ajustes" del proyecto, como dataclasses de Python
│   ├── __init__.py
│   ├── controller.py               # ControllerConfig (PPO, VecNormalize)
│   ├── environment.py              # EnvironmentConfig
│   ├── representation.py           # RepresentationConfig (Autoencoder)
│   ├── reward.py                   # RewardConfig
│   └── world_model.py              # WorldModelConfig (LSTM)
│
├── datasets/                       # El dataset como paquete de Python (código, no solo datos)
│   ├── __init__.py
│   ├── latent_sequence_dataset.py
│   ├── metadata.json
│   ├── transition_dataset.py
│   ├── raw/                        # episodios crudos (.npz), generados, no versionados
│   └── processed/                  # splits normalizados (.npz), generados, no versionados
│
├── environments/                   # El "mundo" del proyecto: SUMO, estado, acción, recompensa e imaginación
│   ├── __init__.py
│   ├── custom_state_builder.py
│   ├── dream_environment.py
│   ├── encoded_traffic_environment.py
│   ├── project_action_space.py
│   ├── project_reward_function.py
│   ├── reseeding_wrapper.py
│   ├── traffic_environment.py
│   ├── traffic_state.py
│   └── single-intersection/
│       ├── single-intersection.net.xml
│       ├── single-intersection.rou.xml                    # demanda ACTUAL (asimétrica)
│       ├── single-intersection_symmetric_backup.rou.xml   # demanda anterior, respaldada
│       └── single-intersection.sumocfg
│
├── models/                          # Las redes neuronales (solo su forma, no el entrenamiento)
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── decoder.py
│   │   └── encoder.py
│   ├── world_model/
│   │   ├── __init__.py
│   │   ├── base.py                 # Protocol TemporalModel
│   │   ├── lstm.py                 # LatentDynamicsLSTM (el modelo del sistema final)
│   │   ├── transformer.py          # LatentDynamicsTransformer (Experimento 3)
│   │   └── tsmixer.py              # LatentDynamicsTSMixer (Experimento 3)
│   └── checkpoints/                 # .pt/.zip/.pkl generados, no versionados; los .json sí
│       ├── controller/              # PPO del sueño (oficial: best_model, semilla 1)
│       ├── controller_direct/       # PPO directo, 10,000 pasos (oficial: best_model, semilla 0)
│       ├── controller_direct_30k/   # PPO directo, 30,000 pasos (verificación, no oficial)
│       └── raw_state/               # LSTM sobre estado crudo (Experimento 0)
│
├── training/                        # Los bucles de entrenamiento
│   ├── __init__.py
│   ├── train_autoencoder.py
│   ├── train_controller.py             # PPO en el Dream Environment
│   ├── train_controller_direct.py      # PPO directo contra SUMO real
│   ├── train_world_model.py            # LSTM sobre z
│   ├── train_world_model_raw.py        # LSTM sobre estado crudo (Experimento 0)
│   ├── train_world_model_transformer.py  # Transformer sobre z (Experimento 3)
│   └── train_world_model_tsmixer.py      # TSMixer sobre z (Experimento 3)
│
├── evaluation/                      # Cómo se mide qué tan bien aprendió un modelo
│   ├── __init__.py
│   ├── autoencoder_evaluation.py
│   ├── evaluate_autoencoder.py
│   └── world_model_evaluation.py
│
├── scripts/                         # Puntos de entrada ejecutables desde la terminal
│   ├── analyze_controller_actions.py
│   ├── collect_dataset.py
│   ├── compare_experiment_0.py
│   ├── compare_experiment_3.py
│   ├── encode_latent_dataset.py
│   ├── evaluate_controller.py            # autoevaluación dentro del sueño
│   ├── evaluate_controller_sumo.py       # evaluación del PPO del sueño en SUMO real
│   ├── evaluate_direct_vs_dream.py
│   ├── evaluate_final_comparison.py      # comparación final consolidada
│   ├── evaluate_world_model.py
│   ├── evaluate_world_model_raw.py
│   ├── evaluate_world_model_transformer.py
│   ├── evaluate_world_model_tsmixer.py
│   ├── merge_dataset.py
│   ├── normalize_dataset.py
│   ├── prepare_raw_sequence_dataset.py
│   ├── split_dataset.py
│   ├── test_environment.py               # prueba de humo manual
│   └── visualize_dataset.py
│
├── tests/                           # Pruebas automatizadas (pytest) — 12 archivos, 62 funciones
│   ├── test_autoencoder.py
│   ├── test_controller.py
│   ├── test_dataset_pipeline.py
│   ├── test_dream_environment.py
│   ├── test_encoded_traffic_environment.py
│   ├── test_latent_sequence_dataset.py
│   ├── test_reseeding_wrapper.py
│   ├── test_traffic_environment.py
│   ├── test_world_model.py
│   ├── test_world_model_evaluation.py
│   ├── test_world_model_transformer.py
│   └── test_world_model_tsmixer.py
│
├── docs/
│   ├── PROPUESTA.md
│   └── DOCUMENTACION_PROYECTO.md    # este documento
│
├── ver_controlador.py               # visualiza el PPO del sueño en la GUI de SUMO
├── pytest.ini
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
├── CLAUDE.md
├── PROJECT_STATUS.md
└── TODO.md
```

**Cómo interactúan las carpetas entre sí:** `environments/` es lo único que habla directamente con SUMO. `datasets/` y `scripts/` producen y transforman datos a partir de lo que genera `environments/`. `models/` define redes neuronales sin saber nada de entrenamiento. `training/` combina `datasets/` + `models/` + `configs/` para producir un modelo entrenado (un checkpoint). `evaluation/` carga esos checkpoints y mide qué tan bien funcionan. `tests/` verifica que todo lo anterior siga funcionando correctamente.

---

## 4. Explicación archivo por archivo

### `configs/environment.py`
**Responsabilidad:** define `EnvironmentConfig`, con los parámetros de la simulación de SUMO: ruta a la red (`net_file`), ruta a las rutas de tráfico (`route_file`), si se muestra la ventana gráfica (`use_gui`), duración de la simulación (`simulation_seconds`), si es un solo agente (`single_agent`) y la semilla aleatoria (`seed`).
**Quién lo usa:** `TrafficEnvironment`, al crearse.

### `configs/reward.py`
**Responsabilidad:** define `RewardConfig`, con los cuatro coeficientes de la fórmula de recompensa: `alpha`, `beta`, `gamma`, `delta`.
**Quién lo usa:** `ProjectRewardFunction`.

### `configs/representation.py`
**Responsabilidad:** define `RepresentationConfig`, todos los hiperparámetros del Autoencoder: dimensión de entrada (`input_dim`, sin valor por defecto a propósito, para que nunca se pueda "olvidar" actualizarla si el estado cambia de tamaño), dimensión de la capa oculta (`hidden_dim`), dimensión del espacio latente (`latent_dim`), función de activación (`activation`), semilla (`seed`), tasa de aprendizaje, optimizador, tamaño de lote, número de épocas, y carpeta donde se guardan los checkpoints.
**Validaciones que hace sola:** que `input_dim` y `latent_dim` sean positivos, que `latent_dim` sea menor que `input_dim` (si no, no habría compresión real), que la semilla no sea negativa, y que la activación sea una de las soportadas.
**Quién lo usa:** `training/train_autoencoder.py` y `evaluation/autoencoder_evaluation.py`.

### `configs/world_model.py`
**Responsabilidad:** define `WorldModelConfig`, con los parámetros del modelo temporal: `sequence_length` (cuántos pasos de historia mirará, 16), `hidden_dim`, y `action_dim`. El campo `latent_dim` **no se define de forma independiente**: si se le pasa una `RepresentationConfig`, `WorldModelConfig` toma su `latent_dim` de ahí automáticamente, y si le das ambos, verifica que coincidan (si no, lanza un error). Esto evita que el tamaño del espacio latente del Autoencoder y el que espera el modelo temporal queden desincronizados — un riesgo real que se materializó una vez en el pasado y se corrigió con esta validación.

### `configs/controller.py`
**Responsabilidad:** define `ControllerConfig`, los hiperparámetros de PPO (Stable-Baselines3) usados tanto por el entrenamiento en el sueño como por el RL directo: `seed` (semilla de entrenamiento, con valor por defecto `1`, la semilla oficial adoptada después de verificar 3 semillas), `total_timesteps`, tasa de aprendizaje, tamaño de lote, número de épocas por actualización, `gamma`, y los campos de normalización `normalize_reward` y `reward_clip` usados por `VecNormalize`.
**Quién lo usa:** `train_controller.py` y `train_controller_direct.py`.

### `datasets/transition_dataset.py`
**Responsabilidad:** envolver un archivo `.npz` ya procesado en un objeto que PyTorch pueda usar directamente para entrenar.
**Clase principal:** `TransitionDataset` (hereda de `torch.utils.data.Dataset`). Cada elemento que entrega es una tupla de 8 valores: estado, acción, recompensa, siguiente estado, identificador de episodio, paso dentro del episodio, y las banderas `terminated`/`truncated`.

### `datasets/latent_sequence_dataset.py`
**Responsabilidad:** construir ventanas de secuencia (de longitud `sequence_length=16`) de pares `(z, acción)`, respetando siempre los límites de cada episodio — una ventana nunca mezcla datos de dos episodios distintos, porque eso rompería el sentido de "historia continua" que el LSTM necesita.
**Quién lo usa:** `training/train_world_model.py` y `training/train_world_model_raw.py`.

### `datasets/metadata.json`
**Responsabilidad:** documento de referencia (no código) que describe qué campos tiene el dataset y qué significa cada uno.

### `environments/traffic_environment.py`
**Responsabilidad:** es el único punto de contacto entre el proyecto y SUMO. Todo lo demás en el proyecto debería hablar con SUMO *a través de esta clase*, nunca directamente.
**Clase principal:** `TrafficEnvironment`.
**Qué hace `reset()`:** reinicia la simulación de SUMO, construye el estado del proyecto (no la observación nativa de SUMO) y lo devuelve junto con información adicional.
**Qué hace `step(action)`:** valida que la acción sea válida, avanza la simulación un paso, construye el nuevo estado, calcula métricas (tiempo de espera total, longitud de cola total, vehículos que completaron su viaje), calcula la recompensa del proyecto (no la de SUMO), y devuelve `(siguiente_estado, recompensa, terminado, truncado, información)`.
**Qué devuelve:** los vectores de estado son arreglos NumPy de 26 posiciones.

### `environments/custom_state_builder.py`
**Responsabilidad:** construir el estado del proyecto leyendo directamente el simulador SUMO a través de TraCI (el protocolo de control), sin depender de la observación que trae `sumo-rl` por defecto.
**Clase principal:** `CustomStateBuilder`.
**Qué hace:** encuentra el semáforo de la intersección, encuentra los carriles de entrada, encuentra cuántas fases tiene el semáforo, y con eso construye un objeto `TrafficState` leyendo, para cada carril: número de vehículos, longitud de cola, tiempo de espera acumulado, velocidad media y ocupación; y para el semáforo: la fase actual (codificada como one-hot) y el tiempo transcurrido/restante de esa fase.

### `environments/traffic_state.py`
**Responsabilidad:** representar el estado del tráfico como un objeto claro y con nombre (no solo un arreglo de números sin etiquetas), y saber convertirse a vector cuando haga falta.
**Clase principal:** `TrafficState` — guarda `vehicle_counts`, `queue_lengths`, `waiting_times`, `mean_speeds`, `occupancies` (uno por carril), `phase_one_hot`, `elapsed_phase_time` y `remaining_phase_time`.
**Método importante:** `to_vector()`, que concatena todo lo anterior en un único arreglo de 26 números, en un orden fijo y siempre igual.

### `environments/project_action_space.py`
**Responsabilidad:** definir qué acciones puede tomar el controlador de semáforo.
**Clase principal:** `ProjectActionSpace` — dos acciones posibles. Sabe generar una acción aleatoria (`sample()`) y validar si una acción es válida (`contains()`).
**Nota importante (ver también Sección 19):** la acción **no** es un interruptor mantener/cambiar, como decía la especificación original: `sumo_rl` (`TrafficSignal.set_next_phase`) trata el número recibido como el **índice de la fase verde de destino**. Con 2 fases, la acción 1 equivale a "cambiar" solo cuando la fase actual es la 0; cuando es la 1, la acción 1 mantiene y la acción 0 cambia. Como cada fase ocupa ~50% del tiempo, la coincidencia con la descripción original se da en aproximadamente la mitad de los pasos. El docstring de la clase ya lo explica (commit `c253d88`); el comportamiento no se cambió, porque todo el pipeline usa esta convención de forma consistente.

### `environments/project_reward_function.py`
**Responsabilidad:** calcular qué tan "buena" fue una transición.
**Clase principal:** `ProjectRewardFunction` — implementa `R = -α·espera - β·cola + γ·flujo - δ·cambio_de_fase`, usando los coeficientes de `RewardConfig`.
**Dato importante:** el "flujo" (`throughput`) se calcula como vehículos que **terminaron** su recorrido en ese paso (dato real de SUMO), no como vehículos presentes — si se contaran los presentes, se estaría premiando la congestión en vez de penalizarla.
**Otro dato importante:** el término `cambio_de_fase` lee `info["phase_change"]`, que pese a su nombre vale 1 cuando se **pidió la fase 1** (`action == 1`), no cuando el semáforo cambió de fase de verdad. Se mantiene así a propósito, porque todos los datos y modelos entrenados usan esa definición; está documentado en el código y sigue abierto en `TODO.md`.

### `environments/dream_environment.py`
**Responsabilidad:** el corazón de la idea de "World Model" — un entorno compatible con Gymnasium que imagina transiciones usando el `LatentDynamicsLSTM` ya entrenado, **sin ejecutar SUMO ni una sola vez**.
**Clase principal:** `DreamEnvironment`.
**Cómo funciona:** cada episodio imaginado se siembra con una ventana real de contexto (16 pasos, tomados de un episodio real ya codificado) — no puede empezar de la nada, porque el LSTM necesita ese contexto para predecir bien. Desde ahí, las acciones que se prueban son las que decide el agente (no las que de verdad ocurrieron).
**Por qué trunca a 7 pasos (`max_dream_steps=7`):** se investigó con evidencia real que el LSTM extrapola mal cuando se le pide imaginar rachas de la misma acción más largas que las que vio durante su entrenamiento — 7 es el límite calibrado empíricamente para mantenerse dentro de lo confiable.
**Por qué recorta la recompensa imaginada:** para evitar que una predicción extrema y poco realista del LSTM (fuera de su rango de confianza) distorsione el aprendizaje del controlador que se entrena encima.

### `environments/encoded_traffic_environment.py`
**Responsabilidad:** el puente entre SUMO real y el PPO entrenado en el sueño.
**Clase principal:** `EncodedTrafficEnvironment` — envuelve `TrafficEnvironment`, normaliza el estado crudo con el mismo `scaler.pkl` que usó el Autoencoder durante su entrenamiento, y lo pasa por el Encoder congelado para producir `z` antes de que la política PPO lo vea.
**Por qué existe:** el PPO del sueño nunca vio el estado crudo de 26 dimensiones — solo aprendió sobre `z` de 16. Sin este puente, sería imposible evaluar esa política contra SUMO real. Es el único punto del proyecto donde el Encoder corre contra datos de SUMO en vivo, en vez de contra episodios ya codificados de antemano.

### `environments/reseeding_wrapper.py`
**Responsabilidad:** asignar una semilla nueva de SUMO en cada `reset()` que no reciba una explícitamente.
**Clase principal:** `ReseedingWrapper` (envoltorio de Gymnasium).
**Por qué existe:** `sumo_rl` reutiliza la última semilla de tráfico usada si no se le da una nueva explícitamente — sin este envoltorio, muchos episodios "distintos" en realidad compartirían la misma realización de tráfico, y la única variedad real vendría de las acciones aleatorias, no del tráfico en sí. Se usa tanto en la recolección del dataset (`collect_dataset.py`) como en el entrenamiento y la evaluación periódica del RL directo.

### `environments/single-intersection/`
**Responsabilidad:** contiene los archivos que describen la red de SUMO propia del proyecto: la geometría de calles (`.net.xml`), la demanda de vehículos (`.rou.xml`), y la configuración de la simulación (`.sumocfg`). Es una intersección de 4 brazos con un semáforo de 2 fases.
**Demanda actual (asimétrica):** 500 veh/h por brazo en la vía principal (Norte-Sur), 150 veh/h por brazo en la vía secundaria (Este-Oeste), constante durante todo el episodio. Se conserva un respaldo de la demanda simétrica anterior (350 veh/h en las 4 direcciones) en `single-intersection_symmetric_backup.rou.xml`.

### `models/representation/encoder.py`
**Responsabilidad:** la red neuronal que comprime el estado.
**Clase principal:** `Encoder` — dos capas totalmente conectadas: de `input_dim` (26) a `hidden_dim` (16), con una activación en el medio, y de ahí a `latent_dim` (16).

### `models/representation/decoder.py`
**Responsabilidad:** la red simétrica al Encoder — reconstruye el estado a partir de `z`.
**Clase principal:** `Decoder` — de `latent_dim` a `hidden_dim` (con activación), y de vuelta a `input_dim`.

### `models/representation/autoencoder.py`
**Responsabilidad:** unir `Encoder` y `Decoder` en un solo modelo entrenable, **determinista** (no un VAE — sin muestreo aleatorio ni divergencia KL).
**Clase principal:** `Autoencoder`, con `encode(x)`, `decode(z)`, `reconstruct(x)` y `forward(x)`.

### `models/world_model/lstm.py`
**Responsabilidad:** el modelo temporal — la pieza que realmente aprende la dinámica del tráfico en el espacio latente.
**Clase principal:** `LatentDynamicsLSTM` — recibe una secuencia de 16 pasos de `(z_t, acción_t)` y predice `(ẑ_{t+1}, r̂_{t+1})`, tanto el siguiente estado latente como la recompensa asociada, en una sola pasada.
**Cómo está implementado:** con `torch.nn.LSTM` estándar de PyTorch — no con las ecuaciones de compuertas (olvido, entrada, salida) escritas a mano, que era la implementación que sugería la propuesta original (ver Sección 19 para el detalle de esta diferencia).
**`action_dim=2`:** la acción se codifica como one-hot, no como un número crudo, para no sugerir un orden entre las dos opciones que no existe.

### `models/world_model/transformer.py`
**Responsabilidad:** alternativa al LSTM evaluada en el Experimento 3.
**Clase principal:** `LatentDynamicsTransformer` — proyecta cada par `(z, acción)` a 128 dimensiones, suma una codificación posicional sinusoidal y lo pasa por 2 capas de encoder Transformer estándar de PyTorch (4 cabezas de atención). Como el LSTM, usa la salida del último paso y tiene las mismas dos cabezas de salida `(ẑ, r̂)`. No usa máscara causal: el valor a predecir está fuera de la ventana, así que no hay fuga de información futura.

### `models/world_model/tsmixer.py`
**Responsabilidad:** segunda alternativa al LSTM evaluada en el Experimento 3, sin recurrencia ni atención.
**Clase principal:** `LatentDynamicsTSMixer` — 2 bloques que alternan una mezcla temporal (una capa densa que opera sobre el eje del tiempo) y una mezcla de variables (un MLP), cada una con normalización y conexión residual. Exige una longitud de secuencia fija (16), porque sus capas temporales operan directamente sobre esa dimensión.

### `models/world_model/base.py`
**Responsabilidad:** define `TemporalModel`, un contrato (`Protocol` de Python) con la interfaz común (`forward(latent_sequence, action_sequence) -> (z_hat, r_hat)`) que cualquier arquitectura temporal alternativa (Transformer, TSMixer) debería implementar para ser intercambiable con el LSTM sin tocar el resto del sistema.
**Estado actual:** lo implementan las tres arquitecturas, `LatentDynamicsLSTM`, `LatentDynamicsTransformer` y `LatentDynamicsTSMixer`. El sistema final usa el LSTM (ver el Experimento 3 en la Sección 11).

### `training/train_autoencoder.py`
**Responsabilidad:** el bucle completo de entrenamiento del Autoencoder — ver Sección 10 para el detalle paso a paso.

### `training/train_world_model.py`
**Responsabilidad:** entrena `LatentDynamicsLSTM` sobre las secuencias de `z` codificadas del dataset. Fija semillas aleatorias y guarda los hiperparámetros usados junto al checkpoint.

### `training/train_world_model_raw.py`
**Responsabilidad:** el mismo entrenamiento, pero sobre el estado crudo normalizado en vez de `z` — es la mitad del Experimento 0 (la comparación que decide si el Autoencoder realmente vale la pena).

### `training/train_world_model_transformer.py`, `training/train_world_model_tsmixer.py`
**Responsabilidad:** entrenan el Transformer y el TSMixer del Experimento 3. Importan el protocolo de entrenamiento de `train_world_model.py` (semilla, optimizador, épocas, early stopping, pérdida) en vez de copiarlo, y **leen** el `reward_scaler.json` que generó el entrenamiento del LSTM en vez de recalcularlo. Así las tres arquitecturas se entrenan en condiciones idénticas, por construcción.

### `training/train_controller.py`
**Responsabilidad:** entrena PPO (Stable-Baselines3) dentro de `DreamEnvironment`.
**Detalle importante:** la selección del mejor checkpoint **no se hace mirando la recompensa imaginada** — se investigó y se encontró que no tiene ninguna relación con el desempeño real (correlación de Pearson ≈ 0.08) — sino evaluando periódicamente contra SUMO real, a través de `EncodedTrafficEnvironment`. Usa `VecNormalize` para normalizar la recompensa, corrigiendo un problema donde la red de valor de PPO no aprendía nada (`explained_variance` ≈ 0).

### `training/train_controller_direct.py`
**Responsabilidad:** entrena PPO directamente contra `TrafficEnvironment` (SUMO real), sin Autoencoder ni Dream Environment — el baseline de RL directo que pide la Sección 18 de la propuesta.
**Detalle importante:** usa `VecNormalize` para la recompensa **y** para las observaciones — el estado crudo de 26 dimensiones tiene una escala muy desigual entre variables (hasta ~1764 veces de diferencia entre la dimensión más y menos variable), algo que `z` no tiene porque ya viene normalizado por el Autoencoder.
**Presupuesto:** 10,000 pasos reales de entrenamiento (`DIRECT_TOTAL_TIMESTEPS`). Una verificación con 30,000 pasos, hecha sin modificar el script, está archivada en `models/checkpoints/controller_direct_30k/` (ver Sección 11).

### `evaluation/autoencoder_evaluation.py`
**Responsabilidad:** funciones reutilizables para medir y visualizar un Autoencoder ya entrenado.

### `evaluation/evaluate_autoencoder.py`
**Responsabilidad:** el script que realmente se ejecuta: carga el mejor checkpoint del Autoencoder, evalúa contra el conjunto de prueba, e imprime/guarda los resultados.

### `evaluation/world_model_evaluation.py`
**Responsabilidad:** funciones reutilizables para evaluar el modelo temporal: `rollout_episode` (predicción autorregresiva a varios horizontes), `predict_next_step` (un solo paso, reutilizado también por `DreamEnvironment`), y utilidades para cargar checkpoints y su escalador de recompensa. `build_world_model` construye la arquitectura correcta (LSTM, Transformer o TSMixer) a partir de la clave `architecture` del `.json` de cada checkpoint; los checkpoints sin esa clave son LSTM.

### `scripts/collect_dataset.py`
**Responsabilidad:** generar episodios reales usando `TrafficEnvironment` y guardarlos, uno por archivo, en `datasets/raw/`. Genera 80 episodios de 60 pasos cada uno, con acciones aleatorias y **una semilla de SUMO distinta por episodio** (`seed_start + episode_idx`) — corrigiendo un bug donde los 40 episodios originales compartían la misma semilla y toda la variedad venía solo de las acciones, no del tráfico.

### `scripts/split_dataset.py`
**Responsabilidad:** decidir qué episodios completos van a entrenamiento (56), validación (12) y prueba (12) — nunca mezclando transiciones sueltas de un mismo episodio entre particiones distintas.

### `scripts/merge_dataset.py`
**Responsabilidad:** contiene `merge_files()`, usada internamente por `split_dataset.py`. Ejecutado directamente, junta todos los episodios sin separar — un modo pensado solo para exploración, nunca para entrenar.

### `scripts/normalize_dataset.py`
**Responsabilidad:** normaliza los datos calculando media y desviación **únicamente con el split de entrenamiento**, guardando el resultado en `scaler.pkl` (reutilizado también por `EncodedTrafficEnvironment`).

### `scripts/visualize_dataset.py`
**Responsabilidad:** imprime estadísticas básicas de un split ya procesado y guarda un histograma de la recompensa.

### `scripts/test_environment.py`
**Responsabilidad:** una prueba manual y visual (no parte de `pytest`) que crea el entorno, ejecuta pasos con acciones aleatorias, e imprime la recompensa de cada paso.

### `scripts/encode_latent_dataset.py`
**Responsabilidad:** corre el Encoder ya entrenado (congelado) sobre cada split del dataset, produciendo los archivos `*_latent.npz` con las secuencias de `z` que usa el LSTM.

### `scripts/evaluate_world_model.py`
**Responsabilidad:** ejecuta la evaluación del LSTM (Experimento 1): compara contra un baseline "nada cambia", en horizontes de 1 a 10 pasos.

### `scripts/prepare_raw_sequence_dataset.py`, `scripts/evaluate_world_model_raw.py`, `scripts/compare_experiment_0.py`
**Responsabilidad conjunta:** el Experimento 0 completo. El primero prepara secuencias sobre el estado crudo (en vez de `z`); el segundo evalúa el LSTM entrenado sobre ese estado crudo; el tercero compara ambos resultados y decide si el Autoencoder se queda en el sistema.

### `scripts/evaluate_world_model_transformer.py`, `scripts/evaluate_world_model_tsmixer.py`, `scripts/compare_experiment_3.py`
**Responsabilidad conjunta:** el Experimento 3 completo. Los dos primeros evalúan cada alternativa con exactamente el mismo protocolo que `evaluate_world_model.py` (mismo conjunto de prueba, horizontes y baseline). El tercero compara las tres arquitecturas por `reward_mse` en cada horizonte, reporta el número de parámetros de cada una y aplica un criterio de decisión fijado antes de ver los resultados.

### `scripts/evaluate_controller.py`
**Responsabilidad:** autoevaluación del PPO del sueño **dentro** del propio `DreamEnvironment` — útil como diagnóstico rápido, pero su propio docstring aclara que no es una medición de desempeño real contra SUMO.

### `scripts/evaluate_controller_sumo.py`
**Responsabilidad:** evalúa el PPO del sueño contra SUMO real (a través de `EncodedTrafficEnvironment`), comparado contra tiempo fijo y acción aleatoria.

### `scripts/evaluate_direct_vs_dream.py`
**Responsabilidad:** compara el PPO del sueño (oficial) contra el PPO de RL directo, ambos en SUMO real.

### `scripts/evaluate_final_comparison.py`
**Responsabilidad:** el script de evaluación final y definitivo — compara, en SUMO real, el PPO del sueño, el PPO directo, control de tiempo fijo, y la regla trivial "pedir siempre la fase contraria". Es la tabla que responde la pregunta de investigación central del proyecto (Sección 11).

### `scripts/analyze_controller_actions.py`
**Responsabilidad:** herramienta de diagnóstico usada durante la investigación de una anomalía en el recorte de recompensa del `DreamEnvironment` — mide si el PPO está "explotando" ese recorte en vez de aprender control genuino.

### `ver_controlador.py`
**Responsabilidad:** script de demostración: abre la GUI de SUMO y muestra en vivo al PPO del sueño controlando el semáforo. No forma parte del pipeline ni de los tests.

### `pytest.ini`
**Responsabilidad:** le dice a `pytest` que solo busque pruebas dentro de `tests/`, y agrega la raíz del proyecto a la ruta de Python automáticamente.

### `requirements.txt`
**Responsabilidad:** lista de librerías necesarias: `numpy`, `pandas`, `matplotlib`, `jupyter`, `ipykernel`, `traci`, `sumolib`, `sumo-rl`, `gymnasium`, `stable-baselines3`, `torch`, `pytest`.

### `.gitignore`
**Responsabilidad:** excluye archivos generados automáticamente: entorno virtual, caché de Python, datasets generados, checkpoints entrenados (`.pt`, `.zip`, `.npz`, `.pkl`), e imágenes de resultados. Los `.json` de hiperparámetros junto a cada checkpoint **sí** se versionan.

### `README.md`, `CLAUDE.md`, `PROJECT_STATUS.md`, `TODO.md`
**Responsabilidad:** documentación operativa del proyecto — la puerta de entrada (`README.md`), el contexto permanente para retomar el trabajo con asistencia de IA (`CLAUDE.md`), el registro cronológico de resultados y hallazgos (`PROJECT_STATUS.md`, el documento más extenso y detallado de todos), y la lista de pendientes (`TODO.md`). Ver Sección 19 sobre partes de estos documentos que quedaron desactualizadas.

### `LICENSE`
Archivo de licencia presente en el repositorio, sin contenido definido todavía.

---

## 5. Flujo completo del proyecto

```
SUMO (demanda asimétrica: 500 veh/h vía principal, 150 veh/h vía secundaria)
    ↓
TrafficEnvironment (único punto de contacto con SUMO)
    ↓
CustomStateBuilder (lee TraCI y arma el estado) → TrafficState → vector de 26 números
    ↓
collect_dataset.py (80 episodios, semilla de SUMO distinta por episodio)
    ↓
split_dataset.py (separa episodios completos en train/validation/test)
    ↓
normalize_dataset.py (normaliza usando solo estadísticas de train)
    ↓
train_autoencoder.py (entrena Encoder + Decoder)
    ↓
encode_latent_dataset.py (codifica los splits a z, 16 dims)
    ↓
train_world_model.py (entrena LatentDynamicsLSTM sobre z)
    ↓
evaluate_world_model.py (Experimento 1: supera al baseline persistente, 10/10 horizontes)
    ↓
train_world_model_raw.py + compare_experiment_0.py (Experimento 0: z sigue ganando, 9/10)
    ↓
train_world_model_{transformer,tsmixer}.py + compare_experiment_3.py
    (Experimento 3: el LSTM gana en los 10 horizontes; se mantiene)
    ↓
DreamEnvironment (usa el LSTM congelado para imaginar transiciones, sin SUMO)
    ↓
train_controller.py (PPO entrenado SIN tocar SUMO durante el aprendizaje,
                      seleccionado por evaluación periódica en SUMO real)
    │
    └── en paralelo, sin relación de dependencia:
        train_controller_direct.py (PPO entrenado DIRECTO contra SUMO real)
    ↓
evaluate_final_comparison.py (comparación final: PPO del sueño vs. PPO directo vs.
                               tiempo fijo vs. regla trivial, en SUMO real)
```

---

## 6. Configuración

| Config | Contiene | Usada por |
|---|---|---|
| `EnvironmentConfig` | Ruta de la red SUMO, uso de GUI, duración, `single_agent`, semilla | `TrafficEnvironment` |
| `RewardConfig` | Coeficientes α, β, γ, δ de la recompensa | `ProjectRewardFunction` |
| `RepresentationConfig` | Dimensiones y entrenamiento del Autoencoder | `train_autoencoder.py`, `autoencoder_evaluation.py` |
| `WorldModelConfig` | `latent_dim` (derivado de `RepresentationConfig`), `action_dim`, `sequence_length`, `hidden_dim` | `train_world_model.py`, `train_world_model_transformer.py`, `train_world_model_tsmixer.py` |
| `ControllerConfig` | Hiperparámetros de PPO, `seed` (default 1 para el sueño), `normalize_reward`, `reward_clip` | `train_controller.py`, `train_controller_direct.py` |

---

## 7. Dataset

- **80 episodios**, cada uno con una semilla de SUMO distinta — corrigiendo el bug donde los 40 episodios originales compartían la misma semilla.
- Split: 56 episodios de entrenamiento, 12 de validación, 12 de prueba (3360/720/720 transiciones), sin NaN ni Inf.
- Normalización ajustada solo con el split de entrenamiento (`scaler.pkl`), reutilizado también por `EncodedTrafficEnvironment`.
- Cada registro de una transición: `states`, `actions`, `rewards`, `next_states`, `terminated`, `truncated`, `episode_id`, `time_step`.
- Vector de estado: 26 dimensiones (4 carriles × 5 variables + 4 de fase one-hot + 2 de tiempo de fase).

---

## 8. Environments

`TrafficEnvironment` sigue siendo el único punto de contacto directo con SUMO. Sobre él se construyen tres envoltorios especializados: `EncodedTrafficEnvironment` (traduce el estado real a `z`), `ReseedingWrapper` (asegura variedad de tráfico real), y `DreamEnvironment` (opera enteramente en el espacio latente, sin tocar SUMO).

---

## 9. Modelos

- **`Encoder`/`Decoder`/`Autoencoder`**: comprimen el estado de 26 a 16 dimensiones. Determinista, no un VAE.
- **`LatentDynamicsLSTM`**: recibe una ventana de 16 pasos de `(z, acción)` y predice `(ẑ_{t+1}, r̂_{t+1})`. Implementado con `torch.nn.LSTM` estándar. Es el modelo temporal del sistema final.
- **`LatentDynamicsTransformer`** y **`LatentDynamicsTSMixer`**: alternativas con la misma entrada y la misma salida (interfaz `TemporalModel`), evaluadas en el Experimento 3. Ninguna mejora al LSTM.

Ambos se entrenan por separado, en ese orden — el Autoencoder primero, congelado después, y el LSTM se entrena sobre los `z` que produce.

---

## 10. Entrenamiento

Bucle de entrenamiento del Autoencoder (`train_autoencoder.py`), paso a paso:
1. Carga los splits normalizados.
2. Construye `Encoder`+`Decoder` con `RepresentationConfig`.
3. Fija semillas aleatorias.
4. Entrena minimizando el error de reconstrucción (MSE) entre el estado original y el reconstruido.
5. Guarda el mejor checkpoint según la pérdida de validación, junto con sus hiperparámetros en un `.json`.

Los demás bucles (`train_world_model.py` y sus variantes de Transformer y TSMixer, `train_controller.py`, `train_controller_direct.py`) siguen el mismo patrón: semillas fijas, guardado del mejor checkpoint según una métrica de validación, e hiperparámetros persistidos.

---

## 11. Evaluación

### Resultado oficial (segunda ronda del escenario asimétrico, `PROJECT_STATUS.md`)

Con el presupuesto original del RL directo (10,000 pasos de entrenamiento; 13,000 interacciones reales por semilla):

| Política | Episodios | Media | Desv. estándar | Gana a tiempo fijo | Episodios < -600 | Interacciones reales de SUMO |
|---|---|---|---|---|---|---|
| **PPO del sueño (World Model)** | 90 | **-326.79** | 147.42 | 75/90 | 5/90 | **13,800 en total para 3 semillas** (4,800 del dataset, recolectado una sola vez y compartido, + 3 × 3,000 de selección) → ~4,600 por semilla |
| PPO directo (RL directo) | 90 | -453.74 | 217.13 | 49/90 | 23/90 | 39,000 en total (3 × 13,000) → 13,000 por semilla |
| Tiempo fijo (ciclo=5) | 30 | -411.27 | 51.75 | — | 0/30 | — |
| Regla "pedir fase contraria" | 30 | -502.53 | 118.11 | 9/30 | 7/30 | — |

**Significancia estadística:**
- A nivel de episodio (90 vs. 90): la diferencia equivale a 4.56 errores estándar (p ≈ 1e-5).
- A nivel de semilla de entrenamiento (3 vs. 3): **no alcanza significancia estadística** (p = 0.145).
- Con este presupuesto, la diferencia más robusta es la **consistencia**: las tres semillas del sueño superan a las tres del directo, y el directo varía mucho más entre semillas (una de sus tres colapsó de vuelta a la regla trivial).

### Verificación: RL directo con 3x presupuesto (30,000 pasos)

Para saber si la desventaja del RL directo se debía a su presupuesto, sus 3 semillas se reentrenaron con 30,000 pasos (sin modificar el script ni reemplazar los checkpoints oficiales) y se evaluaron con el mismo protocolo:

| Política | Media (90 episodios) | Mediana | Desv. de las medias por semilla | Episodios < -600 | Interacciones reales por semilla |
|---|---|---|---|---|---|
| PPO directo, 10,000 pasos | -453.74 | -381.75 | 96.2 | 23/90 | 13,000 |
| PPO directo, 30,000 pasos | -335.24 | -271.60 | 30.6 | 11/90 | 39,000 |
| PPO del sueño (World Model) | -326.79 | -293.00 | 19.3 | 5/90 | ~4,600 |

Con 30,000 pasos, la semilla que colapsaba a la regla trivial deja de hacerlo, y el RL directo **alcanza un desempeño comparable** al del World Model: la diferencia de medias baja a 8.46 puntos y deja de ser significativa (p = 0.74 a nivel de episodio, 0.71 a nivel de semilla). El directo gana además la mayoría de las comparaciones episodio a episodio (159 de 270). Pero cada una de sus semillas consume 39,000 interacciones reales, **~8.5 veces las del World Model**. Conclusión precisa: la ventaja demostrada del World Model es de **eficiencia en interacciones reales**; no es mejor control cuando el RL directo dispone de presupuesto de sobra. Salvedad: con más presupuesto, el directo también elige su mejor checkpoint entre más evaluaciones (30 en vez de 10), y esta verificación no separa ese efecto del de entrenar más.

### Resultado del Experimento 1 (LSTM vs. baseline persistente)

El LSTM supera al baseline "nada cambia" en los 10 horizontes evaluados. El error de predicción de recompensa a un paso es del 2.7% del error del baseline (mejoró notablemente frente a un dataset anterior más pequeño, en parte por un modelo mejor y en parte por un conjunto de prueba sin valores extremos fuera de rango).

### Resultado del Experimento 0 (Autoencoder vs. estado crudo)

El Autoencoder gana en 9 de 10 horizontes; su ventaja se invierte solo en el horizonte más largo (10).

### Resultado del Experimento 3 (LSTM vs. Transformer vs. TSMixer)

El LSTM tiene el menor error de predicción de recompensa (`reward_mse`) en los **10 horizontes**, con una reducción mediana de 38.1% frente al Transformer y de 56.1% frente a TSMixer; se mantiene. TSMixer sigue perdiendo en los 10 horizontes incluso entrenado 300 épocas sin early stopping. **Hallazgo:** el Transformer predice mejor el estado latente que el LSTM y tiene menor pérdida de validación, pero predice peor la recompensa — el mismo patrón que el reward imaginado del PPO (correlación ≈ 0.08 con el real): una métrica de entrenamiento que no predice la que realmente importa.

---

## 12. Tests

**62 funciones de test, en 12 archivos, todas pasando.** Cubren desde el entorno de SUMO hasta la pila completa de `VecNormalize` sincronizada entre entrenamiento y evaluación.

```powershell
pytest -v
```

---

## 13. Comandos importantes

```powershell
# Entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pipeline de datos
python scripts\collect_dataset.py
python scripts\split_dataset.py
python scripts\normalize_dataset.py

# Representación y dinámica
python training\train_autoencoder.py
python scripts\encode_latent_dataset.py
python training\train_world_model.py
python scripts\evaluate_world_model.py

# Experimento 0
python scripts\prepare_raw_sequence_dataset.py
python training\train_world_model_raw.py
python scripts\evaluate_world_model_raw.py
python scripts\compare_experiment_0.py

# Experimento 3 (requiere el reward_scaler.json del entrenamiento del LSTM)
python training\train_world_model_transformer.py
python training\train_world_model_tsmixer.py
python scripts\evaluate_world_model_transformer.py
python scripts\evaluate_world_model_tsmixer.py
python scripts\compare_experiment_3.py

# Controladores
python training\train_controller.py
python training\train_controller_direct.py

# Evaluación final
python scripts\evaluate_final_comparison.py

# Tests
pytest -v
```

---

## 14. Qué mostrar en la sustentación

1. **La pregunta de investigación y la arquitectura completa** (diagrama de la Sección 5).
2. **Las tablas de la Sección 11** — el resultado oficial y su verificación con 3x presupuesto: el World Model alcanza el mismo nivel de control con ~8.5 veces menos interacciones reales.
3. **La historia de cómo se llegó ahí, no solo el número final** (Sección 20) — al menos tres bugs reales encontrados y corregidos con evidencia demuestran rigor metodológico, no solo un resultado.
4. **La honestidad sobre las limitaciones**: episodios catastróficos que persisten, falta de significancia estadística a nivel de semilla, que con suficiente presupuesto el RL directo alcanza al World Model, y por qué se mantuvo el LSTM frente a Transformer y TSMixer.

---

## 15. Preguntas probables de la profesora

**Sobre los fundamentos:**

1. **¿Por qué usar un Autoencoder y no entrenar directo sobre el estado?**
 Porque el estado (26 números) podría tener variables redundantes; el Experimento 0 comprueba si comprimir realmente ayuda, en vez de asumirlo.

2. **¿Por qué no usar una red convolucional (CNN)?**
 El estado es un vector de números, no una imagen; las convoluciones sirven para datos con estructura espacial.

3. **¿Qué contiene exactamente un estado?**
 26 números: por cada uno de los 4 carriles, número de vehículos, cola, espera, velocidad y ocupación (20 valores); 4 números de fase (one-hot); 2 de tiempo de fase.

4. **¿Por qué separar entrenamiento, validación y prueba por episodios completos?**
 Para no filtrar información entre conjuntos — se verifica con un test automatizado dedicado.

5. **¿Cómo se define la recompensa?**
 `R = -α·espera - β·cola + γ·flujo - δ·cambio_de_fase`; el flujo se mide como vehículos que llegaron a destino, no presentes (para no premiar la congestión).

**Sobre el modelo y el resultado final:**

6. **¿Por qué el resultado no es "estadísticamente significativo"?**
 Con solo 3 semillas de entrenamiento por método, la prueba a ese nivel no tiene suficiente potencia — la diferencia es clara y consistente, pero afirmar significancia formal exigiría más corridas.

7. **¿Por qué el PPO directo tuvo tanta variación entre semillas?**
 Con 10,000 pasos, una de sus tres semillas colapsó a la regla trivial: su mejor checkpoint se quedó en esa meseta temprana. Con 30,000 pasos esa semilla deja de colapsar y la variación entre semillas baja de 96.2 a 30.6 — era un síntoma de presupuesto insuficiente, no una propiedad del método.

8. **¿Cuántas interacciones reales con SUMO usó cada método?**
 El World Model usa 13,800 en total para 3 semillas (~4,600 por semilla; el dataset se recolecta una sola vez y se comparte). El RL directo usa 13,000 por semilla con 10,000 pasos de entrenamiento, y 39,000 por semilla con 30,000 pasos, que es lo que necesita para alcanzar al World Model.

9. **¿Por qué se mantuvo el LSTM y no Transformer o TSMixer?**
 Porque se probaron (Experimento 3) y el LSTM predice mejor la recompensa en los 10 horizontes. Se pospusieron hasta estabilizar el núcleo, y se ejecutaron después sobre la misma interfaz (`TemporalModel`), sin rediseñar el sistema.

10. **¿Qué son los "episodios catastróficos" y por qué siguen ahí?**
 Un número pequeño de episodios donde el controlador se comporta mucho peor que el promedio; se investigaron varias hipótesis, algunas se descartaron con evidencia, pero la causa completa no se identificó — limitación abierta, documentada.

11. **¿Qué es `VecNormalize` y por qué hizo falta?**
 La red de valor de PPO no aprendía nada (`explained_variance` ≈ 0) porque las recompensas no estaban en escala manejable; normalizarlas lo resolvió para el sueño, y el directo necesitó además normalizar sus observaciones.

12. **¿Cómo se sabrá si el Autoencoder realmente ayuda?**
 Comparando, con el mismo modelo temporal, el error de predicción usando `z` contra el estado crudo — el Experimento 0.

13. **¿Qué garantiza que el proyecto sea reproducible?**
 Semillas fijas, hiperparámetros junto a cada checkpoint, y 62 tests automatizados.

14. **Entonces, ¿el World Model controla mejor que el RL directo?**
 Con el mismo presupuesto original, sí; pero con el triple de presupuesto el RL directo lo alcanza. Lo que el World Model demuestra es eficiencia: llega al mismo nivel de control con ~8.5 veces menos interacciones reales, que es justo lo que pregunta la pregunta de investigación.

---

## 16. Qué falta por desarrollar

- **Más semillas por método** (siguen siendo 3), para afirmar o descartar diferencias a nivel de semilla.
- **Una curva de desempeño frente a interacciones reales del RL directo** (hoy solo dos puntos: 13,000 y 39,000 por semilla).
- **Investigar la causa completa de los episodios catastróficos** que persisten en ambos métodos.
- **Demanda de tráfico variable en el tiempo** (hoy es asimétrica pero constante).

---

## 17. Glosario completo

- **TraCI:** protocolo que permite controlar y leer datos de una simulación de SUMO desde un programa externo.
- **Gymnasium:** interfaz estándar de entornos de RL (`reset`, `step`, espacios de acción/observación).
- **Episode (episodio):** una simulación completa, desde el inicio hasta que termina o se trunca.
- **Transition (transición):** un solo paso: `(estado, acción, recompensa, siguiente estado)`.
- **Latent Space (espacio latente):** el conjunto de vectores comprimidos que puede producir un Encoder.
- **Encoder / Decoder:** redes que comprimen / reconstruyen una entrada.
- **Checkpoint:** archivo con los pesos aprendidos de una red, para reutilizarla sin reentrenar.
- **Batch (lote) / Epoch (época):** grupo de ejemplos procesado junto / una pasada completa por los datos de entrenamiento.
- **MSE:** Error Cuadrático Medio, mide la distancia entre predicción y valor real.
- **ReLU / Adam:** función de activación estándar / algoritmo de optimización con tasa de aprendizaje adaptativa.
- **Forward / Inference:** pasar una entrada por la red para obtener una salida / usar un modelo ya entrenado sin seguir ajustándolo.
- **Dream Environment:** entorno simulado que usa un modelo aprendido para imaginar transiciones sin ejecutar el simulador real.
- **Semilla (seed):** número inicial que determina una secuencia "aleatoria" reproducible.
- **Reseed:** asignar una semilla nueva en cada episodio, para variedad real.
- **`VecNormalize`:** utilidad de Stable-Baselines3 que normaliza recompensas y/u observaciones durante el entrenamiento de RL.
- **`explained_variance`:** qué tan bien la red de valor de PPO predice los retornos reales; cerca de 0 = no aprendió nada útil.
- **PPO (Proximal Policy Optimization):** algoritmo de Reinforcement Learning usado para entrenar el controlador.
- **Transformer:** arquitectura basada en atención, que relaciona todos los pasos de una secuencia entre sí en vez de recorrerlos uno a uno.
- **TSMixer:** arquitectura para series de tiempo hecha solo con capas densas, que alterna mezclas sobre el eje del tiempo y sobre las variables.

---

## 18. Resumen final (lectura de 5 minutos)

Este proyecto construyó, de punta a punta, un sistema de World Models para control de semáforos: un Autoencoder que comprime el estado del tráfico, un LSTM que aprende a predecir cómo evoluciona ese estado comprimido, un entorno imaginado (Dream Environment) que permite entrenar un controlador PPO sin tocar el simulador, y un baseline de RL directo para comparar. En el camino se encontraron y corrigieron varios bugs reales — una lectura incorrecta de la fase del semáforo, un puente entre SUMO y PPO que no normalizaba los datos correctamente, y un criterio de selección de "mejor modelo" que no tenía relación con el desempeño real — cada uno diagnosticado con evidencia antes de aplicar el arreglo.

El resultado final, verificado con 3 semillas de entrenamiento por método y evaluado en SUMO real: con el presupuesto original del RL directo, el controlador entrenado en el sueño lo supera de forma clara y consistente usando aproximadamente un tercio de las interacciones reales, aunque la diferencia no alcanza significancia estadística formal con solo 3 semillas. Al triplicar el presupuesto del RL directo, este alcanza un desempeño comparable, pero consumiendo ~8.5 veces más interacciones reales que el World Model: la ventaja del World Model es de eficiencia. Además, el Experimento 3 confirmó que el LSTM predice mejor que un Transformer y que un TSMixer, y quedan episodios catastróficos sin explicar del todo en ambos métodos. Es un resultado honesto: no perfecto, pero real, medido con rigor, y con sus límites declarados explícitamente.

---

## 19. Observaciones técnicas y deuda conocida

- **La semántica de la acción no es "mantener/cambiar"**: `sumo_rl` la trata como el índice de fase verde destino. La documentación ya lo dice (docstring de `ProjectActionSpace`, commit `c253d88`; Sección 12 de la propuesta, `30c3fae`); el comportamiento no se cambió porque el pipeline completo usa la convención de forma consistente.
- **`info["phase_change"]` mide si se pidió la fase 1, no si el semáforo cambió de fase de verdad**, y la recompensa penaliza eso. Documentado en el código; corregir el cálculo sigue pendiente en `TODO.md`, porque invalidaría todos los resultados entrenados con la definición actual.
- **El LSTM se implementó con `torch.nn.LSTM` estándar**, no replicando manualmente las ecuaciones de compuertas como sugería la propuesta original — una simplificación de implementación razonable que no cambia el comportamiento del modelo.
- **El Autoencoder es determinista, no un VAE** — la propuesta original mencionaba VAE con reparametrización; se implementó la versión más simple, suficiente para el Experimento 0.
- **`compare_experiment_0.py` tiene una nota interna desactualizada** ("8 vs. 26 dimensiones" cuando el espacio latente real es 16) — no afecta el resultado.
- **`CLAUDE.md` tiene partes desactualizadas**: marca como pendientes bloques que ya están completos. `README.md` se reescribió en el commit `dd0e547` y ya refleja el estado real.
- La demanda de tráfico actual es **constante en el tiempo** dentro de cada episodio — no varía por hora pico.
- El Dream Environment no implementa un "búfer de planificación de acciones candidatas" como describía conceptualmente la propuesta original (Sección 16) — en su lugar, se usa como entorno de entrenamiento completo para PPO, un diseño distinto pero que cumple el mismo propósito de fondo (aprender sin tocar SUMO).

---

## 20. Historia del proyecto: de la primera intersección al resultado final

Un resumen cronológico de los hitos más importantes, útil para entender *por qué* el proyecto se ve como se ve hoy:

1. **Construcción del entorno**: SUMO, la intersección propia, `TrafficEnvironment`, el estado de 26 dimensiones y la recompensa propia — reemplazando la observación nativa de `sumo-rl` por una diseñada a medida.
2. **Pipeline de datos**: recolección, división por episodio (evitando fuga de datos), normalización.
3. **Autoencoder**: arquitectura, entrenamiento, evaluación, con hiperparámetros persistidos junto a cada checkpoint.
4. **Bug crítico #1 — lectura de la fase del semáforo**: se descubrió que el estado leía la fase del semáforo de una fuente de TraCI que `sumo_rl` nunca actualiza, dejando esa variable congelada durante toda la recolección de datos. Se corrigió leyendo la fase real desde `sumo_rl` directamente, y se regeneró el dataset completo.
5. **Modelo temporal LSTM**: implementado, entrenado, y validado en el Experimento 1 contra un baseline persistente.
6. **Experimento 0**: comparación controlada entre `z` y el estado crudo — el Autoencoder demostró su valor con evidencia, no por asunción.
7. **Dream Environment**: construido, con una limitación real descubierta (extrapolación pobre en rachas de acción largas) y mitigada con un límite de pasos calibrado empíricamente.
8. **Primer controlador PPO**: entrenado en el sueño, con una investigación seria de por qué su primera evaluación parecía inconsistente.
9. **Bug crítico #2 — el puente SUMO-PPO no normalizaba**: se descubrió que `EncodedTrafficEnvironment` no aplicaba la misma normalización que usó el Autoencoder, produciendo resultados de evaluación completamente inválidos. Se corrigió y se re-evaluó todo.
10. **Bug crítico #3 — la selección del mejor checkpoint estaba rota**: se encontró que elegir "el mejor modelo" por recompensa imaginada no tenía ninguna relación con el desempeño real (correlación ≈ 0.08). Se cambió el criterio a evaluación periódica en SUMO real.
11. **Bug crítico #4 — la función de valor de PPO no aprendía**: `explained_variance` ≈ 0 en ambos controladores; se corrigió con `VecNormalize`, primero solo la recompensa, y para el RL directo también las observaciones.
12. **Escenario de demanda asimétrica**: diseñado y calibrado para que la solución trivial del problema dejara de ser óptima, permitiendo una comparación real de calidad de control entre métodos.
13. **Verificación de robustez con 3 semillas por método**, en dos rondas de evaluación en SUMO real, llegando al resultado final documentado en la Sección 11.
14. **Experimento 3**: Transformer y TSMixer implementados sobre la interfaz `TemporalModel`, entrenados con el mismo protocolo que el LSTM y evaluados en el mismo conjunto de prueba. El LSTM gana en los 10 horizontes y se mantiene; el Transformer dejó el hallazgo de que una mejor pérdida de validación no garantiza una mejor predicción de la recompensa.
15. **Verificación del presupuesto del RL directo**: con el triple de pasos de entrenamiento, el RL directo alcanza al World Model, pero con ~8.5 veces sus interacciones reales. Esto precisó la conclusión del proyecto: la ventaja del World Model es de eficiencia en interacciones.

Cada uno de estos hitos se investigó con evidencia real (no se aceptó ningún resultado "porque parecía razonable"), y cada corrección se verificó comparando antes/después — es la razón por la que el resultado final, aunque no perfecto, es defendible con confianza.
