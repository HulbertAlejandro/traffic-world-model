# Traffic World Model

Control de un semáforo en una intersección simulada con SUMO mediante un *World Model*
(Ha & Schmidhuber, 2018): el controlador se entrena dentro de un modelo aprendido de la
dinámica del tráfico, sin interactuar con el simulador, y se compara contra un controlador
entrenado directamente en SUMO. Trabajo de grado de Ingeniería de Sistemas, Universidad
del Quindío.

## Estado actual

El sistema está completo, entrenado y evaluado en SUMO real:

```text
SUMO → estado (26 dims) → Autoencoder → z (16 dims) → LSTM → (ẑ_{t+1}, r̂_{t+1})
     → Dream Environment → PPO (entrenado sin tocar SUMO) → evaluación en SUMO real
```

| Experimento | Pregunta | Resultado |
|---|---|---|
| 0 | ¿Ayuda comprimir el estado con un Autoencoder? | Sí: gana en 9 de 10 horizontes; se mantiene |
| 1 | ¿El LSTM predice mejor que un baseline persistente? | Sí, en los 10 horizontes |
| 2 | ¿Un PPO entrenado en el sueño controla bien en SUMO real? | Sí; ver el resultado principal |
| 3 | ¿Transformer o TSMixer mejoran al LSTM? | No: el LSTM gana en los 10 horizontes; se mantiene |

**Resultado principal:** el World Model alcanza un control comparable al del RL directo
con muchas menos interacciones reales con SUMO. Con el presupuesto original del RL
directo (13,000 interacciones por semilla), el World Model controla mejor (-326.79 frente
a -453.74, 3 semillas por método). Al triplicarle el presupuesto, el RL directo lo alcanza
(-335.24), pero consume **~8.5 veces más interacciones reales** (39,000 frente a ~4,600
por semilla).

## Instalación

Requiere Python 3.11 (los resultados se obtuvieron con 3.11.9) y
[SUMO](https://eclipse.dev/sumo/) **1.27.1**, que se instala aparte con el instalador oficial
de SUMO, no con pip. La variable de entorno `SUMO_HOME` debe apuntar a su carpeta de
instalación. `requirements.txt` fija las versiones exactas de las librerías de Python; `traci`
y `sumolib` deben coincidir con la versión de SUMO instalada.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
sumo --version
python scripts\test_environment.py   # prueba de humo del entorno
```

## Estructura

```text
traffic-world-model/
├── configs/            # Dataclasses de configuración: entorno, recompensa, Autoencoder, modelo temporal, PPO
├── datasets/           # Datasets de PyTorch; raw/ y processed/ se generan con los scripts (no versionados)
├── docs/               # PROPUESTA.md y DOCUMENTACION_PROYECTO.md
├── environments/       # TrafficEnvironment (SUMO), estado, acción, recompensa, DreamEnvironment,
│   └── single-intersection/   # red SUMO propia (demanda asimétrica)
├── evaluation/         # Utilidades de evaluación del Autoencoder y del modelo temporal
├── models/
│   ├── representation/ # Encoder, Decoder, Autoencoder
│   ├── world_model/    # TemporalModel, LSTM, Transformer, TSMixer
│   └── checkpoints/    # Pesos generados (no versionados); sí se versionan sus .json de hiperparámetros
├── scripts/            # Puntos de entrada: datos, evaluación, comparaciones
├── tests/              # Pruebas automatizadas (pytest)
├── training/           # Bucles de entrenamiento
├── ver_controlador.py  # Visualiza el PPO del sueño en la GUI de SUMO
├── CLAUDE.md, PROJECT_STATUS.md, TODO.md
├── pytest.ini
└── requirements.txt
```

## Pipeline completo

Cada script usa sus valores por defecto; no requieren argumentos. Orden de ejecución:

```powershell
# 1. Datos: 80 episodios con semilla de SUMO distinta, split por episodio, normalización
python scripts\collect_dataset.py
python scripts\split_dataset.py
python scripts\normalize_dataset.py

# 2. Representación y dinámica (Experimento 1)
python training\train_autoencoder.py
python scripts\encode_latent_dataset.py
python training\train_world_model.py
python scripts\evaluate_world_model.py

# 3. Experimento 0: modelo temporal sobre el estado crudo, comparado contra z
python scripts\prepare_raw_sequence_dataset.py
python training\train_world_model_raw.py
python scripts\evaluate_world_model_raw.py
python scripts\compare_experiment_0.py

# 4. Experimento 3: Transformer y TSMixer (requieren el reward_scaler.json del paso 2)
python training\train_world_model_transformer.py
python training\train_world_model_tsmixer.py
python scripts\evaluate_world_model_transformer.py
python scripts\evaluate_world_model_tsmixer.py
python scripts\compare_experiment_3.py

# 5. Controladores: PPO en el sueño y PPO directo contra SUMO
python training\train_controller.py
python training\train_controller_direct.py

# 6. Comparación final en SUMO real (sueño, directo, tiempo fijo, regla trivial)
python scripts\evaluate_final_comparison.py
```

Los pasos 3 y 4 son experimentos de validación y no los necesitan los pasos 5 y 6.

## Tests

```powershell
pytest -v
```

67 tests en 13 archivos.

## Documentación

- [`docs/DOCUMENTACION_PROYECTO.md`](docs/DOCUMENTACION_PROYECTO.md): el proyecto
  explicado archivo por archivo, para estudiarlo a fondo.
- [`docs/PROPUESTA.md`](docs/PROPUESTA.md): la propuesta académica, con el diseño
  experimental, las hipótesis evaluadas y la justificación de cada tecnología.
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): registro detallado de resultados y hallazgos,
  con todas las cifras.

## Referencias

- Ha, D., & Schmidhuber, J. (2018). *World Models*.
- Hafner, D., et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*.
- Alegre, L. N. *SUMO-RL* — https://github.com/LucasAlegre/sumo-rl
