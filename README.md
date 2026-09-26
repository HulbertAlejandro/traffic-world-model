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
| 0 | ¿Ayuda comprimir el estado con un Autoencoder? | Sí, de forma moderada y mayoritaria: 5 semillas por rama, 37/50 pares semilla × horizonte, mejora mediana +10.9%; se mantiene |
| 1 | ¿El LSTM predice mejor que un baseline persistente? | Sí, en los 10 horizontes |
| 2 | ¿Un PPO entrenado en el sueño controla bien en SUMO real? | Sí; ver el resultado principal |
| 3 | ¿Transformer o TSMixer mejoran al LSTM? | No: el LSTM gana en los 10 horizontes; se mantiene |

**Resultado principal:** el World Model iguala o supera al RL directo usando muchas menos
interacciones reales con SUMO. Media de todas las semillas de entrenamiento (sueño 3, RL
directo 4), en SUMO real:

| | Escenarios oficiales | Escenarios nuevos (7000–7029) | Interacciones reales por semilla |
|---|---|---|---|
| PPO del sueño (World Model) | **-326.79** | **-293.81** | ~4,600 (7,800 sin amortizar el dataset) |
| RL directo, 10k pasos | -454.51 | -439.73 | 13,240 |
| RL directo, 30k pasos | -402.13 | -316.59 | 39,208 |
| Tiempo fijo | -411.27 | -391.70 | — |

- **Frente al RL directo de 10k**, el World Model es mejor en los dos conjuntos de
  escenarios.
- **Frente al de 30k no se detectó una diferencia consistente:** hay brecha a favor del
  World Model en los escenarios oficiales y ninguna detectable en los nuevos.
- El World Model lo logra con entre 2.9x y 8.5x menos interacciones reales (1.7x a 5.0x
  sin amortizar el dataset) y es el método con menos episodios catastróficos.

Detalle y pruebas estadísticas en [`PROJECT_STATUS.md`](PROJECT_STATUS.md); resultados
por episodio en [`docs/results/`](docs/results/).

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

Orden de ejecución. Los scripts de entrenamiento que escriben en carpetas oficiales
(`train_world_model.py`, `train_controller_direct.py`) **se niegan a sobrescribirlas** sin
`--overwrite-official`: en un clon nuevo, donde esas carpetas solo tienen los `.json`,
hay que pasar esa bandera para regenerarlas. `train_autoencoder.py`, `train_controller.py`
y los entrenamientos del Experimento 3 no tienen ese guardia y escriben directamente en
`models/checkpoints/`: respalda esa carpeta antes de ejecutarlos.

```powershell
# 1. Datos: 80 episodios con semilla de SUMO distinta, split por episodio, normalización
python scripts\collect_dataset.py
python scripts\split_dataset.py
python scripts\normalize_dataset.py

# 2. Representación y dinámica (Experimento 1)
python training\train_autoencoder.py
python scripts\encode_latent_dataset.py
python training\train_world_model.py --overwrite-official
python scripts\evaluate_world_model.py

# 3. Experimento 0: modelo temporal sobre el estado crudo frente a z, 5 semillas por rama,
#    entrenadas hasta que corte el early stopping (tope de 300 épocas)
python scripts\prepare_raw_sequence_dataset.py
foreach ($s in 0..4) {
  python training\train_world_model.py --seed $s --epochs 300 --output-dir models\checkpoints\exp0_multiseed_300ep\z\seed$s
  python training\train_world_model_raw.py --seed $s --epochs 300 --output-dir models\checkpoints\exp0_multiseed_300ep\raw\seed$s
}
python scripts\compare_experiment_0_multiseed.py `
  --z-dirs (0..4 | % { "models\checkpoints\exp0_multiseed_300ep\z\seed$_" }) `
  --raw-dirs (0..4 | % { "models\checkpoints\exp0_multiseed_300ep\raw\seed$_" }) `
  --output docs\results\experiment_0_multiseed_300ep.json

# 4. Experimento 3: Transformer y TSMixer (requieren el reward_scaler.json del paso 2)
python training\train_world_model_transformer.py
python training\train_world_model_tsmixer.py
python scripts\evaluate_world_model_transformer.py
python scripts\evaluate_world_model_tsmixer.py
python scripts\compare_experiment_3.py

# 5. Controladores. PPO en el sueño (semilla oficial 2 por defecto; las semillas 0 y 1
#    requieren cambiar ControllerConfig.seed). RL directo: 4 semillas × 10k y 30k pasos,
#    cada una en su carpeta.
python training\train_controller.py
foreach ($s in 0..3) {
  python training\train_controller_direct.py --seed $s --total-timesteps 10000 --output-dir models\checkpoints\direct_10k\seed$s
  python training\train_controller_direct.py --seed $s --total-timesteps 30000 --output-dir models\checkpoints\direct_30k\seed$s
}

# 6. Evaluación en SUMO real: todas las semillas de cada método, escenarios oficiales
#    (3000 y 5000) y nuevos (7000-7029), con pruebas estadísticas y resultados por episodio
python scripts\evaluate_multiseed_statistical.py --help
python scripts\evaluate_final_comparison.py   # solo los checkpoints oficiales
```

Los pasos 3 y 4 son experimentos de validación y no los necesitan los pasos 5 y 6.

## Tests

```powershell
pytest -v
```

81 tests en 16 archivos.

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
