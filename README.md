# Traffic World Model

Proyecto de investigación y desarrollo para controlar semáforos inteligentes mediante World Models en una intersección simulada de SUMO.

## Objetivo

Diseñar un pipeline para aprender la dinámica del tráfico de una intersección con una representación compacta del estado y luego usar esa aproximación para apoyar la toma de decisiones del controlador semafórico.

La idea central es:

```text
SUMO -> Estado del tráfico -> Autoencoder/VAE -> representación latente
-> Modelo temporal (LSTM/Transformer) -> predicción de z_{t+1}
-> Dream Environment -> Controlador -> Acción -> SUMO
```

La implementación actual ya estableció la capa de infraestructura y la integración con SUMO, y en este punto el proyecto está avanzando en la definición del estado del tráfico usando TraCI.

## Estado actual del proyecto

### ✅ Completado

- Entorno reproducible en Python 3.11 con dependencias definidas en `requirements.txt`.
- Integración con SUMO + TraCI + `sumo-rl`.
- Red propia de una intersección construida y versionada bajo `environments/single-intersection/`.
- Wrapper de entorno centralizado en `environments/traffic_environment.py`.
- Arquitectura modular separada en:
  - `configs/` para configuración del entorno
  - `environments/` para builder/reward y lógica del entorno
  - `scripts/` para pruebas y ejecuciones por etapa
  - `training/`, `evaluation/` y `models/` para la fase de aprendizaje
- Estado personalizado construido desde TraCI mediante `environments/custom_state_builder.py`.
- Smoke test funcional del entorno.

### 🔄 En desarrollo

- Definición final del vector de estado personalizado.
- Función de recompensa del proyecto alineada con la lógica de tráfico.
- Recolección de dataset de transiciones `(s_t, a_t, r_t, s_{t+1})`.
- Entrenamiento del autoencoder / VAE.
- Modelo temporal para dinámica latente.
- Controlador semafórico y comparación con baselines.

## Estructura del repositorio

```text
traffic-world-model/
├── configs/                      # configuración del entorno, entrenamiento y modelos
│   ├── environment.py
│   ├── environment_config.py
│   ├── reward.py
│   ├── training.py
│   ├── world_model.py
│   ├── raw/
│   └── processed/
├── data/                         # datos de tráfico y datasets
│   ├── raw/
│   └── processed/
├── docs/                         # documentación y material auxiliar
├── environments/                 # entorno SUMO y definiciones del estado/recompensa
│   ├── __init__.py
│   ├── custom_state_builder.py
│   ├── default_reward_function.py
│   ├── default_state_builder.py
│   ├── reward_function.py
│   ├── single-intersection/
│   ├── state_builder.py
│   ├── traffic_environment.py
│   └── __pycache__/
├── evaluation/                   # scripts y resultados de evaluación
├── experiments/                  # experimentos y registros
├── external_sumo_rl/             # referencia del proyecto original sumo-rl
├── models/                       # módulos de encoder, dynamics y controller
├── notebooks/                    # exploración y visualización
├── scripts/                      # entrypoints del proyecto
│   ├── collect_dataset.py
│   ├── evaluate_world_model.py
│   ├── test_environment.py
│   ├── test_sumo_rl.py
│   ├── train_controller.py
│   ├── train_representation.py
│   ├── train_world_model.py
│   └── visualize_dataset.py
├── tests/                        # pruebas del proyecto
├── training/                     # pipeline de entrenamiento
├── utils/                        # utilidades varias
├── external_sumo_rl/             # copia de referencia de sumo-rl
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## Red de simulación

La red base del proyecto está en:

- `environments/single-intersection/single-intersection.net.xml`
- `environments/single-intersection/single-intersection.rou.xml`
- `environments/single-intersection/single-intersection.sumocfg`

Se trata de una intersección de 4 brazos con un único semáforo controlando los cruces principales. La topología está diseñada para ser una base reproducible para la investigación del problema de tráfico inteligente.

## Arquitectura actual del entorno

El punto de entrada principal del proyecto es `TrafficEnvironment`, ubicado en `environments/traffic_environment.py`.

Este wrapper:

- encapsula la creación del simulador `sumo_rl.SumoEnvironment`
- centraliza el ciclo de `reset()` y `step()`
- permite inyectar un `state_builder` y un `reward_function`
- mantiene la lógica del proyecto separada de las API internas de SUMO

La implementación actual del estado usa TraCI y no depende directamente de la observación nativa de `sumo-rl`.

## Cómo reproducir el entorno

### 1) Crear entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Verificar que SUMO esté disponible

```powershell
sumo --version
```

### 3) Ejecutar la prueba de humo del entorno

```powershell
python scripts\test_environment.py
```

También puede validarse la compatibilidad con el ejemplo de `sumo-rl` mediante:

```powershell
python scripts\test_sumo_rl.py
```

## Convenciones de desarrollo

- La lógica del proyecto no debe depender directamente de la implementación interna del simulador cuando exista una abstracción estable.
- La interfaz del entorno se mantiene en `TrafficEnvironment` como punto de integración.
- El estado y la recompensa se construyen mediante componentes dedicados, con posibilidad de reemplazo por versiones más complejas sin romper el resto del sistema.
- El trabajo actual sigue priorizando la infraestructura y el estado del tráfico antes de entrar en la fase de VAE, dynamics y controlador.

## Referencias

- Ha, D., & Schmidhuber, J. (2018). World Models.
- Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination.
- Alegre, L. N. (2019). SUMO-RL.
- LucasAlegre/sumo-rl (repositorio de referencia).

## Nota final

El README se actualizó para reflejar el estado real del proyecto en este momento: infraestructura funcional, entorno centralizado, estado personalizado con TraCI y preparación de la etapa de aprendizaje del World Model.