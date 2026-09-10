# Control inteligente de semáforos mediante World Models

Proyecto de grado — Ingeniería de Sistemas, Universidad del Quindío (2026)

Diseño e implementación de un sistema basado en **World Models** (Ha & Schmidhuber, 2018)
para aprender la dinámica del tráfico de una intersección simulada en **SUMO**, y usar ese
modelo aprendido para apoyar el entrenamiento y/o selección de acciones de un controlador
semafórico — comparando este enfoque contra un agente de Reinforcement Learning entrenado
directamente contra el simulador.

## Idea central

```
SUMO → Estado del tráfico → Autoencoder/VAE → z → Modelo temporal (LSTM/Transformer)
     → Predicción de z_{t+1} y recompensa → Dream Environment → Controller → Acción → SUMO
```

En vez de que el agente aprenda interactuando miles de veces con el simulador real,
primero se le enseña una aproximación comprimida y temporal de cómo se comporta el
tráfico ("V" + "M"), y luego el controlador ("C") puede entrenarse o decidir usando
esa aproximación, reduciendo potencialmente el número de interacciones necesarias
con SUMO.

## Estado actual del proyecto

- [x] Entorno de desarrollo reproducible (Python 3.11, entorno virtual, `requirements.txt`)
- [x] SUMO 1.27.1 instalado y verificado (consola + GUI + netedit)
- [x] `traci`, `sumolib`, `sumo-rl`, `gymnasium`, `stable-baselines3` integrados
- [x] Pipeline TraCI ↔ Python validado con la red de ejemplo de `sumo-rl`
- [x] Red propia de una intersección construida con `netgenerate` (4 brazos, 1 semáforo)
- [x] Demanda de tráfico propia definida (350 veh/h por brazo, sin giros en U)
- [x] Validación visual en `sumo-gui`: semáforo cambiando de fase, vehículos fluyendo
- [x] Pipeline TraCI ↔ Python re-validado apuntando a la red propia
- [ ] Definición final del vector de estado y función de recompensa personalizada
- [ ] Recolección de dataset de trayectorias `(s_t, a_t, r_t, s_{t+1})`
- [ ] Autoencoder / VAE sobre el vector de estado
- [ ] Modelo temporal (LSTM) para predicción de `z_{t+1}`
- [ ] Dream Environment / imaginación
- [ ] Controlador (RL) y comparación contra baselines

## Estructura del repositorio

```
traffic-world-model/
├── data/
│   ├── raw/                 # trayectorias crudas recolectadas de SUMO
│   └── processed/           # datasets ya normalizados/estructurados
├── docs/                    # propuesta de proyecto y documentación
├── environments/
│   └── single-intersection/ # red propia: .net.xml, .rou.xml, .sumocfg
├── experiments/             # resultados y configuraciones de experimentos
├── models/
│   ├── encoder/             # Autoencoder / VAE
│   ├── dynamics/            # modelo temporal (LSTM / Transformer)
│   └── controller/          # controlador (RL)
├── notebooks/                # exploración y visualización
├── scripts/                 # scripts ejecutables por etapa del pipeline
├── external_sumo_rl/         # clon de referencia de LucasAlegre/sumo-rl (no versionado)
├── requirements.txt
└── README.md
```

## La red del proyecto: `single-intersection`

Construida con `netgenerate` (100% reproducible, ver comando exacto abajo), representa
una intersección en cruz con 4 brazos de 200 m cada uno, un carril por sentido, y un
semáforo de 2 fases (Norte-Sur / Este-Oeste) en el nodo central.

Comando de generación de la geometría:

```powershell
netgenerate --grid --grid.number=1 --grid.length=200 --grid.attach-length=200 `
  --default.lanenumber=1 --default.speed=13.89 --tls.set=A0 `
  --output-file environments/single-intersection/single-intersection.net.xml
```

Demanda de tráfico (`single-intersection.rou.xml`): 350 veh/h por brazo, repartidos en
recto (200), izquierda (90) y derecha (60) — un régimen moderado, no saturado, elegido
deliberadamente para que las comparaciones entre baselines y World Model no queden
contaminadas por una red en colapso desde el inicio.

## Cómo reproducir lo que hay hasta ahora

```powershell
# 1. Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Verificar que SUMO está instalado y accesible
sumo --version

# 3. Validar el pipeline completo (SUMO + TraCI + sumo-rl) contra nuestra propia red
python scripts\test_sumo_rl.py
```

## Referencias principales

- Ha, D., & Schmidhuber, J. (2018). *World Models*.
- Hafner, D., et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*.
- Alegre, L. N. (2019). *SUMO-RL* — https://github.com/LucasAlegre/sumo-rl
- Tallec, C., et al. *Reimplementation of World Models in PyTorch* — https://github.com/ctallec/world-models
- Dai et al. (2022). *Image-based traffic signal control via world models*.

## Notas de reproducibilidad

- Todos los archivos de red (`.net.xml`, `.rou.xml`, `.sumocfg`) están versionados en
  `environments/`, junto con el comando exacto de `netgenerate` usado para generarlos.
- `external_sumo_rl/` es solo una copia de referencia del repositorio de `sumo-rl`
  (clonada para consultar redes de ejemplo); no se versiona ni se modifica directamente.
- Las semillas, configuración de simulación y versiones de modelos se documentarán en
  `experiments/` a partir de la etapa de recolección de datos.