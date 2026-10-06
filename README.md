# Traffic World Model — v2: corredor de 4 intersecciones

Control de semáforos con un *World Model* (Ha & Schmidhuber, 2018) en un corredor simulado en SUMO
con **cuatro intersecciones coordinadas** (A0, B0, C0, D0). Se compara contra el RL directo sobre
el simulador y contra reglas sin aprendizaje. Trabajo de grado de Ingeniería de Sistemas,
Universidad del Quindío.

Esta rama (`v2/four-intersections`) es la **versión 2**. La versión 1, con una sola intersección,
está congelada (ver [Versión 1](#versión-1)).

## Qué es la v2

```text
SUMO (corredor A0-B0-C0-D0) → estado de 104 dims (26 por semáforo) → normalización (scaler.pkl)
  → modelo del mundo: LSTM o Transformer (~152k parámetros, 10 semillas cada uno)
  → (a) PPO entrenado en el sueño     (b) planificador que imagina desde el estado real
  → evaluación en SUMO real, con pre-registro y conjuntos de test reservados
```

- **Acción:** 4 bits, mantener o cambiar la fase en cada semáforo.
- **Recompensa:** suma de la recompensa estilo v1 de los 4 semáforos (espera más cola).
- **Sin Autoencoder:** el modelo del mundo trabaja sobre el estado normalizado.
- **Planificador (v2.1-A):** en cada paso real, con la ventana real de los últimos 16 pasos,
  evalúa en lote las 16 acciones conjuntas. Imagina H = 3 pasos, continuando con el PPO del
  sueño, y aplica la de mayor retorno imaginado. No reentrena nada.
- **Referencias sin aprendizaje:** `fijo_2_3`, `min_verde_y_cambiar`, `cola_mas_larga`,
  `max_presion` y `espera_mas_larga`.

## Resultados

Cada controlador aprendido tiene 10 semillas o réplicas. Retorno medio por episodio (más cerca de
0 es mejor). Catastróficos: episodios con retorno < −3,700. Las cifras son de
[`docs/v2/ADDENDUM_PLANIFICACION.md`](docs/v2/ADDENDUM_PLANIFICACION.md), sección 18 (test
25000–25047, 48 escenarios), y de [`docs/v2/ADDENDUM_OOD.md`](docs/v2/ADDENDUM_OOD.md), sección 7
(OOD 23000–23029, 30 escenarios). Cada conjunto se evaluó una sola vez.

| Política | Test: media | Test: catastróficos | OOD: media | OOD: catastróficos |
|---|---|---|---|---|
| **Planificador con Transformer** (`plan_ppo_transformer`, H = 3) | **−1,165.4** | **0/480** | **−1,149.9** | **0/300** |
| Planificador con LSTM (`plan_ppo_lstm`, H = 3) | −1,625.0 | 14/480 | −1,658.9 | 7/300 |
| PPO del sueño, Transformer | −3,473.8 | 36/480 | −3,104.6 | 26/300 |
| PPO del sueño, LSTM | −17,256.5 | 214/480 | −16,335.0 | 132/300 |
| RL directo, 30k pasos | −1,752.3 | 1/480 | −1,703.1 | 1/300 |
| RL directo, 10k pasos | −5,736.8 | 179/480 | −6,309.2 | 106/300 |
| Mejor regla en el total (`espera_mas_larga`) | −1,191.8 | 0/48 | −1,169.6 | 0/30 |

**Lo que muestran las comparaciones pre-registradas** (12 por conjunto, Bonferroni α' = 0.05/12;
test principal: Welch sobre las 10 medias por semilla):

- **El planificador con Transformer supera al RL directo de 30k, de forma significativa en los dos
  conjuntos:** +587.0 en test (IC al 99.58% [+423.3, +750.6]) y +553.2 en OOD (IC
  [+365.7, +740.8]).
  - Usa 21,240 interacciones reales por réplica sin compartir el dataset, o 6,552
    compartiéndolo, frente a 39,208 del RL directo: entre **1.8x y 6.0x menos**.
- **Empata con la mejor regla en el total:** +26.4 en test (p = 0.10) y +19.7 en OOD (p = 0.30),
  brechas no significativas.
- **Pierde, de forma significativa, frente a la mejor regla de cada intersección en B0 y C0**, en
  los dos conjuntos. En C0, `min_verde_y_cambiar` es mejor que todos los métodos aprendidos.
- **Frente al PPO del sueño sin planificación**, el planificador elimina casi todos los episodios
  catastróficos y el bloqueo de fases. La diferencia de retorno no es significativa en el test
  principal porque una sola semilla del sueño domina la varianza (LSTM s5 y Transformer s1).
- **Ningún método aprendido de la v2 supera a la mejor regla sin aprendizaje en el total.**

El resto se registra en los addendums: el control con PPO del sueño (Fase 3), el hallazgo de un
desfase en la ventana de acciones del Dream Environment (corregirlo no cambió el control), la
validación y todas las cifras por intersección. Ver la sección de documentación.

## Estructura del repositorio

```text
traffic-world-model/
├── configs/                 # Configuración: entorno, recompensas (v1 y corredor), modelos, PPO
├── datasets/                # Datasets de PyTorch; los datos generados no se versionan
├── docs/
│   ├── v2/                  # Diseños, pre-registros y resultados de la v2 (ver abajo)
│   ├── v1/README_v1.md      # El README de la v1, conservado
│   ├── results/v2/          # Resultados por episodio de la v2 (JSON y CSV), versionados
│   └── PROPUESTA.md, DOCUMENTACION_PROYECTO.md, ...   # documentos de la v1
├── environments/
│   ├── four-intersection-corridor/   # Red y demanda SUMO del corredor
│   ├── corridor_environment.py       # Entorno real: estado de 104, acción de 4 bits, recompensa sumada
│   ├── scaled_corridor_environment.py# Puente: normaliza con el mismo scaler.pkl del modelo
│   ├── corridor_dream_environment.py # Dream Environment de la v2 (window_alignment legacy/aligned)
│   ├── corridor_planner.py           # Planificador con el modelo del mundo (v2.1-A)
│   └── ...                           # Entornos de la v1
├── evaluation/              # Evaluación del modelo temporal (rollout_episode, Experimento 1)
├── models/
│   ├── world_model/         # LSTM, Transformer, TSMixer
│   └── checkpoints/v2/      # Pesos (no versionados); sí sus .json y run_info.json
├── scripts/v2/              # Puntos de entrada de la v2: dataset, entrenamiento, evaluación, análisis
├── training/                # train_controller_v2.py (PPO del sueño y RL directo) y entrenamientos de modelos
├── tests/                   # pytest
├── CLAUDE.md, PROJECT_STATUS.md
└── requirements.txt
```

**Documentos de la v2, en orden:**

1. [`docs/v2/DISENO_RED_4_INTERSECCIONES.md`](docs/v2/DISENO_RED_4_INTERSECCIONES.md): la red y la
   calibración de la demanda.
2. [`docs/v2/DISENO_ESTADO_ACCION_RECOMPENSA.md`](docs/v2/DISENO_ESTADO_ACCION_RECOMPENSA.md): el
   estado, la acción y la recompensa.
3. [`docs/v2/ADDENDUM_DATASET.md`](docs/v2/ADDENDUM_DATASET.md): el dataset y los splits de
   semillas.
4. [`docs/v2/ADDENDUM_AUTOENCODER.md`](docs/v2/ADDENDUM_AUTOENCODER.md) y
   [`docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md`](docs/v2/ADDENDUM_LSTM_VS_TRANSFORMER.md): el modelo
   del mundo.
5. [`docs/v2/DISENO_CONTROL.md`](docs/v2/DISENO_CONTROL.md) y
   [`docs/v2/ADDENDUM_CONTROL.md`](docs/v2/ADDENDUM_CONTROL.md): Fase 3, el PPO del sueño frente al
   RL directo y las reglas.
6. [`docs/v2/ADDENDUM_SUENO_CORREGIDO.md`](docs/v2/ADDENDUM_SUENO_CORREGIDO.md): el desfase de la
   ventana y su corrección.
7. [`docs/v2/ADDENDUM_PLANIFICACION.md`](docs/v2/ADDENDUM_PLANIFICACION.md): el planificador
   (v2.1-A), su validación y el test nuevo.
8. [`docs/v2/ADDENDUM_OOD.md`](docs/v2/ADDENDUM_OOD.md): la confirmación final en OOD.

[`PROJECT_STATUS.md`](PROJECT_STATUS.md) resume el estado y las cifras;
[`CLAUDE.md`](CLAUDE.md) recoge las decisiones de diseño y las convenciones de trabajo.

## Instalación

Igual que la v1: Python 3.11 (los resultados se obtuvieron con 3.11.9) y
[SUMO](https://eclipse.dev/sumo/) **1.27.1**, instalado aparte, con `SUMO_HOME` apuntando a su
carpeta. `requirements.txt` fija las versiones de las librerías de Python.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
sumo --version
```

## Reproducir

Los datasets y los pesos no se versionan; se regeneran con los scripts. Cada paso tiene su
pre-registro en `docs/v2/`.

1. **Dataset:** `scripts/v2/collect_dataset_v2.py` y `scripts/v2/prepare_dataset_v2.py`.
2. **Modelos del mundo:** `scripts/v2/run_lstm_vs_transformer.py`.
3. **Controladores:**
   - PPO del sueño: `scripts/v2/run_control_stage2.py`.
   - RL directo: `scripts/v2/run_control_stage3.py`.
   - Sueño corregido: `scripts/v2/run_control_aligned.py`.
4. **Evaluación:**
   - `scripts/v2/evaluate_control_v2.py`, con políticas `dream:`, `direct:`, `ref:` y `plan:`.
   - Validación y test del planificador: `run_planning_validation.py` y `run_planning_test.py`.
   - OOD: `run_ood.py`.
5. **Análisis:** los `scripts/v2/analyze_*.py`, que no simulan nada.

Los lanzadores usan 4 procesos y son reanudables. Se niegan a arrancar sin corriente o con 2 GB o
menos de memoria, y a escribir sobre resultados oficiales.

## Tests

```powershell
pytest -v
```

Los tests se corren con el Python del entorno virtual del proyecto
(`.\.venv\Scripts\python.exe -m pytest -v`), no con el `python` del sistema, que no tiene las
dependencias.

159 tests en 25 archivos. Uno es un `xfail` estricto que documenta el desfase de ventana del
Dream Environment de la v1.

## Versión 1

La **versión 1** (una intersección con demanda asimétrica: Autoencoder, LSTM y PPO del sueño
frente a RL directo) está **congelada** en la rama `main` y en la etiqueta
[`v1-final`](https://github.com/HulbertAlejandro/traffic-world-model/tree/v1-final). No recibe
cambios, y la v2 no se fusiona en ella.

- Su README está en [`docs/v1/README_v1.md`](docs/v1/README_v1.md).
- Su código, datos y documentos siguen en esta rama, en `docs/`, `scripts/`, `environments/`,
  etc.
- Su resultado principal: el World Model igualó o superó al RL directo con muchas menos
  interacciones reales, en el escenario de una intersección.
- Dos hallazgos de la v2 sobre la v1:
  - el Dream Environment de la v1 tiene el mismo desfase de ventana, y su efecto allí no se
    midió;
  - en la v1 el controlador del sueño solo se comparó con tiempo fijo y con una regla de fase
    contraria.

  Ver [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Referencias

- Ha, D., & Schmidhuber, J. (2018). *World Models*.
- Hafner, D., et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*.
- Alegre, L. N. *SUMO-RL* — https://github.com/LucasAlegre/sumo-rl
