# Pre-registro: control en el corredor de 4 intersecciones (v2, Fase 3)

**Escrito el 4 de octubre de 2026, antes de entrenar ningún controlador y antes de evaluar nada en
SUMO** (ni los controladores ni las referencias). El diseño técnico está en `DISENO_CONTROL.md`.
Lo que este documento fija no se cambia después de ver resultados; cualquier cambio se agrega como
enmienda fechada, diciendo si se escribió antes o después de ver qué datos.

## 1. Brazos

| Brazo | Qué es | Semillas |
|---|---|---|
| `sueno_lstm` | PPO entrenado en el Dream Environment de la LSTM (oculta 137) | 0–9 |
| `sueno_transformer` | PPO entrenado en el Dream Environment del Transformer (d_model 80) | 0–9 |
| `directo_10k` | PPO entrenado directamente en SUMO real, 10,000 pasos | 0–9 (etapa 3) |
| `directo_30k` | PPO entrenado directamente en SUMO real, 30,000 pasos | 0–9 (etapa 3) |

- **El controlador del sueño de semilla *i* se entrena en el sueño del modelo de semilla *i* de su
  arquitectura** (`models/checkpoints/v2/arch_comparison/<arq>_raw_s<i>/`), con semilla de PPO *i*.
  Así, la variación entre semillas incluye la del modelo del mundo, que es parte del método.
- **Referencias sin aprendizaje** (deterministas, una corrida por escenario): `fijo_2_3`,
  `min_verde_y_cambiar`, `cola_mas_larga`, `max_presion`, `espera_mas_larga`, las políticas del
  validador de la Fase 0 (`scripts/v2/validate_corridor_demand.py::make_policy`), sin
  reimplementar.
- 10 semillas por brazo desde el principio. **Ninguna semilla se descarta.** Si un entrenamiento
  falla por algo ajeno al método (memoria, Smart App Control), se repite con la misma semilla y se
  documenta.
- Etapas: la **etapa 2** entrena los 20 controladores del sueño; la **etapa 3** entrena los 20 del
  RL directo y corre las evaluaciones finales. Esta etapa 1 no entrena ninguno de los 40.

## 2. Uso de las semillas de SUMO

| Rango | Uso | Cuándo |
|---|---|---|
| 20000–20111 (`train`) | Solo el dataset del modelo del mundo y las ventanas semilla del sueño. **Ninguna evaluación las usa.** | — |
| 21000–21023 (`validation`) | Selección de checkpoints, referencias, umbral de catastróficos, diagnóstico de fidelidad y cualquier decisión de diseño | etapas 1–3 |
| 22000–22023 (`test`) | **Comparación principal**, una sola vez por controlador | al final de las etapas 2 y 3 |
| 23000–23029 (`ood`) | Comprobación final de generalización, **una sola vez**, con todo ya decidido | al final de la etapa 3 |
| **30000–30999** | Semillas de entrenamiento del RL directo (`ReseedingWrapper.training_seeds(30000)`: 30000, 30001, …; 30k pasos usan unas 660) | etapa 3 |

- El rango 30000–30999 es disjunto de todos los anteriores, del piloto del dataset (19000–19017) y
  de la calibración de la Fase 0 (11000–11019). Todas las semillas del RL directo recorren la
  misma secuencia de escenarios, como en la v1.
- **Selección de checkpoints** (sueño y directo): `EvalCallback` en SUMO real sobre las semillas de
  validación **21000–21004**, cíclicas (5 episodios por evaluación), igual que la v1 con
  20000–20004. Se elige el checkpoint con mejor retorno real; **nunca por la recompensa
  imaginada** (correlación de 0.08 en la v1).
- El script de evaluación se niega a usar semillas 20000–20111 y pide confirmación explícita para
  `test` y `ood`.

## 3. Hiperparámetros de PPO

Los de `ControllerConfig()` de la v1, **el mismo objeto de partida en todos los brazos**:
`learning_rate` 3e-4, `n_steps` 256, `batch_size` 64, `n_epochs` 10, `gamma` 0.99,
`normalize_reward` True, `reward_clip` 10, `MlpPolicy` por defecto de SB3. Por brazo cambian solo
estos campos, con su justificación:

- `seed`: la del controlador.
- `total_timesteps`: 50,000 pasos imaginados en el sueño (default de la v1); 10,000 o 30,000
  reales en el directo (los presupuestos de la v1).
- `normalize_obs`: **False en el sueño** (la observación ya es el estado normalizado con
  `scaler.pkl`, la escala del modelo), **True en el directo**, como en la v1. El RL directo no usa
  `scaler.pkl` porque sale del dataset: si lo usara, dependería de las 9,600 transiciones del
  dataset y habría que sumárselas.
- Frecuencia de evaluación periódica: cada 5,000 pasos en el sueño (10 evaluaciones) y cada
  `max(n_steps, 1000)` en el directo, como en la v1.

**`dream_max_steps = 7`, como en la v1.** No se ajusta con `test` ni con nada.

Recorte de la recompensa imaginada: la regla de la v1 (percentiles 1 y 99 del split de
entrenamiento), recalculada para la v2: **[−345.43, 0.0]**.

## 4. Métricas

Sobre cada episodio real (300 s, 60 pasos de control):

1. **Retorno total** del episodio (suma de la recompensa estilo v1 de los 4 semáforos). Métrica
   principal.
2. **Retorno por intersección** (A0, B0, C0, D0), de `info["reward_per_signal"]`.
3. **Episodios catastróficos**: retorno total por debajo de un umbral fijado así, **antes de ver
   el valor**: **1.5 × el peor retorno de `fijo_2_3` en las 24 semillas de validación,
   redondeado hacia abajo a la centena** (más negativo). Por construcción, `fijo_2_3` no tiene
   catastróficos en validación. El valor se calcula en el Paso 5 de la etapa 1 y queda escrito en
   la sección 9 sin cambiar la regla.
4. **Interacciones reales**, con las dos contabilidades de la v1:
   - **Dataset del modelo del mundo:** 160 episodios × 60 = **9,600 transiciones** (train,
     validation y test; los tres se usaron para construir o elegir el modelo). El split `ood` no
     se cuenta: no se usó para el modelo.
   - **Selección del sueño:** 10 evaluaciones × 5 episodios × 60 = **3,000 pasos reales por
     semilla** (contados por `run_info.json`, no deducidos).
   - **Sin compartir el dataset** (comparación conservadora): 9,600 + 3,000 = **12,600 por
     semilla**.
   - **Compartiéndolo entre las 10 semillas del brazo:** 960 + 3,000 = **3,960 por semilla**.
     (Compartido entre los 20 controladores del sueño serían 3,480; esa cifra no se usa como
     principal.)
   - **RL directo:** pasos de entrenamiento (SB3 completa el último rollout) + evaluaciones
     periódicas; con la v1 eran 13,240 (10k) y 39,208 (30k), y aquí se toman de `run_info.json`.
   - **Nota escrita antes de ver resultados:** el dataset de la v2 es el doble que el de la v1, así
     que, sin compartirlo, el sueño (12,600) cuesta casi lo mismo que el directo de 10k (≈13,240):
     la razón conservadora frente a 10k será ≈ 1.05x, no 1.7x. Se reporta como rango y con las dos
     cifras, nunca como un solo número.

Descriptivas adicionales: espera y cola medias, llegadas (`arrivals_total`), cambios de fase reales
por semáforo.

## 5. Estadística

Unidad: **las medias por semilla de entrenamiento** (10 por brazo aprendido) y **los 24 escenarios
de test** pareados. Los tests por episodio son pseudorreplicación y no se usan para decidir.

Para cada comparación se calculan:

- **Welch sobre las medias por semilla** (10 frente a 10). Contra una referencia determinista, la
  versión equivalente: t de una muestra sobre las 10 diferencias (media de la semilla − media de
  la referencia en los mismos 24 escenarios).
- **t pareada por escenario** (24 escenarios; por escenario, la media de las 10 semillas de cada
  brazo).
- **Wilcoxon pareado por escenario**, mismos pares.

**Test principal de cada comparación: el de nivel de semilla** (Welch o su versión de una
muestra), porque es el único que dice algo sobre el método. Los otros dos se reportan siempre; si
discrepan del principal se dice.

### 5.1 Comparaciones planificadas (sobre `test`)

| # | Comparación | Métrica |
|---|---|---|
| P1 | `sueno_lstm` frente a `sueno_transformer` | retorno total |
| P2 | sueño adoptado frente a `directo_10k` | retorno total |
| P3 | sueño adoptado frente a `directo_30k` | retorno total |
| P4 | sueño adoptado frente a la mejor referencia (total) | retorno total |
| P5 | sueño adoptado frente a la mejor referencia en B0 | retorno de B0 |
| P6 | sueño adoptado frente a la mejor referencia en C0 | retorno de C0 |

- **Bonferroni sobre las 6:** α' = 0.05 / 6 = **0.00833**; los intervalos de confianza se dan al
  **1 − α' = 99.17%**.
- **"Mejor referencia"** se elige en **validación** (Paso 5), no en test: la de mayor retorno
  medio total (P4), la de mayor retorno medio en B0 (P5) y la de mayor retorno medio en C0 (P6).
  Pueden ser referencias distintas.
- **Sueño adoptado:** el que decida P1 (siguiente punto).
- Lo demás (otras referencias, otras intersecciones, catastróficos, OOD) es **descriptivo**, sin
  corrección y sin decidir nada.
- Una diferencia numérica cuyo test principal no alcanza α' se reporta como **"brecha no
  significativa"**, nunca como diferencia de control.

### 5.2 Decisión LSTM frente a Transformer (P1)

- Si el IC al 99.17% de Welch sobre las medias por semilla de la diferencia (LSTM − Transformer)
  **excluye 0**, la arquitectura con mayor retorno medio es la oficial de la v2.
- **Desempate, fijado ahora:** si el IC incluye 0, se adopta **la LSTM** (inferencia más barata,
  continuidad con la v1), y queda escrito como limitación que el control no distinguió las dos
  arquitecturas.
- P2–P6 se calculan con el brazo adoptado. El otro brazo se reporta igual, de forma descriptiva.

### 5.3 OOD

Las semillas 23000–23029 se corren **una sola vez**, al final de la etapa 3, con todos los brazos y
las 5 referencias. Se reportan con los mismos estadísticos, de forma **descriptiva**: no cambian
ninguna decisión tomada con test.

## 6. Hipótesis

**Lo que el controlador puede ver.** El PPO usa `MlpPolicy`: decide solo con el estado actual de
104 dimensiones, **sin memoria de los pasos anteriores**. El modelo del mundo usa 16 pasos de
historia para imaginar, pero esa historia no llega a la política. Lo único que el controlador sabe
del pasado es lo que el propio estado ya resume: el tiempo en la fase actual
(`elapsed_phase_time`), la espera acumulada por carril (`waiting_times`), los vehículos detenidos
(`queue_lengths`), además de conteos, velocidades, ocupación y la fase.

**H1 (principal).** El controlador aprendido en el sueño supera a las reglas sin aprendizaje en
**B0 y C0**, las dos intersecciones interiores: reciben los pelotones de sus vecinas y, en las
fases 0 y 1, fueron donde las referencias no superaron a alternar rápido (C0 significativamente
peor; B0 sin significancia; `DISENO_RED_4_INTERSECCIONES.md`, sección 7.7). Se prueba con P5 y P6.

**Sobre la memoria (con cuidado).** H1 no supone memoria del controlador: si el aprendido supera a
las reglas, lo hace con la misma información que ellas tienen en el estado (más el tiempo en fase
y la espera acumulada, que `cola_mas_larga` no mira). Este experimento **no prueba** si la memoria
ayudaría. Si H1 falla, que las reglas "cambian tarde por construcción" o que haga falta memoria son
explicaciones posibles, pero **no se pueden afirmar** con estos datos: quedan como hipótesis para
otro experimento.

**H2 (secundaria).** El sueño adoptado iguala o supera al RL directo de 10k con, como mucho, una
cantidad similar de interacciones reales (P2), y la pregunta frente al de 30k (P3) queda abierta,
como en la v1.

## 7. Plan de cómputo

- **4 procesos en paralelo por defecto** (`CLAUDE.md`, 7.7 GB de RAM). Más solo si el autor
  confirma que no hay aplicaciones pesadas abiertas.
- **Antes de lanzar cada tanda de entrenamiento se comprueba que haya más de 2 GB de memoria
  disponible**; si no, no se lanza y se avisa.
- Los entrenamientos se pueden reanudar: una corrida cuyo `run_info.json` ya existe se salta.

## 8. Lo que hace la etapa 1 (este pre-registro, sin entrenar los 40)

1. Código y tests (`DISENO_CONTROL.md`, sección 4).
2. **Diagnóstico de fidelidad**, descriptivo: para los 20 modelos, sobre **validación**, Pearson y
   sesgo (imaginado − real) entre el retorno imaginado a 7 pasos (con el mismo
   `CorridorDreamEnvironment` que ve PPO, recompensa recortada, y también sin recortar) y el real,
   con las acciones registradas en cada ventana; IC al 95% por bootstrap de episodios (2,000
   remuestreos, semilla 0). Por arquitectura: la media de los 10 modelos. No decide nada.
3. **Referencias** en las 24 semillas de validación, con retorno por intersección. Fijan el umbral
   de catastróficos y las mejores referencias de P4–P6. **No se corren en test** hasta el final.
4. **Piloto de humo**: semilla 0 de cada arquitectura, 10,000 pasos imaginados, evaluación en 3
   semillas de validación (21005–21007). Solo para comprobar el flujo y medir tiempos; **no es un
   resultado** y no se usa para decidir nada. Sus salidas van a `models/checkpoints/v2/control_pilot/`.

## 9. Valores fijados por la regla después del Paso 5

Escrito el 4 de octubre de 2026 con `docs/results/v2/control/references_validation.json` (las 5
referencias en las 24 semillas de validación), aplicando las reglas de las secciones 4 y 5.1 sin
cambiarlas:

- **Umbral de catastróficos: −3,600.** Peor retorno de `fijo_2_3` en validación: −2,354.0
  (semilla 21000); 1.5 × −2,354.0 = −3,531, redondeado hacia abajo a la centena. Queda en
  `scripts/v2/evaluate_control_v2.py::CATASTROPHIC_THRESHOLD`.
- **Mejor referencia en el total (P4): `espera_mas_larga`** (−1,081.0).
- **Mejor referencia en B0 (P5): `espera_mas_larga`** (−95.0; `cola_mas_larga`, −98.9).
- **Mejor referencia en C0 (P6): `min_verde_y_cambiar`** (−292.2).

## 10. Registro de la etapa 1 (escrito después de los Pasos 4–6; no cambia nada de lo anterior)

### 10.1 Tests

`tests/test_v2_control.py`, 12 tests, todos en verde, y la suite completa en 135 passed. El test
del puente reproduce el episodio 21001 en SUMO con sus acciones guardadas: las 61 observaciones
coinciden con los estados normalizados del dataset (tolerancia 1e-5) y las 60 recompensas son
idénticas. Con el escalador alterado un 0.1%, ese test falla.

### 10.2 Fidelidad (validación; descriptivo)

Retorno imaginado a 7 pasos, con las acciones registradas, frente al real. Son 936 ventanas por
modelo (24 episodios × 39), con un retorno real medio de −350.9. IC al 95% por bootstrap de
episodios.

| Arquitectura | Pearson (media de 10) | Rango entre modelos | Sesgo imaginado − real (media) | Rango entre modelos |
|---|---|---|---|---|
| LSTM | **0.871** [0.827, 0.900] | 0.841 – 0.888 | **+6.2** [−13.1, +26.4] | −14.8 – +27.0 |
| Transformer | **0.897** [0.861, 0.926] | 0.870 – 0.928 | **+25.0** [+5.4, +44.6] | −19.4 – +55.0 |

Con la recompensa recortada, que es la que ve PPO. Sin recortar se obtiene casi lo mismo: LSTM
0.874 y +6.8; Transformer 0.898 y +23.2. El recorte cambia entre el 5% y el 29% de los retornos
de 7 pasos, según el modelo.

**Matices:**

- Mide la fidelidad **con las acciones del dataset**, no con las que elegirá PPO, así que no
  garantiza la fidelidad fuera de esa distribución (ver 10.4).
- No es comparable con el 0.08 de la v1, que correlacionaba la recompensa imaginada y la real
  **entre checkpoints** de un mismo entrenamiento.
- El Transformer es algo más preciso, pero optimista en promedio: imagina retornos unos 25 puntos
  mejores que los reales en 7 pasos.

### 10.3 Referencias en validación (24 semillas)

| Referencia | Total | A0 | B0 | C0 | D0 | Peor episodio | < −3,600 |
|---|---|---|---|---|---|---|---|
| `fijo_2_3` | −1,482.8 | −347.1 | −177.1 | −805.2 | −153.3 | −2,354 | 0 |
| `min_verde_y_cambiar` | −1,226.0 | −451.3 | −201.0 | **−292.2** | −281.6 | −1,727 | 0 |
| `cola_mas_larga` | −1,105.2 | −375.4 | −98.9 | −534.1 | −96.7 | −2,149 | 0 |
| `max_presion` | −7,162.2 | −1,408.2 | −2,375.8 | −1,168.9 | −2,209.3 | −13,010 | 22 |
| `espera_mas_larga` | **−1,081.0** | −388.7 | **−95.0** | −495.4 | −101.9 | −2,149 | 0 |

- **Reproducibilidad:** en los 16 episodios de validación donde el dataset usó la misma política
  (`fijo_2_3` o `cola_mas_larga`), el retorno coincide exactamente con el del manifiesto
  (diferencia máxima 0.0).
- **`max_presion` es muy mala en esta red, y ya lo era.** En la calibración de la Fase 1
  (`it5_offset_espera_300s.json`) su media fue −7,299. Es el mecanismo del giro a la izquierda
  permisivo de `DISENO_RED_4_INTERSECCIONES.md`: mantiene el verde a una cola que no avanza. No es
  un fallo del adaptador.
- C0 repite lo de las fases 0 y 1: la mejor regla allí es alternar rápido
  (`min_verde_y_cambiar`), no una referencia que mira el estado.

### 10.4 Piloto de humo (no es un resultado)

Semilla 0 de cada arquitectura, 10,240 pasos imaginados, 2 evaluaciones periódicas (600 pasos
reales) y la evaluación en 21005–21007. Comprueba que el flujo completo funciona: entrenar en el
sueño, seleccionar en SUMO real, guardar, cargar y evaluar por intersección. Lo que se observó **no
se usa para decidir nada**; se registra porque es inesperado:

- **El piloto de la LSTM bloquea el corredor.** Casi nunca cambia de fase en A0 y B0 (0 o 1
  cambios en 60 pasos), las transversales se quedan sin verde y la espera acumulada explota:
  retornos de −99,117 a −261,798. En el dataset de entrenamiento, A0 nunca mantuvo la fase más de
  19 pasos seguidos (C0, 15). El sueño no puede mostrar las consecuencias de mantenerla 60 pasos:
  cada episodio imaginado arranca de una ventana real (fases con poco tiempo acumulado) y dura 7
  pasos, y la recompensa imaginada por paso está recortada en −345.43.
- **El piloto del Transformer no se bloquea**: cambia de fase con regularidad en los 4 semáforos.
- Con 10,240 pasos, ninguno de los dos dice cómo será el entrenamiento completo de 50,000. **Nada
  de lo pre-registrado cambia** (horizonte 7, recorte, selección en SUMO real). Si el autor
  decide cambiar algo antes de la etapa 2, se hará como enmienda fechada, antes de entrenar.

**Tiempos medidos** (un proceso, sin otros entrenamientos):

| Medida | LSTM | Transformer |
|---|---|---|
| Por paso imaginado, con las actualizaciones de PPO | 3.1 ms | 3.0 ms |
| Por episodio real en la evaluación periódica (60 pasos) | 2.7 s | 2.3 s |
| Por episodio real en `evaluate_control_v2.py` | 2.4 s | 2.2 s |
| Referencias (sin PyTorch), por episodio | 2.0 s | |

### 10.5 Estimación de las etapas 2 y 3

Supuestos:

- Episodio real ≈ 2.5 s.
- Paso real de entrenamiento del RL directo ≈ 40 ms: unos 35 ms de SUMO más PPO, extrapolado de lo
  anterior y **no medido**.
- Con 4 procesos en paralelo, unas 3 veces más rápido que en serie.
- Los episodios con mucha congestión pueden tardar más.

**Etapa 2 (20 controladores del sueño):**

| Tarea | Cálculo | En serie | Con 4 procesos |
|---|---|---|---|
| Entrenar cada controlador | 50,176 × 3.1 ms ≈ 2.6 min de entrenamiento + 50 episodios de selección ≈ 2.2 min | ≈ 5 min por controlador; 100 min en total | **≈ 35 min** |
| Evaluar en test | 20 × 24 episodios ≈ 20 min, más las 5 referencias (120 episodios) ≈ 4 min | ≈ 25 min | |

**Total de la etapa 2: ≈ 1 h.**

**Etapa 3 (20 controladores del RL directo):**

| Tarea | Cálculo | En serie | Con 4 procesos |
|---|---|---|---|
| 10k | 10,240 × 40 ms ≈ 7 min + 50 episodios ≈ 2 min | ≈ 9 min por semilla | |
| 30k | 30,208 × 40 ms ≈ 20 min + 150 episodios ≈ 6 min | ≈ 26 min por semilla | |
| Las 20 corridas | | ≈ 6 h | **≈ 2 h** |
| Evaluar en test | 480 episodios | ≈ 20 min | |
| OOD | 40 controladores × 30 + 5 × 30 = 1,350 episodios | ≈ 55 min | |

**Total de la etapa 3: ≈ 3–3.5 h.**

**Memoria:** al terminar la etapa 1 había **1.76 GB disponibles** (de 7.7). Es menos de los 2 GB de
la sección 7: la etapa 2 no se lanza así.

## 11. Etapa 2: plan (escrito el 4 de octubre de 2026, antes de lanzar los entrenamientos)

Lo de esta sección se escribe antes de entrenar ninguno de los 20 controladores y no cambia
ningún diseño, hiperparámetro ni regla de las secciones 1–9.

### 11.1 Entrenamiento

- `python scripts/v2/run_control_stage2.py --workers 4` lanza `training/train_controller_v2.py`
  para `dream_lstm` y `dream_transformer`, semillas 0–9, intercaladas por semilla. Cada uno usa los
  valores por defecto: 50,000 pasos imaginados, el modelo del mundo de su misma semilla y la
  selección en validación 21000–21004.
- **4 procesos.** Cada proceso hijo usa un solo hilo de PyTorch/BLAS, para no saturar los 12 hilos
  de la CPU. No cambia lo que se entrena.
- **Salidas:** `models/checkpoints/v2/control/dream_<arq>_s<i>/`, con `train.log` en cada carpeta.
- **Reanudable:** una corrida con `run_info.json` se salta; una carpeta a medio escribir se
  reentrena desde cero con la misma semilla.
- **Memoria:** había 1.99 GB disponibles. Con la autorización del autor se cerraron Steam, Edge y
  Copilot, y quedaron **3.12 GB**. El lanzador se niega a arrancar con 2 GB o menos.

### 11.2 Evaluación en test (una sola vez, cuando terminen los 20)

Antes de la evaluación se agrega aquí la nota "evaluación en test iniciada" con la hora. Después
se corre una sola vez:

```
python scripts/v2/evaluate_control_v2.py --split test --confirm-held-out --reference-agreement \
  --policy sueno_lstm=dream:<las 10 carpetas dream_lstm_s0..s9> \
  --policy sueno_transformer=dream:<las 10 carpetas dream_transformer_s0..s9> \
  --policy fijo_2_3=ref:fijo_2_3 --policy min_verde_y_cambiar=ref:min_verde_y_cambiar \
  --policy cola_mas_larga=ref:cola_mas_larga --policy max_presion=ref:max_presion \
  --policy espera_mas_larga=ref:espera_mas_larga --level 0.9916666667 \
  --compare sueno_lstm:sueno_transformer \
  --compare <cada brazo>:espera_mas_larga --compare <cada brazo>:espera_mas_larga:B0 \
  --compare <cada brazo>:min_verde_y_cambiar:C0 \
  --output docs/results/v2/control/test_stage2
```

`python scripts/v2/analyze_control_stage2.py` lee ese resultado sin simular nada. El script se
escribió y se probó con datos sintéticos antes de la evaluación, y aplica:

- **P1:** IC de Welch al 99.17% sobre las medias por semilla. Si excluye 0, se adopta el brazo
  mejor; si no, la LSTM, por el desempate pre-registrado.
- **P4–P6:** sueño adoptado frente a `espera_mas_larga` (total y B0) y a `min_verde_y_cambiar`
  (C0), significativas solo con p < 0.05/6 en el test de nivel de semilla. Las mismas
  comparaciones del otro brazo son descriptivas.
- **P2–P3:** se calculan en la etapa 3. Bonferroni sigue siendo sobre 6.

**Además, descriptivo, por controlador:**

- Retorno total y por intersección, y catastróficos (< −3,600).
- **Cambios de fase reales por semáforo** (media por episodio). Un controlador es **"bloqueado"**
  si hace, en promedio, **menos de 3 cambios por episodio en algún semáforo**; se cuentan por
  arquitectura.
- **Mejor evaluación de la selección:** su número (de 10), el paso en que ocurrió y el retorno
  real medio en validación 21000–21004 (`evaluations.npz`).
- Pasos reales consumidos (`run_info.json`) y las dos contabilidades de interacciones.
- **Acuerdo con cada referencia por intersección, en los mismos estados.** En cada paso de la
  trayectoria del controlador se pregunta a las 5 referencias qué harían en cada semáforo, sin
  ejecutarlo, y se registra la fracción de pasos en que coinciden. Un test comprueba que esto no
  altera la trayectoria. Limitación: mide el parecido en los estados que visita el controlador, no
  en los que visitaría la referencia.

Si los resultados salen malos, no se cambia nada del sueño ni del dataset: se reporta y se decide
con el autor.

### 11.3 Evaluación en test iniciada: 2026-10-04 14:38 (hora local)

Los 20 entrenamientos terminaron sin fallos (40 min con 4 procesos). Cada uno hizo 50,176 pasos
imaginados y 3,000 pasos reales de selección. Hasta este momento no se ha visto ningún resultado en
test. Lo siguiente es la evaluación única de la sección 11.2.

### 11.4 Resultados de la etapa 2 en test (escrito después de la evaluación única)

Fuentes:

- `docs/results/v2/control/test_stage2.json` / `.csv`: una sola evaluación, 20 controladores y 5
  referencias × 24 semillas (22000–22023).
- `test_stage2_analysis.json`, de `scripts/v2/analyze_control_stage2.py`, escrito antes de la
  evaluación.

No se repitió ni se reevaluó nada.

**Retorno en test** (media de 24 escenarios; en los brazos aprendidos, media de 10 semillas):

| | Total | Mediana | A0 | B0 | C0 | D0 | Catastróficos (< −3,600) |
|---|---|---|---|---|---|---|---|
| `sueno_lstm` | −16,168.1 | −3,225.0 | −3,394.4 | −2,247.3 | −5,432.6 | −5,093.7 | **118/240** |
| `sueno_transformer` | −2,340.4 | −1,567.0 | −581.3 | −502.9 | −943.8 | −312.4 | **18/240** |
| `fijo_2_3` | −1,591.7 | −1,617.0 | −347.9 | −209.5 | −864.2 | −170.2 | 0/24 |
| `min_verde_y_cambiar` | −1,309.7 | −1,305.5 | −451.5 | −230.4 | **−365.3** | −262.5 | 0/24 |
| `cola_mas_larga` | −1,260.0 | −1,049.5 | −565.8 | −107.0 | −485.0 | **−102.2** | 1/24 |
| `max_presion` | −7,565.5 | −6,555.5 | −1,585.2 | −2,916.0 | −1,086.8 | −1,977.5 | 22/24 |
| `espera_mas_larga` | −1,279.5 | **−1,010.0** | −556.5 | −107.9 | −486.4 | −128.8 | 1/24 |

**Medias por semilla (total):**

- LSTM: −25,000.8, −24,022.8, −3,220.3, −8,558.6, −1,826.5, −66,496.6, −16,734.8, −3,007.3,
  −6,065.1 y −6,747.7.
- Transformer: −1,515.0, −7,337.8, −1,367.9, −1,471.4, −1,220.2, −1,820.2, −2,609.8, −1,346.9,
  −3,092.8 y −1,622.2.

#### P1 y decisión de arquitectura

| | Diferencia | Welch por semilla | IC al 99.17% | t pareada | Wilcoxon | Escenarios ganados |
|---|---|---|---|---|---|---|
| LSTM − Transformer | −13,827.6 | p = 0.053 | [−34,707.5, +7,052.2] | p = 4.0e-10 | p = 1.2e-7 | 0/24 |

- **El IC incluye 0, así que por la regla pre-registrada (sección 5.2) se adopta la LSTM**, por
  desempate.
- **Esto hay que leerlo con cuidado:**
  - El desempate se escribió para el caso de dos brazos parecidos. Aquí la LSTM es mucho peor en la
    media y la mediana, y en los 24 escenarios.
  - El IC es ancho por la enorme varianza entre semillas de la LSTM (una semilla en −66,497 y dos
    en unos −25,000).
  - Las pruebas pareadas por escenario son muy significativas, pero generalizan sobre escenarios
    para estas 10 políticas, no sobre semillas de entrenamiento, y no son el test principal.
- La regla se aplica tal como está escrita. Que el autor decida si la enmienda: sería una enmienda
  **posterior a ver los datos de test** y tendría que registrarse así.

#### P4–P6 (sueño adoptado = LSTM; α' = 0.00833)

| # | Comparación | Diferencia | p por semilla | IC al 99.17% | t pareada | Wilcoxon | Gana | ¿Significativo? |
|---|---|---|---|---|---|---|---|---|
| P4 | LSTM − `espera_mas_larga`, total | −14,888.5 | 0.040 | [−35,772.6, +5,995.6] | 1.0e-10 | 1.2e-7 | 0/24 | no |
| P5 | LSTM − `espera_mas_larga`, B0 | −2,139.5 | 0.028 | [−4,888.5, +609.6] | 5.2e-6 | 1.2e-7 | 0/24 | no |
| P6 | LSTM − `min_verde_y_cambiar`, C0 | −5,067.3 | 0.245 | [−18,771.3, +8,636.7] | 1.1e-4 | 1.2e-7 | 0/24 | no |

- **Ninguna alcanza α' en el test principal: son brechas no significativas.** Todas van en contra
  del sueño: la LSTM no gana en ningún escenario.
- **H1 no tiene apoyo:** en B0 y C0 el aprendido no supera a la mejor regla, con ninguna de las dos
  arquitecturas.
- P2 y P3 (frente al RL directo) quedan para la etapa 3.

**Transformer, descriptivo (las mismas comparaciones):**

| Comparación | Diferencia | p por semilla | t pareada | Gana |
|---|---|---|---|---|
| Total, frente a `espera_mas_larga` | −1,060.9 | 0.10 | 2.7e-3 | 1/24 |
| B0, frente a `espera_mas_larga` | −395.0 | 0.071 | 3.2e-5 | 1/24 |
| C0, frente a `min_verde_y_cambiar` | −578.5 | 0.018 | 6.3e-4 | 1/24 |

El Transformer tampoco supera a la mejor referencia. Su mediana (−1,567) queda entre `fijo_2_3` y
`min_verde_y_cambiar`.

#### Cambios de fase y bloqueo

**Con la regla fijada en 11.2** (media por episodio < 3 en algún semáforo), la LSTM tiene **1/10
controladores bloqueados** (semilla 1, D0: 1.2 cambios por episodio) y el Transformer **0/10**.

**Agregado después de ver los datos, descriptivo.** La media por episodio esconde bloqueos de
episodios sueltos. Contando los episodios con menos de 3 cambios en algún semáforo:

| | Episodios | Por semáforo | Retorno medio de esos episodios | Retorno medio del resto |
|---|---|---|---|---|
| LSTM | 75/240 | D0 56, B0 17, C0 8, A0 4 | −40,463 | −5,125 |
| Transformer | 2/240 | B0 2, C0 1 | −40,404 | −2,021 |

- Por controlador de la LSTM: s1 24, s5 22, s0 11, s6 9, s3 7, s8 1, s9 1, y s2, s4 y s7 ninguno.
- **El bloqueo de una fase explica la mayor parte de lo catastrófico**, el mismo mecanismo del
  piloto (sección 10.4).

Cambios medios por episodio, por controlador (A0 / B0 / C0 / D0):

| Controlador | Cambios | Controlador | Cambios |
|---|---|---|---|
| lstm s0 | 15.3 / 22.2 / 22.2 / 5.4 | transformer s0 | 23.4 / 23.0 / 24.9 / 17.3 |
| lstm s1 | 23.1 / 24.2 / 26.8 / **1.2** | transformer s1 | 19.8 / 18.5 / 19.1 / 16.2 |
| lstm s2 | 25.3 / 16.6 / 25.2 / 14.7 | transformer s2 | 21.3 / 20.8 / 25.1 / 13.5 |
| lstm s3 | 22.8 / 3.6 / 23.6 / 21.8 | transformer s3 | 21.6 / 22.8 / 26.6 / 19.5 |
| lstm s4 | 18.3 / 14.0 / 21.3 / 27.6 | transformer s4 | 20.3 / 8.5 / 24.4 / 16.4 |
| lstm s5 | 19.0 / 12.5 / 11.2 / 10.6 | transformer s5 | 22.0 / 23.5 / 24.2 / 23.2 |
| lstm s6 | 10.8 / 18.7 / 20.6 / 3.2 | transformer s6 | 16.9 / 12.4 / 21.8 / 14.5 |
| lstm s7 | 26.4 / 26.8 / 22.3 / 24.0 | transformer s7 | 19.9 / 13.1 / 26.3 / 23.7 |
| lstm s8 | 13.7 / 20.2 / 22.6 / 22.0 | transformer s8 | 24.5 / 25.9 / 25.2 / 23.3 |
| lstm s9 | 18.4 / 17.1 / 22.0 / 17.1 | transformer s9 | 19.2 / 15.2 / 22.3 / 17.8 |

Referencias, todas iguales en los 4 semáforos: `fijo_2_3` 23 y `min_verde_y_cambiar` 29.
`espera_mas_larga` y `cola_mas_larga` hacen ≈ 18 / 8 / 21 / 8.5, y `max_presion` 13 / 4 / 17 / 5.

#### Selección del checkpoint (validación 21000–21004)

Mejor evaluación (número de 10) y su retorno real medio en validación:

| Semilla | LSTM | Transformer |
|---|---|---|
| s0 | 8: −11,712 | 8: −1,581 |
| s1 | 10: −15,508 | **1: −3,902** |
| s2 | 10: −2,214 | 10: −1,491 |
| s3 | 6: −3,734 | 10: −1,511 |
| s4 | 8: −1,783 | 8: −1,421 |
| s5 | **1: −19,585** | 10: −1,701 |
| s6 | 10: −19,513 | 5: −1,506 |
| s7 | 7: −1,865 | 10: −1,504 |
| s8 | 5: −2,246 | 10: −1,748 |
| s9 | **1: −2,891** | 10: −1,691 |

- En 4 controladores de la LSTM (s0, s1, s5, s6), **el mejor de los 10 checkpoints ya era
  catastrófico en validación**. La selección en SUMO real no puede rescatar un entrenamiento cuyos
  checkpoints son todos malos; sí lo mostró antes de test.
- En 3 casos (LSTM s5 y s9, Transformer s1) el mejor checkpoint es el primero (5,000 pasos): seguir
  entrenando en el sueño empeoró el control real.

#### Parecido con las referencias, en los mismos estados

Fracción de pasos en que la referencia elegiría la misma acción que el controlador, media de los 10
controladores. Con decisiones binarias, 0.5 es lo que daría el azar.

| Referencia | LSTM: A0 / B0 / C0 / D0 | Transformer: A0 / B0 / C0 / D0 |
|---|---|---|
| `fijo_2_3` | 0.53 / 0.54 / 0.50 / 0.55 | 0.55 / 0.54 / 0.50 / 0.55 |
| `min_verde_y_cambiar` | 0.57 / 0.54 / **0.68** / 0.46 | 0.57 / 0.56 / **0.73** / 0.55 |
| `cola_mas_larga` | 0.52 / 0.55 / 0.58 / 0.43 | 0.64 / 0.61 / 0.60 / 0.62 |
| `max_presion` | 0.54 / **0.66** / 0.54 / **0.61** | **0.68** / **0.67** / 0.56 / **0.70** |
| `espera_mas_larga` | 0.52 / 0.55 / 0.59 / 0.43 | 0.64 / 0.62 / 0.61 / 0.62 |

- La LSTM se parece poco a todas las referencias (0.43–0.68).
- El Transformer se parece más a `max_presion` en A0, B0 y D0, y a `min_verde_y_cambiar` en C0.
- Limitación: el acuerdo se mide en los estados que visita cada controlador.
- El acuerdo por controlador está en `test_stage2_analysis.json`.

#### Interacciones reales

**3,000 pasos reales de selección por semilla en los 20 controladores** (`run_info.json`), con
50,176 pasos imaginados. Por semilla: **12,600 sin compartir el dataset y 3,960 compartiéndolo
entre las 10 semillas del brazo**. La razón frente al RL directo se calcula en la etapa 3.

#### Lo que no se hizo

No se cambió nada del sueño, del dataset ni de las reglas. El RL directo y OOD son la etapa 3.

## 12. Enmienda del 4 de octubre de 2026, ESCRITA DESPUÉS de ver los resultados en test de la etapa 2

Esta sección se escribió **después de conocer los resultados en test de la etapa 2** (sección
11.4), por decisión del autor, y antes de lanzar la etapa 3 (ningún entrenamiento ni evaluación del
RL directo existía al escribirla). No reescribe ninguna regla anterior: amplía el plan y lo dice.

### 12.0 Qué cambia y qué no

1. **P1 queda registrado tal como salió.** IC de Welch al 99.17% [−34,707.5, +7,052.2], que
   incluye 0, así que por la regla de la sección 5.2 **se adopta la LSTM**. La regla no se
   reescribe. P4–P6 siguen siendo las del brazo adoptado (LSTM), ya evaluadas.
2. **Ampliación posterior a test: P2 y P3 comparan el RL directo con los DOS brazos del sueño**,
   no solo con el adoptado:

   | # | Comparación | Métrica |
   |---|---|---|
   | P1 | `sueno_lstm` − `sueno_transformer` | total |
   | P2a | `sueno_lstm` − `directo_10k` | total |
   | P2b | `sueno_transformer` − `directo_10k` | total |
   | P3a | `sueno_lstm` − `directo_30k` | total |
   | P3b | `sueno_transformer` − `directo_30k` | total |
   | P4 | `sueno_lstm` − `espera_mas_larga` | total |
   | P5 | `sueno_lstm` − `espera_mas_larga` | B0 |
   | P6 | `sueno_lstm` − `min_verde_y_cambiar` | C0 |

   **8 comparaciones planificadas → Bonferroni α' = 0.05 / 8 = 0.00625; IC al 99.375%.** Fijado
   aquí, antes de evaluar. Se aplica a las 8, también a P1 y P4–P6, que ya se evaluaron con
   α' = 0.00833. Con el nivel nuevo ninguna conclusión de 11.4 cambia (recalculado con los mismos
   datos):

   | # | p por semilla | IC al 99.375% |
   |---|---|---|
   | P1 | 0.053 | [−35,831.4, +8,176.1] |
   | P4 | 0.040 | [−36,904.3, +7,127.3] |
   | P5 | 0.028 | [−5,037.4, +758.5] |
   | P6 | 0.245 | [−19,513.9, +9,379.3] |

   Los cuatro IC siguen incluyendo 0.
3. **Test principal: Welch sobre las medias por semilla** (10 frente a 10; t de una muestra frente
   a una referencia determinista). La t pareada y el Wilcoxon por escenario se reportan solo como
   **secundarios**, marcados como **pseudorreplicación respecto del método**: sus unidades son los
   escenarios evaluados por las mismas 10 políticas, no entrenamientos independientes.
4. **OOD (23000–23029) NO se evalúa en la etapa 3.** Queda reservado para la comparación final.
   Esto reemplaza la sección 5.3 en cuanto a cuándo se corre.
5. **Cualquier rediseño del sueño (v2.1) se evaluará en semillas nuevas**, nunca en las de test
   (22000–22023) ni en OOD (23000–23029).
6. Etapa 3: los 20 entrenamientos del RL directo (10,000 y 30,000 pasos, semillas 0–9, escenarios
   de entrenamiento desde 30000, selección en validación 21000–21004) y su evaluación **solo en
   test**, una vez. El sueño y las 5 referencias **no se reevalúan**: sus episodios de test son los
   de `test_stage2.json` (episodios deterministas por escenario, mismas 24 semillas) y se combinan
   con los del RL directo para P2–P3.

### 12.1 Diagnóstico: fidelidad on-policy (descriptivo, no entrena nada)

`scripts/v2/control_fidelity_onpolicy.py` → `docs/results/v2/control/fidelity_onpolicy_validation.json`.

**Qué se hizo.**

- Cada uno de los 20 controladores del sueño se corrió en SUMO real en las semillas de validación
  21000–21005, registrando sus estados, acciones y recompensas.
- Para cada ventana (39 por episodio), el modelo del mundo de la misma semilla y arquitectura se
  sembró con la ventana real de 16 pasos y avanzó 7 pasos con **las acciones que el controlador
  tomó de verdad**. Se usa el mismo `CorridorDreamEnvironment` en que entrenó PPO.
- Se comparó el retorno imaginado (recortado, como lo vio PPO, y sin recortar) con el real.
- Total: 2,340 ventanas por arquitectura. Bootstrap por escenario: 6 unidades, así que los IC son
  orientativos.
- **Control de reproducibilidad:** en 21000–21004, el retorno real de estas trayectorias coincide
  con el mejor retorno de validación que registró la selección (`evaluations.npz`) en los 20
  controladores (diferencia máxima 0.0008).

**a) Pearson y sesgo (imaginado − real), media de los 10 controladores, recompensa recortada:**

| | Pearson on-policy | Pearson con acciones del dataset (10.2) | Sesgo on-policy | Sesgo con acciones del dataset |
|---|---|---|---|---|
| LSTM | **0.433** [0.364, 0.563] | 0.871 | **+970.1** [+564.4, +1,334.5] | +6.2 |
| Transformer | **0.667** [0.579, 0.709] | 0.897 | **+32.2** [+18.0, +44.9] | +25.0 |

- Sin recortar, el resultado es prácticamente igual: LSTM 0.440 y +965.9; Transformer 0.667 y
  +32.4.
- **El recorte no es lo que oculta el costo:** las predicciones sin recortar ya son mucho más
  suaves que la realidad.

**b) Sesgo según la fase mantenida más larga entre los 4 semáforos, al empezar a imaginar**
(pasos de control desde el último cambio real, `floor(elapsed_phase_time / 5 s)`):

| Retención | LSTM: ventanas | LSTM: real medio | LSTM: sesgo | LSTM: imaginado > real | Transformer: ventanas | Transformer: real medio | Transformer: sesgo | Transformer: imaginado > real |
|---|---|---|---|---|---|---|---|---|
| 0–5 | 1,020 | −329.1 | +85.7 | 77% | 1,555 | −231.8 | +31.3 | 71% |
| 6–10 | 318 | −658.9 | +345.9 | 84% | 400 | −235.3 | +28.2 | 65% |
| 11–19 | 402 | −1,259.4 | +842.0 | 90% | 283 | −241.2 | +32.3 | 65% |
| ≥ 20 | 600 | −3,465.0 | **+2,890.2** | 92% | 102 | −232.8 | +62.3 | 83% |

Las ventanas con alguna retención de 20 o más, según en qué semáforo ocurre:

| | LSTM | Transformer |
|---|---|---|
| A0 | 42 ventanas, sesgo +8,883 | 0 ventanas |
| B0 | 146 ventanas, sesgo +1,833 | 54 ventanas, sesgo +73 |
| C0 | 13 ventanas, sesgo +27,002 | 3 ventanas, sesgo +444 |
| D0 | 468 ventanas, sesgo +2,505 | 55 ventanas, sesgo +44 |

**Corrección a la premisa del pedido.** Retenciones de 20 pasos o más **sí existen** en el train,
pero solo en B0 y D0:

| Semáforo | Estados del train con retención ≥ 20 | Máximo |
|---|---|---|
| A0 | 1 | 20 pasos |
| B0 | 191 | 44 pasos |
| C0 | 0 | 16 pasos |
| D0 | 224 | 54 pasos |

Todos vienen de `cola_mas_larga`, que deja el verde a la arterial donde la transversal es ligera.
Así que la LSTM se bloquea sobre todo en D0, donde el dataset **sí** tiene retenciones largas, pero
de una política que las usaba cuando la transversal estaba casi vacía.

**c) ¿Sesgo positivo justo con 20 o más pasos?** **Sí, en las dos arquitecturas**: LSTM +2,890
(92% de las ventanas imaginan menos costo que el real) y Transformer +62 (83%).

**Lectura descriptiva (no prueba causalidad):**

- En los estados que visita su propio controlador, el modelo LSTM **subestima el costo de forma
  creciente cuanto más tiempo lleva una fase mantenida**, y el costo real crece mucho más rápido
  que el imaginado.
- Esto es consistente con que PPO aprendió en el sueño a mantener fases, porque allí casi no
  costaba.
- El sesgo por retención está **confundido con la congestión**: las ventanas largas son también
  las más congestionadas (real −3,465 frente a −329). Este diagnóstico no separa las dos cosas.
- El Transformer visita pocos estados de retención larga, que en sus trayectorias son benignos
  (real −233). Su sesgo crece algo con la retención, pero es chico.
- No se cambió nada del sueño ni del dataset.

### 12.2 Etapa 3: evaluación en test iniciada, 2026-10-04 23:57 (hora local)

Los 20 entrenamientos del RL directo terminaron. Cada semilla consumió 13,240 pasos reales con 10k
y 39,208 con 30k (`run_info.json`). Hubo tres lanzamientos, todos con el mismo lanzador, 4
procesos y sin cambiar nada:

1. **Interrumpido por una suspensión del equipo de 17:47 a 22:17** (eventos de Kernel-Power). Al
   reanudar, Claude Code detuvo el lanzador por memoria baja. Las 8 corridas de 30k (s0–s7) ya
   habían terminado.
2. **Detenido por Claude Code por memoria crítica**, después de terminar 10k s0–s3.
3. **Completó el resto.**

En cada relanzamiento, las corridas sin `run_info.json` se reentrenaron desde cero con la misma
semilla, y las terminadas se saltaron. Las 8 de 30k terminadas tienen el mismo md5 y la misma
fecha antes y después (72 archivos). Antes de los relanzamientos se cerró Edge, con la
autorización del autor, y el equipo quedó enchufado.

Hasta este momento no se ha visto ningún resultado del RL directo en test. Lo siguiente es la
evaluación única de los 20 RL directos en 22000–22023 y `scripts/v2/analyze_control_stage3.py`.

### 12.3 Resultados de la etapa 3 en test (escrito después de la evaluación única)

Fuentes:

- `docs/results/v2/control/test_stage3.json` / `.csv`: los 20 RL directos, una sola evaluación en
  22000–22023, con acuerdo con las referencias.
- `test_stage3_analysis.json`, de `scripts/v2/analyze_control_stage3.py`, escrito antes de la
  evaluación. Combina estos episodios con los de la etapa 2, que no se reevaluaron.

**Retorno en test** (media de 24 escenarios; en los brazos aprendidos, media de 10 semillas):

| | Total | Mediana | A0 | B0 | C0 | D0 | Catastróficos |
|---|---|---|---|---|---|---|---|
| `sueno_lstm` (adoptado) | −16,168.1 | −3,225.0 | −3,394.4 | −2,247.3 | −5,432.6 | −5,093.7 | 118/240 |
| `sueno_transformer` | −2,340.4 | −1,567.0 | −581.3 | −502.9 | −943.8 | −312.4 | 18/240 |
| `directo_10k` | −6,138.3 | −3,192.5 | −1,826.9 | −1,737.0 | −1,736.7 | −837.8 | 96/240 |
| `directo_30k` | −1,741.6 | −1,733.0 | −497.9 | −309.0 | −669.2 | −265.4 | 2/240 |
| `espera_mas_larga` | −1,279.5 | −1,010.0 | −556.5 | −107.9 | −486.4 | −128.8 | 1/24 |
| `min_verde_y_cambiar` | −1,309.7 | −1,305.5 | −451.5 | −230.4 | −365.3 | −262.5 | 0/24 |
| `cola_mas_larga` | −1,260.0 | −1,049.5 | −565.8 | −107.0 | −485.0 | −102.2 | 1/24 |
| `fijo_2_3` | −1,591.7 | −1,617.0 | −347.9 | −209.5 | −864.2 | −170.2 | 0/24 |
| `max_presion` | −7,565.5 | −6,555.5 | −1,585.2 | −2,916.0 | −1,086.8 | −1,977.5 | 22/24 |

**Medias por semilla del RL directo (total):**

- 10k: −5,979.8, −7,513.5, −2,677.4, −3,578.4, −4,877.2, −5,163.3, −19,850.2, −3,028.6, −3,915.6
  y −4,799.0.
- 30k: −1,833.5, −1,573.4, −1,714.2, −1,572.9, −1,888.7, −1,908.0, −1,550.6, −1,784.2, −1,934.6
  y −1,655.8.

#### Las 8 comparaciones planificadas

α' = 0.05/8 = 0.00625, IC al 99.375%. Test principal: Welch sobre las medias por semilla (t de una
muestra contra una referencia). La t pareada y el Wilcoxon por escenario son secundarios y
**pseudorreplicación respecto del método**.

| # | Comparación | Diferencia | p principal | IC al 99.375% | ¿Significativo? | t pareada (secundaria) | Wilcoxon (secundario) | Escenarios ganados |
|---|---|---|---|---|---|---|---|---|
| P1 | LSTM − Transformer | −13,827.6 | 0.053 | [−35,831.4, +8,176.1] | no | 4.0e-10 | 1.2e-7 | 0/24 |
| P2a | LSTM − directo 10k | −10,029.8 | 0.148 | [−32,027.3, +11,967.8] | no | 1.1e-6 | 1.2e-6 | 1/24 |
| P2b | Transformer − directo 10k | **+3,797.9** | 0.046 | [−1,867.0, +9,462.7] | no | 1.2e-6 | 1.2e-7 | **24/24** |
| P3a | LSTM − directo 30k | −14,426.5 | 0.045 | [−36,442.2, +7,589.2] | no | 2.6e-10 | 1.2e-7 | 0/24 |
| P3b | Transformer − directo 30k | −598.8 | 0.336 | [−2,680.2, +1,482.5] | no | 0.022 | 0.0072 | 5/24 |
| P4 | LSTM − `espera_mas_larga` (total) | −14,888.5 | 0.040 | [−36,904.3, +7,127.3] | no | 1.0e-10 | 1.2e-7 | 0/24 |
| P5 | LSTM − `espera_mas_larga` (B0) | −2,139.5 | 0.028 | [−5,037.4, +758.5] | no | 5.2e-6 | 1.2e-7 | 0/24 |
| P6 | LSTM − `min_verde_y_cambiar` (C0) | −5,067.3 | 0.245 | [−19,513.9, +9,379.3] | no | 1.1e-4 | 1.2e-7 | 0/24 |

- **Ninguna de las 8 alcanza α' en el test principal: todas son brechas no significativas.**
- Las brechas numéricas más claras:
  - **P2b:** el sueño con Transformer supera al directo de 10k en los 24 escenarios (+3,798, p
    por semilla 0.046).
  - **P3b:** el directo de 30k supera al sueño con Transformer (−599, p 0.34; el sueño gana 5/24).
  - En todo lo demás, la LSTM adoptada queda por debajo.
- Ningún brazo aprendido supera a la mejor referencia en el total. El directo de 30k (−1,741.6)
  también queda por debajo de `espera_mas_larga` (−1,279.5), de forma descriptiva.

#### Interacciones reales por semilla (`run_info.json`)

| | Pasos reales | Frente al sueño sin compartir el dataset (12,600) | Frente al sueño compartiéndolo entre 10 semillas (3,960) |
|---|---|---|---|
| Sueño (los dos brazos) | 3,000 de selección + el dataset de 9,600 | — | — |
| Directo 10k | **13,240** | **1.05x** | **3.3x** |
| Directo 30k | **39,208** | **3.1x** | **9.9x** |

La razón va de **1.05x a 3.1x sin compartir el dataset** (la comparación conservadora) y de **3.3x
a 9.9x compartiéndolo**. Como ningún brazo del sueño supera de forma significativa a ningún brazo
directo, en la v2 **no hay ventaja de eficiencia demostrada**:

- El sueño con Transformer gana en los números al directo de 10k, que cuesta 1.05x–3.3x más.
- El sueño con Transformer pierde en los números contra el directo de 30k, que cuesta 3.1x–9.9x
  más.
- La LSTM adoptada pierde contra los dos.

#### Bloqueo y selección del RL directo

**Bloqueados, con la regla de 11.2:** **0/10 en 10k y 0/10 en 30k**.

**A nivel de episodio, descriptivo:**

| | Episodios con menos de 3 cambios en algún semáforo | Retorno medio de esos episodios |
|---|---|---|
| Directo 10k | 4/240 (B0 3, D0 1) | −27,201 |
| Directo 30k | 0/240 | — |
| Sueño LSTM | 75/240 | — |
| Sueño Transformer | 2/240 | — |

- Los catastróficos del directo de 10k (96/240) **no vienen del bloqueo**: hace 12–27 cambios
  por semáforo.
- **Selección, directo de 10k:** la mejor evaluación es casi siempre la última o la penúltima
  (9 o 10 de 10; s5, la 5). Todavía estaba aprendiendo.
- **Selección, directo de 30k:** la mejor está entre la 23 y la 30 de 30 (s5, la 15). Su
  retorno en validación va de −1,495 a −1,901.

#### Parecido con las referencias, en los mismos estados

Fracción de pasos con la misma decisión, media de los 10 controladores; 0.5 es lo que daría el
azar. Formato A0 / B0 / C0 / D0.

| Referencia | Sueño LSTM | Sueño Transformer | Directo 10k | Directo 30k |
|---|---|---|---|---|
| `fijo_2_3` | 0.53 / 0.54 / 0.50 / 0.55 | 0.55 / 0.54 / 0.50 / 0.55 | 0.53 / 0.53 / 0.49 / 0.53 | 0.53 / 0.53 / 0.49 / 0.53 |
| `min_verde_y_cambiar` | 0.57 / 0.54 / 0.68 / 0.46 | 0.57 / 0.56 / **0.73** / 0.55 | 0.56 / 0.48 / 0.60 / 0.47 | 0.56 / 0.49 / **0.66** / 0.52 |
| `cola_mas_larga` | 0.52 / 0.55 / 0.58 / 0.43 | 0.64 / 0.61 / 0.60 / 0.62 | 0.54 / 0.50 / 0.53 / 0.55 | 0.63 / 0.63 / 0.59 / 0.63 |
| `max_presion` | 0.54 / 0.66 / 0.54 / 0.61 | **0.68 / 0.67** / 0.56 / **0.70** | 0.56 / **0.65** / 0.52 / **0.64** | **0.64 / 0.69** / 0.55 / **0.67** |
| `espera_mas_larga` | 0.52 / 0.55 / 0.59 / 0.43 | 0.64 / 0.62 / 0.61 / 0.62 | 0.54 / 0.50 / 0.54 / 0.55 | 0.64 / 0.63 / 0.60 / 0.63 |

- **El sueño con Transformer y el directo de 30k se parecen entre sí en el patrón de acuerdo:**
  más cerca de `max_presion`, `cola_mas_larga` y `espera_mas_larga` en A0, B0 y D0, y de
  `min_verde_y_cambiar` en C0.
- El directo de 10k y la LSTM quedan más cerca del azar.
- Limitación: el acuerdo se mide en los estados que visita cada controlador.

#### Lo que no se hizo

No se cambió nada del sueño, del dataset ni de las reglas. OOD sigue reservado (12.0, punto 4).

## 13. Semillas para validar la v2.1 (definidas el 5 de octubre de 2026; todavía sin usar)

Escrito después de la etapa 3 y **antes de diseñar ninguna v2.1**. Ninguna de estas semillas se ha
simulado. El test de la v2 (22000–22023) ya se vio completo y no sirve para validar rediseños
(sección 12.0, punto 5), y OOD (23000–23029) sigue reservado para la comparación final.

| Rango | Escenarios | Uso |
|---|---|---|
| **24000–24023** | 24 | **Validación de la v2.1:** decisiones de diseño, selección de checkpoints (como en la v2, con las 5 primeras, 24000–24004, en la evaluación periódica), umbrales y mejores referencias. |
| **25000–25047** | **48** | **Test de la v2.1:** comparación principal, **una sola vez**, al final, con todo ya decidido y pre-registrado. |
| 26000–26999 | — | Reserva por si la v2.1 recolecta un dataset nuevo. No se evalúa nada en este rango. |

**Por qué 48 en el test.** Con 24 escenarios, las pruebas por escenario de la v2 fueron muy
significativas, pero el test principal es por semilla y su potencia depende de las 10 semillas,
no de los escenarios. Duplicar los escenarios reduce el ruido de cada media por semilla (cada
controlador se evalúa en más tráfico) a un costo moderado: unos 2.5 s por episodio, es decir,
≈ 40 min para 20 controladores. La validación queda en 24, como en la v2.

**Comprobación de que no se solapan** (hecha con un script, contra todas las semillas usadas en el
proyecto):

| Grupo | Rangos |
|---|---|
| v2, Fase 0, demanda y dataset | calibración 11000–11019, piloto del dataset 19000–19017 |
| v2, splits | train 20000–20111, validación 21000–21023, test 22000–22023, OOD 23000–23029 |
| v2, entrenamiento del RL directo | **30000–30503** usadas, reserva 30000–30999 |
| Tests | 19900–19901 (concurrencia) |
| v1 | 3000–3014, 5000–5014, 7000–7029, 9000–9035, entrenamiento del RL directo 10000–~10700, evaluación periódica 20000–20004 |

Resultado: **ningún solapamiento**, ni con esos rangos ni entre los tres nuevos.

**Corrección.** El entrenamiento del RL directo de la v2 no usó solo 30000–30009. Cada episodio
consume una semilla nueva (`ReseedingWrapper.training_seeds(30000)`):

| Corrida | Episodios | Semillas |
|---|---|---|
| 30k | 504 | 30000–30503 |
| 10k | 171 | 30000–30170 |

Un RL directo de la v2.1 con hasta ~1,000 episodios sigue dentro de la reserva 30000–30999.

Cuando se use, el rango se agrega a `SPLIT_SEEDS` de `scripts/v2/evaluate_control_v2.py`, con la
misma confirmación explícita para el test que tienen `test` y `ood`, y se pre-registra antes de
entrenar.
