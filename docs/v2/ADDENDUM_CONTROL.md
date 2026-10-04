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

(Se completa con el resultado de las referencias en validación, sin cambiar las reglas de arriba.)
