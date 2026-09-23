# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: el controlador PPO **v1** (`dream_max_steps=7`) es el resultado
oficial del método World Model, evaluado contra SUMO real con el puente
`EncodedTrafficEnvironment` ya corregido (bug de normalización, `990c6e5`); 36/36 tests en
verde. **v1 supera a tiempo fijo en los 30 episodios evaluados (dos semillas), sin
solapamiento de rangos, con ~50% menos espera y ~31% menos cola; en throughput empata o
queda levemente por debajo.** La historia completa, incluido un diagnóstico y un fix
(`dream_max_steps=20`) que resultaron ser artefactos de ese bug, está en la sección
siguiente.

## ✅ Controlador PPO contra SUMO real: historia completa y resultado oficial (v1)

Esta sección reemplaza por completo la versión anterior. Cuenta en orden cronológico lo
que realmente pasó. **Los pasos (a)–(d) se hicieron sin saberlo con un puente de
evaluación defectuoso**; sus números se conservan como registro, no como resultados
válidos. El resultado oficial está en la tabla final de esta sección.

**Infraestructura de evaluación**: `environments/encoded_traffic_environment.py`
(`EncodedTrafficEnvironment`, commit `915bd79`) envuelve `TrafficEnvironment` y corre el
Encoder congelado en vivo para traducir el estado crudo de 26 dimensiones a `z` antes de
que la política PPO (entrenada enteramente en `z` dentro del Dream Environment) lo vea.
Junto con él, `scripts/evaluate_controller_sumo.py` compara PPO contra tiempo fijo y
acción aleatoria con métricas reales de tráfico (espera, cola, throughput) de
`TrafficEnvironment`, no solo con el reward abstracto. Protocolo: 15 episodios por
semilla base, semillas de evaluación `seed_base=3000` y `seed_base=5000`.

### (a) v1 entrenado (`max_dream_steps=7`) y primera evaluación, con el bug sin saberlo

v1 es el PPO entrenado dentro del Dream Environment con su horizonte por defecto de 7
pasos; el detalle del entrenamiento está en la sección "Controlador PPO — implementado y
entrenado dentro del Dream Environment", más abajo. Primera evaluación en SUMO: con 5
episodios PPO parecía ganar a tiempo fijo (-453.18 frente a -547.40). Con 15 episodios
(`seed_base=3000`) se invirtió: **-682.09 ± 652.41** frente a -570.27 ± 86.70, con un
patrón bimodal. 11 de 15 episodios eran mejores que tiempo fijo (media ≈-348), pero 4
eran catastróficos: -927.80, -1033.40, -2540.60 y -1902.50.

### (b) Diagnóstico, también con el bug sin saberlo

Tres de los cuatro episodios catastróficos (seeds 3001, 3011, 3013) tenían rachas de la
misma acción de 6, 9 y 14 pasos, más largas que el horizonte de 7 pasos del Dream
Environment. Se concluyó que la política no había podido experimentar esas rachas
durante el entrenamiento. El cuarto (seed 3005) se clasificó como un modo de fallo
distinto: un pico puntual de `waiting_total=83.00` en el paso 30, sin racha anormal
(`max_run=4`). Quedó documentado como limitación conocida no resuelta. **Ver (i): todo
este diagnóstico resultó ser un artefacto del bug.**

### (c) Fix aplicado por ese diagnóstico: `dream_max_steps=20` → v2

Nuevo campo `ControllerConfig.dream_max_steps=20` (commit `8d84d52`), exclusivo de este
entrenamiento, sin tocar el default de `DreamEnvironment` (7). Se respaldó v1 como
`best_model_v1_dream7.zip`, `ppo_controller_final_v1_dream7.zip` y
`evaluations_v1_dream7.npz`, y se reentrenó completo como v2 (50,176 timesteps).

### (d) v2 evaluado, todavía con el bug: parecía mejor que v1

```
                              |            reward |    espera_prom |    cola_prom |   throughput
v2 (con bug), seed_base=3000  | -290.05 +/-  57.70 | 3.92 +/- 0.81 | 1.09 +/- 0.16 | 13.40 +/- 3.88
v2 (con bug), seed_base=5000  | -316.76 +/-  52.76 | 4.28 +/- 0.73 | 1.16 +/- 0.16 | 12.40 +/- 2.92
```

Sin episodios catastróficos en ninguna semilla. Se documentó como resultado final del
método (commit `a5483e1`). **Estos números no son válidos**: ver (e)–(f).

### (e) Se encuentra el bug real: el puente no normalizaba con `scaler.pkl`

Al preparar el baseline de RL directo se vio que `EncodedTrafficEnvironment` codificaba
el estado crudo de SUMO **sin aplicar `scaler.pkl`**, el scaler con el que
`scripts/normalize_dataset.py` normalizó los datos de entrenamiento del Autoencoder. El
`z` resultante caía fuera del espacio latente donde aprendieron el LSTM y el PPO. Fix
(commit `990c6e5`): cargar `scaler.pkl` en `__init__` y aplicar
`(raw - state_mean) / state_std` antes de `encode()`, con el test
`test_state_is_normalized_with_scaler_before_encode`.

Antes de diagnosticar, se descartó que `scaler.pkl` y `autoencoder_best.pt` vinieran de
datasets distintos. Ambos salen de la misma pasada del pipeline (21/09, 22:26:08 a
22:27:06), sobre los 40 episodios (28/6/6) generados con el fix de fase ya aplicado: la
fase alterna (52.2%/47.8%) y `remaining` está en [0, 5], no en el 86400 del bug antiguo.
Numéricamente: `scaler.mean` coincide exactamente con la media de `train_raw.npz`;
`train.npz` es exactamente `(train_raw - mean) / std`; el Autoencoder reconstruye
`train.npz` con MSE 0.0186 frente a 15.83 sobre el estado crudo; y `train_latent.npz` es
bit a bit `encode(train.npz)`.

### (f) v2 re-evaluado con el puente corregido: peor de lo documentado, aún mejor que tiempo fijo

```
                              |            reward |    espera_prom |    cola_prom |   throughput
v2 (corregido), seed_base=3000| -421.69 +/-  98.90 | 5.82 +/- 1.44 | 1.38 +/- 0.23 | 13.60 +/- 2.50
v2 (corregido), seed_base=5000| -419.65 +/-  75.22 | 5.81 +/- 1.07 | 1.36 +/- 0.18 | 13.40 +/- 2.65
```

Sigue superando a tiempo fijo, comparando episodio por episodio con la misma semilla
(15/15 y 14/15), pero los rangos ya se solapan y el margen es menor: ~28-32% menos espera
y ~17-21% menos cola. Sin episodios catastróficos.

### (g) Por qué el `z` con bug daba mejor reward en v2: investigado, causa no identificada

Hipótesis probadas en orden:

1. **"El `z` con bug está degenerado/saturado": descartada.** Sobre 10 pasos reales
   (seed 3000, acciones aleatorias), el `z` sin normalizar tenía **mayor** desviación
   estándar que el correcto en **16/16** dimensiones. No era degeneración sino estar
   fuera de rango: [-20.72, 22.54] frente a [-13.51, 11.30] en todo `train_latent.npz`
   (dim 15: media 10.25 frente a un máximo de entrenamiento de 8.24). El `z` correcto
   cae en [-6.18, 4.30].
2. **Qué decide la política con cada `z` sobre los mismos estados.** Comparación
   contrafactual: 15 trayectorias guiadas por la versión correcta, 900 pasos, y en cada
   paso se pregunta qué habría elegido cada versión. Discrepan en el 32.2% de los pasos,
   con un sesgo en una sola dirección: cuando la correcta elige 1, la del bug elige 0 en
   273/453 (60%); a la inversa, solo en 17/447 (3.8%). Las rachas contrafactuales llegan
   a 12, frente a un máximo de 6 en la versión correcta. Estas cifras son decisiones
   sobre estados ajenos, no el comportamiento de la versión con bug en su propia
   trayectoria.
3. **"El bug hace menos cambios de fase y ahorra amarillo": descartada** con
   trayectorias propias (cada política conduce SUMO, seeds 3000–3014; el amarillo se
   mide por segundo simulado leyendo `TrafficSignal.is_yellow`):

   ```
   Politica                 |       reward medio | cambios/ep | amarillo_s/ep | verde F0 % | verde F1 %
   v2 corregido             |  -421.69 +/-  98.90 | 24.0 +/- 1.1 |  48.0 +/- 2.2 |       50.4 |       49.6
   v2 con bug               |  -290.05 +/-  57.70 | 26.5 +/- 0.6 |  52.9 +/- 1.2 |       56.3 |       43.7
   Tiempo fijo (ciclo=5)    |  -570.27 +/-  86.70 | 22.0 +/- 0.0 |  44.0 +/- 0.0 |       65.6 |       34.4
   ```

   La versión con bug hacía **más** cambios y más amarillo, no menos. Estas corridas
   reproducen exactamente los rewards ya reportados de las tres políticas.
4. **Mecanismo causal: NO identificado.** El bug sesgaba la política hacia la fase 0,
   pero por qué ese patrón puntuaba mejor queda como **curiosidad abierta**, no
   investigada a fondo. La única hipótesis no descartada, que no se midió, es el
   *momento* de los cambios respecto al estado de las colas.

### (h) Re-evaluación de v1 con el puente corregido, por si el diagnóstico original también estaba contaminado

Mismo protocolo, sin reentrenar. Como control, el mismo checkpoint v1 con el puente
viejo reproduce **exactamente** -682.09 ± 652.41, con los mismos cuatro episodios
catastróficos en las mismas semillas (3001, 3005, 3011, 3013). El checkpoint es el
diagnosticado en (a)–(b), y lo único que cambia entre corridas es la normalización.

### (i) Resultado: v1 corregido es el mejor controlador del proyecto; el diagnóstico de (b) era un artefacto

- **Cero episodios catastróficos en v1**: el peor de 30 es -366.00. Las seeds que antes
  fallaban dan ahora -271.00 (3001), -312.00 (3005), -263.10 (3011) y -312.10 (3013).
  Esto incluye la seed 3005, el "modo de fallo distinto": también era el bug.
- **v1 supera a v2 corregido en las 30 comparaciones** episodio por episodio (15/15 en
  cada semilla), por unos 130 puntos de reward y con menos varianza.
- **v1 supera a tiempo fijo en las 30 comparaciones, sin solapamiento de rangos**:
  v1 [-338.00, -257.00] frente a tiempo fijo [-766.20, -425.20] (`seed_base=3000`);
  v1 [-366.00, -232.00] frente a [-808.20, -418.20] (`seed_base=5000`).
- Conclusión: las rachas de acción largas de (b) no eran una limitación real de
  `max_dream_steps=7`. Eran el efecto de alimentar a la política con un `z` fuera de
  distribución. Con el `z` correcto, el horizonte de 20 **empeoró** la política real.

### (j) Decisión final

Se revierte `ControllerConfig.dream_max_steps` a 7 (commit posterior a `990c6e5`) y v1
pasa a ser el checkpoint oficial (`models/checkpoints/controller/best_model.zip`). v2 se
conserva como `best_model_v2_dream20_deprecated.zip`, como evidencia de un cambio que
parecía buena idea y no lo fue.

### Resultado oficial del método World Model (PPO v1, puente corregido)

```
                               |             reward |     espera_prom |     cola_prom |    throughput
PPO v1, seed_base=3000         |  -287.76 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
PPO v1, seed_base=5000         |  -293.35 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
Tiempo fijo, seed_base=3000    |  -570.27 +/-  86.70 |  8.05 +/-  1.25 | 1.66 +/- 0.21 | 13.67 +/- 2.55
Tiempo fijo, seed_base=5000    |  -605.27 +/- 123.47 |  8.57 +/-  1.75 | 1.73 +/- 0.28 | 13.87 +/- 2.63
Aleatoria, seed_base=3000      | -1753.63 +/- 946.86 | 26.29 +/- 14.62 | 3.12 +/- 1.17 | 14.47 +/- 2.80
Aleatoria, seed_base=5000      | -1702.63 +/- 879.77 | 25.51 +/- 13.62 | 3.03 +/- 1.05 | 12.87 +/- 2.25
```

Frente a tiempo fijo: reward ~50% mejor (49.5% y 51.5%), **espera ~53-55% menor** (52.5%
y 54.5%), **cola ~31-32% menor** (30.7% y 32.4%). **Throughput: PPO sigue sin ser mejor
que tiempo fijo** (13.47 frente a 13.67; empate exacto en 13.87). Esta salvedad se ha
mantenido en todas las corridas del proyecto, con y sin bug.

### Lección metodológica

Se investigó un fallo y se "arregló" (`dream_max_steps=20`, un reentrenamiento completo)
**sin haber descartado antes un bug en la propia herramienta de evaluación**. El
diagnóstico de (b) era internamente coherente (rachas más largas → peores episodios) y
por eso pareció confirmado, pero medía un síntoma del puente defectuoso, no del
controlador. Además, el "fix" pareció funcionar en (d) porque se evaluó con la misma
herramienta rota. Lección: cuando una evaluación da resultados inesperados, verificar
primero que la herramienta de evaluación reproduce fielmente las condiciones de
entrenamiento (aquí, la misma normalización de entrada) antes de cambiar el modelo o su
entrenamiento. Y re-evaluar todo lo medido con una herramienta después de corregirla, no
solo el último resultado.

### Notas técnicas (observaciones, no corregidas)

- **Semántica de la acción**: la acción **no** significa "mantener/cambiar" como dice el
  docstring de `ProjectActionSpace`. sumo-rl la trata como **índice de fase verde
  destino** (`TrafficSignal.set_next_phase`): solo hay cambio si
  `new_phase != green_phase` y ya pasaron `yellow_time + min_green`. Con 2 fases,
  `acción=1` equivale a "cambiar" solo cuando la fase actual es la 0; cuando es la 1,
  `acción=1` significa mantener y `acción=0` significa cambiar. En las trayectorias
  medidas cada fase ocupa ~50% del tiempo, así que la coincidencia con la documentación
  es de alrededor de la mitad de los pasos. El pipeline es coherente internamente
  (dataset, LSTM y PPO usan la misma convención), pero hay que leer retroactivamente así
  los nombres ya usados: "Siempre cambiar (1)" = "siempre pedir la fase 1"; "Siempre
  mantener (0)" = "siempre pedir la fase 0"; una racha de acción = "sostener una fase";
  "Tiempo fijo (ciclo=5)" no es un ciclo simétrico (63.3% de los pasos en fase 0). Queda
  pendiente en TODO.md.
- **La política aleatoria se reproduce exactamente entre corridas** (-1753.63 ± 946.86 y
  -1702.63 ± 879.77 en todas las ejecuciones, incluso en procesos distintos), aunque usa
  `np.random` sin semilla explícita. Explicación más probable, no verificada: al cargar
  el modelo, `PPO.load` llama a `set_random_seed` con la semilla guardada, lo que fija la
  semilla global de numpy antes de que corra la política aleatoria. No se investigó más.

**Tests**: 36/36 en verde (`test_controller_config_rejects_invalid_dream_max_steps`,
`test_state_is_normalized_with_scaler_before_encode`, más los previos).

## ✅ Controlador PPO — implementado y entrenado dentro del Dream Environment

**Commit**: `86228ae` (`configs/controller.py`, `training/train_controller.py`,
`scripts/evaluate_controller.py`, `tests/test_controller.py`; 4 tests nuevos, 33/33 en
verde). Los checkpoints generados (`models/checkpoints/controller/*.zip`,
`evaluations.npz`) NO se commitearon, mismo criterio que los demás pesos entrenados del
proyecto.

**Qué es**: un `PPO` de Stable-Baselines3 (`MlpPolicy`) entrenado enteramente dentro de
`DreamEnvironment` — nunca toca SUMO durante el entrenamiento. `ControllerConfig`
define los hiperparámetros (punto de partida razonable, no afinado empíricamente:
`total_timesteps=50_000`, `learning_rate=3e-4`, `n_steps=256`, `gamma=0.99`), con
`EvalCallback` evaluando sobre `validation_latent.npz` cada ~1000 timesteps y guardando
el mejor checkpoint por recompensa de validación.

**Entrenamiento real ejecutado**: 50,176 timesteps, 196 iteraciones. Curva de
evaluación **ruidosa y no monótona** — oscila entre ~-46 y ~-77 durante todo el
entrenamiento, sin convergencia limpia (mejor punto observado: -46.55 en el timestep
30,000; valor final en el timestep 50,000: -62.90). No hay una mejora clara y estable
como función del número de pasos de entrenamiento.

**Evaluación en el split de test** (`scripts/evaluate_controller.py`, nunca visto por
el PPO ni por el LSTM; 30 episodios × hasta 7 pasos = 210 pasos por política):

```
Politica                  |    mean |    std |      min |      max | streak>=5 | clipped
PPO (entrenado)           |  -53.53 |  31.65 |  -147.04 |  -17.89  |  0/210 (0.0%)  | 28/210 (13.3%)
Accion aleatoria          | -144.80 | 100.49 |  -545.40 |  -32.01  |  5/210 (2.4%)  | 16/210 (7.6%)
Siempre mantener (0)      | -521.80 | 239.24 | -1124.76 | -141.49  | 90/210 (42.9%) | 25/210 (11.9%)
Siempre cambiar (1)       | -493.38 | 197.05 |  -969.11 | -173.03  | 90/210 (42.9%) | 32/210 (15.2%)
Alternando cada paso      | -129.59 |  59.60 |  -303.31 |  -43.02  |  0/210 (0.0%)  | 10/210 (4.8%)
```

PPO supera a las 4 políticas de referencia en `mean` (-53.53, la menos negativa) y
nunca entra en racha de acción ≥5.

**⚠️ Salvedad importante, planteada sin resolver en la corrida original**: la tasa de
`reward_clipped` de PPO (13.3%) es **más alta** que la de la política "alternando"
(4.8%), a pesar de que PPO evita rachas largas por completo. Se planteó como posible
síntoma de que la política está explotando el recorte de recompensa para volver
artificialmente barato el error de extrapolación del modelo, en vez de aprender control
de tráfico genuino.

**Investigación de seguimiento — commit `8044262`**: antes de gastar cómputo en SUMO
real, se investigó la hipótesis con una medición más precisa: la **magnitud** del
recorte (`|raw_predicted_reward - bound|`), no solo su frecuencia. Se expuso
`info["raw_predicted_reward"]` (el valor crudo del LSTM antes de `torch.clamp`) en
`DreamEnvironment.step()`, y se creó `scripts/analyze_controller_actions.py` para
comparar PPO contra las políticas de referencia:

```
Rango de recorte: [-165.05, 1.0]

Politica                  |  n_clips |  mag_media |    mag_max |  accion=0 |  accion=1
-------------------------------------------------------------------------------------
PPO (entrenado)           |       25 |       3.05 |      18.25 |       100 |       110
Alternando cada paso      |        9 |       2.75 |       4.89 |       120 |        90
Siempre mantener (0)      |       29 |      10.15 |      31.60 |       210 |         0
Siempre cambiar (1)       |       34 |      32.84 |      91.25 |         0 |       210
```

**Resultado, sin suavizar**: la magnitud del recorte en PPO (`mag_media=3.05`,
`mag_max=18.25`) es cercana a la de "alternando" (`2.75`/`4.89`) y muy por debajo de
las políticas constantes (`10-33` de media, hasta `91` de máximo). Cuando PPO dispara
el recorte, el valor crudo queda apenas fuera del rango empírico — no profundamente en
territorio alucinado, como sí ocurre con las políticas constantes. La distribución de
acciones de PPO tampoco es degenerada (100 vs. 110, similar al balance de "alternando").
Esto **debilita, pero no descarta por completo**, la hipótesis de explotación del
recorte: no se puede descartar con este análisis que PPO esté eligiendo secuencias de
acción específicas (no necesariamente rachas largas ni una acción constante) que
empujan la predicción justo más allá del borde del rango con más frecuencia que
"alternando" — el conteo marginal de acciones no distingue eso de un patrón temporal
particular más sutil.

**Conclusión de esta investigación**: no resuelve la pregunta de fondo (¿control
genuino o artefacto del Dream Environment?), pero sí reduce la prioridad de la
sospecha más grave (explotación profunda del recorte) lo suficiente como para proceder
a la validación contra SUMO real sin gastar más tiempo en análisis dentro del Dream
Environment. **El resultado del PPO sigue sin estar validado** hasta esa evaluación en
`TrafficEnvironment`.

**Tests**: `test_raw_predicted_reward_exposed_in_info` — **34/34 tests en verde** (33
previos + 1 nuevo).

## ✅ Dream Environment — mitigaciones de extrapolación fuera de distribución (OOD)

**Contexto**: antes de construir el controlador PPO, se hizo una verificación manual
del `DreamEnvironment` con el checkpoint real entrenado (no pesos aleatorios), corriendo
4 políticas simples (aleatoria, siempre mantener, siempre cambiar, alternando) durante
20 episodios imaginados cada una y sumando la recompensa total por episodio.

**Hallazgo**: con `max_dream_steps=10` (el valor original), las políticas de acción
constante ("siempre mantener"/"siempre cambiar") produjeron recompensas imaginadas
totales ~4.7x-5.6x más negativas que el orden de magnitud esperado (`10 × reward_mean
por paso`), mientras que las políticas aleatoria y alternada caían justo en el rango
esperado. Diagnóstico confirmado con los datos reales: en `train_latent.npz`, de 881
rachas de acción idéntica consecutiva observadas en 28 episodios, solo **1** llega a
longitud 10 o más (una racha de 12). El LSTM prácticamente nunca vio secuencias de 10
acciones repetidas durante el entrenamiento — esas dos políticas alimentan al modelo
con entradas fuera de distribución (OOD), y el salto de magnitud es extrapolación
inestable, no una señal físicamente plausible de la dinámica del tráfico.

**Se descartó explícitamente** "corregir" la acción antes de alimentar al LSTM
replicando el filtrado de `min_green` de SUMO: el LSTM se entrenó con la acción
*solicitada* cruda (`ProjectActionSpace.sample()`), nunca con la acción realmente
aplicada tras el filtro de `min_green` — alimentarlo con la versión filtrada
introduciría un patrón distinto, igualmente no visto en entrenamiento, sin resolver
el problema real.

**Dos mitigaciones aplicadas, ambas basadas en evidencia de los datos reales, ninguna
inventada**:

1. `max_dream_steps` bajado de 10 a 7 — el histograma real de rachas
   (`{5: 27, 6: 11, 7: 5, 8: 2, 9: 3, 12: 1}`) muestra que 5-7 pasos está bien
   representado en los datos; 8+ es raro. Mejora el problema (constante-acción bajó de
   ~4.7x-5.6x a ~3.7x-3.9x del orden de magnitud esperado) pero **no lo resuelve**.
2. Recorte (`torch.clamp`) de la recompensa imaginada al rango
   `[REWARD_CLIP_MIN=-165.05, REWARD_CLIP_MAX=1.00]` — percentiles 1 y 99 de las
   recompensas reales en `train_latent.npz` (min real=-324.10, max real=2.90;
   percentiles preferidos sobre el min/max crudo para ignorar outliers extremos raros).
   Expuesto en `info["reward_clipped"]` para monitorear, una vez entrenado el PPO, qué
   tan seguido se activa.

**Verificación final, con ambas mitigaciones activas** (20 episodios × hasta 7 pasos
= 140 pasos por política):

```
Acción aleatoria     | mean= -180.75 | reward_clipped:  12/140 ( 8.6%)
Siempre mantener (0) | mean= -601.31 | reward_clipped:  23/140 (16.4%)
Siempre cambiar (1)  | mean= -533.04 | reward_clipped:  29/140 (20.7%)
Alternando cada paso | mean= -158.09 | reward_clipped:   4/140 ( 2.9%)
```

El recorte se activa 5-7 veces más seguido en las políticas de acción constante que en
aleatoria/alternada — confirma que el diagnóstico está bien dirigido. **Pero, dicho sin
suavizar**: las políticas de acción constante siguen ~3.4x-3.8x por encima del orden de
magnitud esperado (`7 × reward_mean ≈ -156.52`) incluso con el recorte activo — el
recorte acota la consecuencia de un solo paso malo, pero no corrige que el modelo
sistemáticamente predice peor (no solo en outliers) cuando la acción no cambia. Esto
queda documentado como limitación conocida, no como problema resuelto; se decidió
aprobar y avanzar con el controlador PPO de todas formas, monitoreando
`info["reward_clipped"]` y `info["consecutive_action_streak"]` una vez el PPO esté
entrenando.

**Tests**: `test_action_streak_tracked_in_info` (rastrea rachas de acción vía
`info["consecutive_action_streak"]`) y
`test_imagined_reward_is_clipped_to_empirical_range` (recorte, verificado
determinísticamente con `monkeypatch` sobre un rango estrecho) — **29/29 tests en
verde** (27 previos + 2 nuevos).

## ✅ Dream Environment — implementado (primera versión)

**Commits**: `c2bbd8c` (refactor: extrae `predict_next_step` de `rollout_episode` en
`evaluation/world_model_evaluation.py`, sin cambio de comportamiento, verificado número
por número contra la corrida anterior) y `16ec05d` (feat: `environments/dream_environment.py`).

**Qué es**: `DreamEnvironment`, una clase compatible con `gymnasium.Env` (`reset`/`step`)
que imagina trayectorias usando el `LatentDynamicsLSTM` ya entrenado, sin tocar SUMO en
ningún momento. Responde las tres preguntas de diseño que quedaban abiertas en
`TODO.md`:

- **Mecanismo de imaginación**: cada episodio imaginado se **siembra** con una ventana
  real de `sequence_length` pasos tomada de un episodio grabado (`*_latent.npz`) — el
  histórico `(z, acción)` de esa ventana es dato real; solo la acción elegida en cada
  `step()` en adelante es hipotética. `step(action)` reemplaza únicamente la última
  acción de la ventana actual y llama a `predict_next_step` (la misma función que usa
  `rollout_episode` en la evaluación), así que la lógica de ventaneo y desnormalización
  de recompensa tiene una única fuente de verdad para evaluación e imaginación.
- **Horizonte de imaginación**: `max_dream_steps=10` por defecto en esta versión
  inicial — coincide con el rango de horizontes efectivamente validado en el
  Experimento 1 (`evaluate_world_model.py` mide hasta horizonte 10); más allá de eso
  no hay evidencia de qué tan confiables son las predicciones del modelo. **Revisado
  después**: ver la sección "Dream Environment — mitigaciones de extrapolación fuera
  de distribución (OOD)" más arriba — bajado a 7 tras un sanity check manual.
- **Interfaz**: se optó por `reset`/`step` al estilo `TrafficEnvironment` (no una
  función simple de "evaluar una secuencia de acciones candidata"), para poder pasar
  `DreamEnvironment` directamente a un controlador tipo PPO de Stable-Baselines3 más
  adelante, sin una capa de adaptación intermedia.

**Verificación**: 5 tests nuevos (`tests/test_dream_environment.py`) cubriendo reset
válido, step válido, truncamiento en `max_dream_steps`, rechazo de episodios más cortos
que `sequence_length`, y rechazo de acción inválida — **27/27 tests en verde** (22
previos + 5 nuevos), sin regresiones.

**Qué NO hacía todavía en este punto**: no había ningún controlador (PPO) usándolo aún.
**Actualización**: el controlador PPO ya se implementó y entrenó — ver la sección
"Controlador PPO" al inicio de este documento.

## ✅ Sobreajuste residual del LSTM — investigado y cerrado (conclusión: no era el cuello de botella real)

**Cambios aplicados** en `training/train_world_model.py`: `weight_decay=1e-4` en el
optimizador Adam, y early stopping con paciencia de 15 épocas sin mejora en
`validation_loss`.

**Resultado**: mejora marginal en la brecha train/validation (~3.0x → ~2.6x; el early
stopping no llegó a activarse, terminó en 14/15 épocas sin mejora al llegar a la época
100). **Sin cambio significativo en las métricas de evaluación sobre test**
(`reward_mse` horizonte 1: 41.3 → 41.9; horizonte 10: 172.3 → 167.0 — diferencias
dentro del ruido de reentrenar).

**Por qué el cambio fue tan pequeño, explicación confirmada**: `evaluate_world_model.py`
ya usaba `world_model_best.pt` (el checkpoint de menor `validation_loss`, no el de la
última época) desde antes de este fix. Las épocas tardías donde `train_loss` seguía
bajando mientras `validation_loss` se estancaba **nunca afectaron el resultado
reportado** — ya estaban descartadas por la selección de mejor checkpoint. El
"sobreajuste" visible en la curva de pérdida era más cosmético que sustantivo.

**Decisión**: se aprueba y mantiene el cambio (deja infraestructura de regularización
reutilizable para cuando se entrenen Transformer/TSMixer, que podrían sobreajustar más
en serio), pero se documenta honestamente que no resolvió un problema grave porque el
problema nunca fue tan grave como parecía en la curva de pérdida. Palancas adicionales
(weight_decay más agresivo, dropout real con num_layers=2, más episodios,
sequence_length distinto) quedan disponibles pero NO se aplican ahora — no hay evidencia
de que el sobreajuste esté limitando ningún resultado actual del proyecto.

## ✅ Todo lo anterior sigue vigente sin cambios

Entorno, pipeline de dataset, fix de fase del semáforo, fix de normalización de
recompensa, Autoencoder, LSTM sobre `z`, Experimento 0 (Autoencoder confirmado, se
mantiene), y el cierre del sobreajuste residual del LSTM — sin cambios desde el
handoff anterior.

## 🟡 Pendiente

1. Cuenta o app desconocida en GitHub — sigue sin resolver.
2. `DOCUMENTACION_PROYECTO.md` sigue desactualizado (no incluye LSTM, bug de fase,
   Experimento 0, cierre del sobreajuste, ni Dream Environment).
3. Documentación de `ProjectActionSpace` engañosa: la acción es el índice de fase
   verde destino, no "mantener/cambiar" (ver notas técnicas de la sección del
   controlador PPO).

## ⚪ No implementado todavía

- **Baseline de RL directo** (PPO entrenado directo contra SUMO, sin Dream
  Environment) — pedido explícito de la Sección 18 de la propuesta, necesario para
  responder si el World Model ahorró interacciones con SUMO frente a la alternativa
  directa.
- Transformer, TSMixer, evaluación final comparativa de los tres enfoques (tiempo
  fijo vs. RL directo vs. World Model).

## Qué se estaba haciendo justo antes de este handoff

Al preparar el baseline de RL directo se encontró y corrigió un bug en
`EncodedTrafficEnvironment`: no normalizaba con `scaler.pkl` (commit `990c6e5`). Eso
obligó a re-evaluar todo lo medido con ese puente. v2 resultó peor de lo documentado;
v1, re-evaluado, no tiene ningún episodio catastrófico y supera a v2 y a tiempo fijo en
las 30 comparaciones. El diagnóstico de rachas y el fix `dream_max_steps=20` eran
artefactos del bug. Decisión: v1 es el resultado oficial, `dream_max_steps` vuelve a 7 y
el checkpoint de v2 se conserva como `best_model_v2_dream20_deprecated.zip`. Siguiente
paso: el baseline de RL directo que pide la propuesta (Sección 18), comparado contra v1.
Queda pendiente decidir si se cambia la semilla de SUMO en cada `reset` de entrenamiento
y evaluación: sin eso, sumo-rl reutiliza la misma semilla y el PPO directo vería siempre
el mismo tráfico.