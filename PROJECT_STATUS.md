# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: escenario de demanda asimétrica (500/150 veh/h) con el pipeline
completo re-ejecutado (commits `febef8e`, `6c753d2`, `f58347e`), 39/39 tests en verde.
**Con demanda asimétrica la regla trivial "pedir siempre la fase contraria" deja de ser
la mejor política y pasa a ser la peor**, y los dos PPO (sueño y RL directo) toman
decisiones distintas de ella. Ambos superan en promedio a tiempo fijo, pero **con
episodios catastróficos que tiempo fijo no tiene y cuya causa no se identificó por
completo** (limitación abierta). Hallazgos de proceso anotados sin corregir:
`collect_dataset.py` usa siempre la misma semilla de SUMO, y los dos PPO nunca
sostienen la fase 1 más de 8 s. Ver la sección siguiente.

## ✅ Escenario de demanda asimétrica: la regla trivial se rompe, aparecen episodios catastróficos sin explicar del todo

> **Hallazgos de proceso anotados sin corregir** (ver también "🟡 Pendiente"):
> 1. `scripts/collect_dataset.py` llama a `env.reset()` sin semilla, así que los 40
>    episodios del dataset comparten la misma semilla de SUMO (42). La variedad del
>    dataset viene solo de las acciones aleatorias, no de distintas realizaciones de
>    tráfico. Pendiente decidir si esto limita al World Model.
> 2. En los 60 episodios evaluados (punto 5), ninguno de los dos PPO sostiene el verde
>    de la fase 1 (Este/Oeste) más de **8 s**.

### 1. Motivación y diseño

En el escenario simétrico (sección siguiente), los dos PPO convergieron a la misma regla
trivial y el escenario no permitía comparar calidad de control. Se cambió la demanda a
una vía principal Norte/Sur cargada y una secundaria Este/Oeste ligera, donde cambiar de
fase lo antes posible ya no debería ser lo mejor. La demanda simétrica original se
conserva en `single-intersection_symmetric_backup.rou.xml`.

**Proceso de verificación de la demanda (Fase 0)**, con una primera propuesta descartada:

- **Primera propuesta, 700/150 veh/h por brazo (descartada).** Los `vehicle_counts` en
  carril daban una razón de 12.32:1, fuera del rango esperado de 3:1 a 6:1. El conteo de
  vehículos insertados mostró la causa: la entrada real era 3.53:1, pero el carril Sur
  solo aceptó 47 de los ~58 vehículos programados. La vía principal (un carril por
  brazo) quedaba saturada y los vehículos esperaban **fuera de la red**, donde no
  aparecen ni en el estado ni en la recompensa: una cola invisible que una política
  podría explotar.
- **Demanda elegida, 500/150 veh/h por brazo**: Norte/Sur 286 recto + 129 izquierda +
  85 derecha; Este/Oeste 85 + 40 + 25. Razón configurada 3.33:1. Los criterios se
  redefinieron según lo que se puede medir de verdad:
  - Razón de entrada real **2.87:1** en episodios de 300 s. Es menor que la configurada
    por el redondeo: cada flujo inserta su primer vehículo en t=0, lo que pesa más en
    los flujos pequeños. Aceptado como límite estructural de episodios cortos.
  - **Insertados = programados** en los 12 flujos, con la política de recolección
    (acciones aleatorias) y con la regla "fase contraria": ningún vehículo queda fuera.
    Pendientes de inserción: como máximo 2 vehículos durante ≤3 s. Se deben al arranque
    en t=1–2 s, cuando los 3 flujos de cada brazo comparten un solo carril, y a
    coincidencias puntuales entre flujos. **La demanda simétrica original tenía el mismo
    patrón con más frecuencia** (11 y 14 segundos con pendientes, frente a 5 y 9 ahora).
  - Solo la política de referencia deliberadamente mala "siempre fase 1" deja vehículos
    fuera (hasta 36 pendientes). Esto confirma que la **fase 0 da verde a Norte/Sur** y
    la **fase 1 a Este/Oeste**.

### 2. Pipeline completo re-ejecutado (desde cero, sin commits intermedios)

**Cambio de checkpoints, importante para leer las secciones anteriores.** Desde este
bloque, `models/checkpoints/controller/best_model.zip` y
`models/checkpoints/controller_direct/best_model.zip` contienen los modelos del
**escenario asimétrico**. El v1 del escenario simétrico se conserva como
`controller/best_model_v1_dream7.zip` (y v2 como `_v2_dream20_deprecated`). El
checkpoint del RL directo simétrico se borró: sus números siguen documentados en la
sección siguiente, y su `.json` queda en el historial de git. Las secciones anteriores
que dicen "checkpoint oficial" se refieren al escenario simétrico.

**Dataset** (40 episodios, 28/6/6 por episodio, 0 NaN/Inf, acciones balanceadas):

| split | reward medio | std | min | max |
|---|---|---|---|---|
| train | -43.01 | 56.78 | -504.1 | 2.0 |
| validation | -34.23 | 43.71 | -309.1 | 1.0 |
| test | -66.74 | 102.87 | -765.1 | 2.0 |

La asimetría llega a los datos: `vehicle_counts` medio de Sur 8.18, Norte 4.81, Oeste
0.97 y Este 0.93. Sur carga casi el doble que Norte con la misma demanda (no
investigado). El split de test tiene una cola más pesada que train: su mínimo, -765.1,
queda fuera del rango de entrenamiento.

**Autoencoder**: pérdida final de train 0.016387 y de validación 0.021138 (mejor
0.021030). Comparable al escenario simétrico (MSE de 0.0186 sobre train).

**Recorte de recompensa del Dream Environment, recalculado** (commit `6c753d2`): los
percentiles 1 y 99 del nuevo `train_latent.npz` son **[-266.13, 1.00]**, frente a
[-165.05, 1.00] antes. Con los límites viejos se habría recortado el 4.8% de los rewards
reales de entrenamiento; con los nuevos, el 1.2%.

**LSTM**: pérdida final de train 0.069640 y de validación 0.157874 (mejor ≈0.1565 en la
época 90); razón validación/train ~2.3x. Evaluación en test frente al baseline
persistente, con el escenario simétrico como referencia:

```
  h | latent (modelo) | latent (baseline) | reward (modelo) | reward (baseline) || simetrico: latent modelo | reward modelo | reward baseline
  1 |      0.2486     |      0.9475       |      756.3      |      3233.8       ||      0.1943    |     41.9      |     716.6
  2 |      0.3868     |      1.9580       |     1622.1      |      9817.6       ||      0.2625    |     67.5      |    1203.9
  3 |      0.4874     |      2.5791       |     2535.9      |     16186.3       ||      0.3154    |     86.0      |    1359.3
  4 |      0.5399     |      2.8341       |     2762.2      |     20843.4       ||      0.3724    |    111.7      |    1379.2
  5 |      0.5646     |      2.9350       |     2493.4      |     24089.5       ||      0.4261    |    144.8      |    1279.8
  6 |      0.6066     |      3.0184       |     2364.4      |     25927.3       ||      0.4553    |    187.3      |    1069.3
  7 |      0.6469     |      3.1227       |     2518.6      |     27267.6       ||      0.4729    |    200.5      |    1065.5
  8 |      0.6887     |      3.1174       |     2736.1      |     27475.3       ||      0.4715    |    198.3      |    1157.7
  9 |      0.7221     |      3.0030       |     2866.6      |     26181.6       ||      0.4729    |    181.3      |    1300.9
 10 |      0.6966     |      2.9775       |     2885.7      |     24190.0       ||      0.4742    |    167.0      |    1303.2
```

El modelo supera al baseline en todos los horizontes. El error latente acumulado es algo
mayor (crece 2.80x de h=1 a h=10, frente a 2.44x antes). **La predicción de reward a un
paso es relativamente peor**: 23% del error del baseline, frente al 6% del escenario
simétrico. En h=10 la proporción es similar (12% frente a 13%).

**PPO del sueño** (`dream_max_steps=7`, 50,176 timesteps; reward imaginado, 20 episodios
de validación por evaluación; `*` = `best_model.zip`):

```
 1000 -265.93   2000 -156.18   3000 -162.73   4000 -155.14   5000 -143.39   6000 -153.34   7000 -137.99   8000 -132.61
 9000 -144.77  10000 -140.01  11000 -144.03  12000 -152.58  13000 -173.60  14000 -160.65  15000 -160.71  16000 -136.84
17000 -122.21  18000 -162.12  19000 -149.10  20000 -136.24  21000 -149.57  22000 -119.28* 23000 -148.65  24000 -175.37
25000 -158.04  26000 -132.42  27000 -142.79  28000 -150.63  29000 -137.49  30000 -160.72  31000 -139.37  32000 -136.61
33000 -162.88  34000 -145.18  35000 -133.46  36000 -154.68  37000 -138.27  38000 -144.98  39000 -122.26  40000 -144.84
41000 -119.51  42000 -125.91  43000 -144.58  44000 -131.89  45000 -135.31  46000 -148.30  47000 -148.61  48000 -141.42
49000 -147.75  50000 -136.99
```

Curva ruidosa y no monótona, como en el escenario simétrico. `explained_variance` entre
-0.0006 y 0.0017 durante todo el entrenamiento.

**PPO de RL directo** (10,000 timesteps reales; semillas verificadas: 171 de
entrenamiento únicas y 5 de evaluación que ciclan; `*` = `best_model.zip`):

```
 1000 -23143.46 +/- 10925.07     6000 -284.00 +/-  75.62 *
 2000  -2105.66 +/-   603.68     7000 -301.76 +/-  52.45
 3000   -362.88 +/-    74.07     8000 -310.24 +/-  65.79
 4000   -311.30 +/-    50.81     9000 -309.84 +/- 115.45
 5000   -477.68 +/-   300.22    10000 -299.76 +/-  51.07
```

A diferencia del escenario simétrico, la curva no se congela: mejora hasta el timestep
6000 y después fluctúa. **`explained_variance` se mantuvo entre -0.0016 y 0.00014 durante
todo el entrenamiento**, igual que antes: la función de valor no aprende (ver punto 5).

### 3. Resultado principal: la regla trivial se rompe

`scripts/evaluate_final_comparison.py` (commit `f58347e`), 15 episodios por semilla base:

```
seed_base=3000                 |               reward |      espera_prom |      cola_prom |       throughput
PPO (sueno)                    |   -357.89 +/-  202.71 |    4.87 +/-  2.81 |   1.26 +/- 0.58 |   11.73 +/-  1.73
    [-180.6, -431.2, -918.8, -211.6, -241.6, -332.5, -316.5, -253.8, -272.8, -524.8, -171.5, -398.0, -221.7, -682.2, -210.8]
PPO (RL directo)               |   -381.01 +/-  275.58 |    5.25 +/-  3.85 |   1.27 +/- 0.75 |   11.00 +/-  1.71
    [-148.1, -928.0, -192.9, -396.2, -509.2, -872.0, -248.0, -195.0, -213.9, -801.9, -143.0, -573.1, -166.9, -188.0, -139.0]
Tiempo fijo (ciclo=5)          |   -397.07 +/-   57.40 |    5.39 +/-  0.76 |   1.44 +/- 0.18 |   13.73 +/-  2.95
    [-360.2, -383.2, -364.2, -369.2, -421.2, -592.2, -397.2, -427.2, -372.2, -343.2, -362.2, -404.2, -369.2, -374.2, -416.2]
Regla: pedir fase contraria    |   -509.57 +/-  122.55 |    6.76 +/-  1.62 |   1.88 +/- 0.43 |   11.93 +/-  3.13
    [-559.1, -416.1, -700.1, -563.1, -400.1, -418.1, -386.1, -435.1, -356.1, -675.1, -632.1, -332.1, -696.1, -536.1, -538.1]

seed_base=5000                 |               reward |      espera_prom |      cola_prom |       throughput
PPO (sueno)                    |   -411.83 +/-  214.76 |    5.58 +/-  2.94 |   1.46 +/- 0.66 |   12.60 +/-  2.36
    [-632.2, -220.6, -378.9, -359.7, -823.0, -242.8, -676.1, -197.6, -269.0, -273.9, -164.6, -369.8, -700.1, -661.6, -207.6]
PPO (RL directo)               |   -328.02 +/-  178.20 |    4.51 +/-  2.43 |   1.14 +/- 0.53 |   12.00 +/-  2.85
    [-726.2, -707.0, -198.0, -343.9, -134.0, -222.0, -300.1, -225.0, -359.0, -313.0, -192.0, -205.0, -525.0, -289.0, -181.1]
Tiempo fijo (ciclo=5)          |   -425.47 +/-   40.71 |    5.77 +/-  0.55 |   1.52 +/- 0.13 |   13.27 +/-  2.46
    [-402.2, -425.2, -393.2, -446.2, -413.2, -537.2, -362.2, -458.2, -423.2, -456.2, -397.2, -458.2, -387.2, -426.2, -396.2]
Regla: pedir fase contraria    |   -495.50 +/-  113.06 |    6.59 +/-  1.48 |   1.84 +/- 0.39 |   13.13 +/-  2.39
    [-531.1, -660.1, -421.1, -327.1, -391.1, -608.1, -328.1, -580.1, -466.1, -703.1, -549.1, -463.1, -491.1, -357.1, -556.1]
```

- **La regla "pedir siempre la fase contraria" pasa de ser la mejor política (escenario
  simétrico) a ser la PEOR** (-509.57 y -495.50), peor incluso que tiempo fijo.
- **Tiempo fijo mejora frente a la regla**: su ciclo no es simétrico (63% del tiempo en
  fase 0, la de Norte/Sur), lo que por casualidad favorece a la vía principal.
- **Los PPO ya no siguen la regla.** Comparación contrafactual sobre los mismos estados
  reales (el PPO del sueño conduce SUMO en las 30 semillas; 1800 pasos, 666 bloqueados
  por `min_green`, 37.0%):

  ```
  Par                  | acuerdo total | desacuerdos | en pasos bloqueados | acuerdo en pasos NO bloqueados
  sueno / directo      |     81.6%     |     332     |    182 (54.8%)      |   86.8%  (150 de 1134)
  sueno / regla        |     61.1%     |     700     |    189 (27.0%)      |   54.9%  (511 de 1134)
  directo / regla      |     47.7%     |     942     |    343 (36.4%)      |   47.2%  (599 de 1134)
  ```

  En el escenario simétrico el acuerdo con la regla en pasos no bloqueados era del 100% y
  el 99.2%; ahora es del 54.9% y el 47.2%. Los dos PPO coinciden entre sí en el 86.8% de
  esas decisiones: aprendieron comportamientos parecidos por caminos distintos.

### 4. Hallazgo nuevo: mejores en promedio, pero con episodios catastróficos

| | Episodios que ganan a tiempo fijo (misma semilla) | Episodios peores que -600 | Peor episodio |
|---|---|---|---|
| PPO sueño | 11/15 + 10/15 = **21/30** | **7/30** | -918.8 |
| PPO directo | 9/15 + 12/15 = **21/30** | **5/30** | -928.0 |
| Tiempo fijo | — | **0/30** | -592.2 |

En sus buenos episodios, los PPO llegan a -130 / -250, muy por encima de tiempo fijo; en
los malos caen a -680 / -930. Tiempo fijo es entre 4 y 5 veces más estable (desviación
de 41–57 frente a 178–276). El throughput de los PPO es menor (11.0–12.6 frente a
13.3–13.7), la misma salvedad que en el escenario simétrico. Los episodios catastróficos
dependen de la política, no solo del tráfico: el PPO del sueño falla en 3002, 3013, 5000,
5004, 5006, 5012 y 5013, y el directo en 3001, 3005, 3009, 5000 y 5001. Solo comparten la
semilla 5000.

### 5. Investigación de los episodios catastróficos

Cada PPO conduce su propia trayectoria real, que reproduce exactamente la tabla del
punto 3. Se compararon los 12 episodios catastróficos con los 5 mejores de cada semilla
(10 por política), midiendo por segundo simulado.

- **Cola invisible (vehículos pendientes de inserción): DESCARTADA.** El máximo es de 8
  pendientes en los 60 episodios, catastróficos y buenos, en las dos políticas. Ese
  máximo es el efecto de arranque en t=1 s. El total es de 14–24 vehículo·s por episodio
  en ambos grupos, y su correlación con el reward en los 30 episodios es -0.09 (sueño) y
  -0.13 (directo).
- **Racha de fase 1 sostenida: DESCARTADA como causa.** Las dos políticas tienen una
  racha máxima de verde en fase 1 de **exactamente 8 s en los 60 episodios**, así que no
  distingue episodios catastróficos de buenos. Sobre el origen de esos 8 s: no es un
  tope que imponga el entorno, porque la política "siempre fase 1" de la Fase 0 sostuvo
  esa fase todo el episodio. Coincide con la duración mínima posible del verde tras un
  cambio: 3 s de verde en el paso del cambio (5 s menos 2 de amarillo) más un paso
  completo de 5 s bloqueado por `min_green`. Eso sugiere que **ambos PPO abandonan la
  fase 1 en cuanto se lo permiten**. No se investigó a fondo.
- **Fracción de tiempo en fase 1: señal PARCIAL, solo en el PPO del sueño.**
  Catastróficos 29.9% frente a 25.8% en los buenos; correlación con el reward en los 30
  episodios de -0.59. En el PPO directo no hay señal: 23.3% frente a 23.1%, correlación
  -0.05.
- **Escala de la función de valor (`explained_variance` ≈ 0 en AMBOS entrenamientos):
  confirmada en magnitud, NO en causalidad.** Los retornos no están normalizados y son
  de cientos a miles: en el dataset, el retorno por episodio tiene desviación estándar
  1488.56 y el retorno descontado (γ=0.99) 1046.06. En los dos entrenamientos,
  `value_loss` es del orden de la varianza real de los retornos:
  - RL directo: `value_loss` ~1e4–4e4 al final, con varianza de sus retornos de 1.9e4
    (desviación estándar 139.09).
  - Sueño: `value_loss` entre 5,220 y 45,000 (mediana 10,100), con varianza de los
    retornos imaginados de 5.2e3 (desviación estándar 72.27).

  Esto es lo esperable si la red de valor predice aproximadamente la media. No se
  implementó `VecNormalize` ni otra normalización para comprobar si eso arregla la
  función de valor o elimina los episodios catastróficos.

**Conclusión honesta: la causa completa de los episodios catastróficos NO se
identificó.** Queda documentada como limitación abierta, no como resuelta.

### 6. Trabajo futuro identificado, en orden de prioridad

1. **Normalizar recompensas o retornos (por ejemplo `VecNormalize`) en los dos
   entrenamientos de PPO**, y volver a evaluar si `explained_variance` sube y si
   desaparecen los episodios catastróficos. Es el experimento más prometedor y mejor
   fundamentado de los que quedaron pendientes.
2. **Si (1) no lo explica todo: investigar el momento de los cambios de fase** respecto a
   las colas de cada brazo (cola del brazo que pasa a verde frente a la del que pasa a
   rojo en cada cambio), comparando episodios catastróficos con buenos.

### 7. Nota sobre el costo de interacciones reales (sin conclusión firme)

El RL directo llegó a su mejor evaluación (-284.00, semillas 20000–20004) en **6,000
pasos reales de entrenamiento más 1,800 de evaluación**. El método World Model usó
**2,400 transiciones reales** para el dataset. Es el mismo orden de magnitud que en el
escenario simétrico. **Esta comparación es prematura mientras los episodios catastróficos
no se entiendan**: podrían estar distorsionando cualquiera de los dos promedios, y
ninguno de los dos métodos produce todavía una política estable.

## ✅ Baseline de RL directo y hallazgo final: ambos métodos convergen a la misma regla, sin ahorro de interacciones demostrable en este escenario

### 1. Qué se construyó

- **`environments/reseeding_wrapper.py`** (`ReseedingWrapper`). sumo-rl solo cambia la
  semilla de tráfico cuando `reset()` la recibe explícitamente; si no, reutiliza la
  última (`SumoEnvironment.reset`). Stable-Baselines3 llama a `reset()` sin semilla en
  cada frontera de episodio, así que sin el wrapper el PPO directo habría entrenado y
  se habría evaluado siempre sobre la misma realización de tráfico. El wrapper asigna
  una semilla nueva en cada episodio de entrenamiento (10000, 10001, …) y un conjunto
  fijo que cicla en `EvalCallback` (20000–20004); una semilla explícita siempre gana.
  Verificado con un registro de cada `reset()` real:
  - Entrenamiento: 171 reinicios, 171 semillas únicas. La primera es 0 (la pasa SB3
    explícitamente por `PPO(seed=0)`), después 10000–10169 consecutivas.
  - `EvalCallback`: 60 reinicios. Cada una de las 10 evaluaciones cubre exactamente una
    vez las 5 semillas, rotadas una posición porque `DummyVecEnv` reinicia
    automáticamente tras el quinto episodio y consume una semilla que se descarta. Esto
    solo funciona porque `n_eval_episodes` (5) coincide con la longitud del ciclo.
  - 3 tests nuevos (`tests/test_reseeding_wrapper.py`), sin SUMO.
- **`training/train_controller_direct.py`**: PPO (`MlpPolicy`, mismos hiperparámetros de
  `ControllerConfig`) entrenado directamente sobre el estado crudo de 26 dimensiones de
  `TrafficEnvironment`, sin Autoencoder ni Dream Environment. Presupuesto deliberadamente
  modesto: 10,000 timesteps (SB3 completa hasta 10,240, múltiplo de `n_steps`), más
  10 evaluaciones × 5 episodios × 60 pasos = 3,000 pasos de evaluación. Duró 262 s. Su
  `.json` de hiperparámetros registra `total_timesteps=10000`; `dream_max_steps` aparece
  porque es un campo de `ControllerConfig`, pero no tiene efecto en esta corrida.
- **`scripts/evaluate_direct_vs_dream.py`**: v1, PPO directo y tiempo fijo contra SUMO
  real, mismo protocolo de siempre (15 episodios, `seed_base=3000` y `5000`, semillas
  distintas de las de entrenamiento y `EvalCallback`).

### 2. Primera señal de alarma

- **PPO directo y v1 dan un reward casi idéntico episodio por episodio**: 22 de 30
  episodios difieren en menos de 1 punto (por ejemplo -294.0/-293.7, -257.0/-257.0,
  -261.1/-261.1), a pesar de haber aprendido en espacios completamente distintos (`z`
  imaginado frente a estado crudo real).
- **`explained_variance` se mantuvo entre -0.002 y 0.0003 durante todo el
  entrenamiento directo**: la función de valor no aprendió nada.
- **La curva de `EvalCallback` se estanca desde el timestep 2000**:

  ```
  timestep | reward (5 episodios, semillas 20000-20004)
     1000  | -549.12 +/- 287.15
     2000  | -285.20 +/-  47.56
     3000  | -285.06 +/-  35.52
     4000  | -285.14 +/-  35.51
     5000  | -285.18 +/-  35.50
     6000  | -285.02 +/-  35.59
     7000  | -285.02 +/-  35.55
     8000  | -286.08 +/-  35.13
     9000  | -284.34 +/-  35.60
    10000  | -284.10 +/-  35.58   <- best_model.zip
  ```

  Después del timestep 2000 las mejoras son de décimas. El reward del rollout
  (política estocástica, media de los últimos 100 episodios) baja de -2020 a -330.

### 3. Investigación: dos pruebas

**(a) Reglas triviales, evaluadas de forma independiente** (su propio
`TrafficEnvironment`, mismo `run_policy` del script de evaluación):

```
                                          |             reward |    espera_prom |    cola_prom |   throughput
"Pedir siempre la fase contraria", 3000   |  -287.83 +/-  23.87 |  3.82 +/- 0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
"Pedir siempre la fase contraria", 5000   |  -293.43 +/-  41.28 |  3.90 +/- 0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
"Alternar 0/1 cada paso", 3000            |  -733.53 +/-  64.83 | 10.50 +/- 0.93 | 1.90 +/- 0.15 | 13.40 +/- 2.47
"Alternar 0/1 cada paso", 5000            |  -748.60 +/-  55.87 | 10.69 +/- 0.81 | 1.96 +/- 0.13 | 13.47 +/- 3.10
```

"Pedir siempre la fase contraria" (acción = `1 - green_phase`, es decir, cambiar en
cuanto `min_green` lo permite) **reproduce a v1 casi exacto**: la misma espera, cola y
throughput a dos decimales, la misma desviación del reward, y un reward por episodio
0.0–0.1 peor que v1. "Alternar 0/1 cada paso" es mucho peor que tiempo fijo, porque la
mitad de sus peticiones caen en pasos bloqueados por `min_green`.

**(b) Comparación contrafactual sobre las trayectorias reales de v1** (v1 conduce SUMO
en las semillas 3000–3014 y 5000–5014; en cada estado real se pregunta qué elegirían
v1, el PPO directo y la regla, sin ejecutar nada más). Los 30 rewards de v1 reproducen
sus valores oficiales. Un paso está "bloqueado" cuando
`time_since_last_phase_change < yellow_time + min_green`: ahí ninguna acción cambia el
tráfico, solo la penalización de 0.1 (ver nota técnica, punto 7).

```
1800 pasos, 930 bloqueados por min_green (51.7%; 31 de 60 en cada episodio)

Par                   | acuerdo total | desacuerdos | en pasos bloqueados | acuerdo en pasos NO bloqueados
v1 / PPO directo      |     78.7%     |     383     |     376 (98.2%)     |   99.2%  (7 de 870)
v1 / regla            |     97.5%     |      45     |      45 (100%)      |  100.0%  (0 de 870)
PPO directo / regla   |     79.8%     |     364     |     357 (98.1%)     |   99.2%  (7 de 870)
```

En los pasos donde la acción sí afecta al tráfico, **v1 coincide con la regla en 870 de
870** y el PPO directo en 863 de 870. Los desacuerdos restantes caen casi enteramente en
pasos bloqueados, donde la acción no tiene efecto real y solo mueve la penalización de
0.1.

### 4. Conclusión honesta

**Ninguno de los dos PPO aprendió control dependiente del estado del tráfico.** Ambos
redescubrieron la misma regla simple, "pedir siempre la fase contraria", que es la mejor
política encontrada para este escenario concreto (demanda baja y simétrica, dos fases).
No está demostrado que sea óptima: supera a tiempo fijo y a "alternar 0/1", y dos
optimizadores independientes convergieron a ella, lo que la señala como la mejor
estrategia alcanzable aquí, pero no es una prueba de optimalidad.

**En este escenario no hay ahorro demostrable de interacciones reales con SUMO.**
Contando todo lo que cada método necesitó de SUMO real para llegar a esa regla:

| Método | Pasos reales de SUMO hasta la regla |
|---|---|
| World Model (v1) | **2,400** (dataset de 40 episodios × 60 pasos con acciones aleatorias, usado para entrenar y seleccionar el Autoencoder y el LSTM, sembrar el Dream Environment y validar el PPO) + 0 durante el entrenamiento del PPO |
| RL directo | **≤2,000** de entrenamiento + 600 de evaluación = **≤2,600** (primera evaluación en el timestep 2000 ya en -285.20; como se evaluaba cada 1000 pasos, 2000 es una cota superior) |

Los dos costos son del mismo orden de magnitud, y el del RL directo es una cota superior.
El PPO v1 sí entrenó con cero pasos reales, pero el método World Model completo no: su
costo en SUMO está en la recolección del dataset.

La pregunta de investigación ("¿puede un modelo aprendido de la dinámica reducir las
interacciones necesarias con SUMO sin perder desempeño?") se responde así para este
escenario: **ambos métodos alcanzan el mismo desempeño con un costo de interacción
comparable; no hay reducción demostrable.** Además, la comparación sostiene conclusiones
sobre el costo para llegar a la regla, no sobre la calidad de control, porque ninguno de
los dos aprendió control. El ahorro de interacciones que predice la propuesta solo sería
demostrable en un escenario donde el RL directo necesitara **sustancialmente más de
2,400 pasos reales** para converger. Este escenario, por ser demasiado simple, nunca lo
exige (ver punto 6).

### 5. Tabla final (SUMO real, 15 episodios por semilla, mismo protocolo)

```
                                       |             reward |     espera_prom |     cola_prom |    throughput
PPO v1 (World Model), seed_base=3000   |  -287.76 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
PPO v1 (World Model), seed_base=5000   |  -293.35 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
PPO RL directo, seed_base=3000         |  -293.57 +/-  37.21 |  3.91 +/-  0.49 | 1.16 +/- 0.12 | 13.47 +/- 2.31
PPO RL directo, seed_base=5000         |  -297.63 +/-  38.60 |  3.96 +/-  0.52 | 1.18 +/- 0.10 | 13.67 +/- 2.65
Tiempo fijo (ciclo=5), seed_base=3000  |  -570.27 +/-  86.70 |  8.05 +/-  1.25 | 1.66 +/- 0.21 | 13.67 +/- 2.55
Tiempo fijo (ciclo=5), seed_base=5000  |  -605.27 +/- 123.47 |  8.57 +/-  1.75 | 1.73 +/- 0.28 | 13.87 +/- 2.63
Regla "fase contraria", seed_base=3000 |  -287.83 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
Regla "fase contraria", seed_base=5000 |  -293.43 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
```

La regla "fase contraria" es la referencia del techo alcanzado en este escenario (la
mejor política encontrada, sin aprendizaje). Frente a ella, v1 empata y el PPO directo
queda levemente por debajo: su peor episodio es -403.6 (semilla 3009), frente a -314.1
de la regla en la misma semilla. Las ventajas de v1 sobre tiempo fijo documentadas en
la sección anterior (~50% menos espera, ~31% menos cola) siguen siendo ciertas
numéricamente, pero son ventajas de la regla, no de control aprendido.

### 6. Limitación más importante del proyecto (trabajo futuro)

El escenario actual (demanda baja y simétrica: 200/90/60 vehículos por hora por acceso
para recto/izquierda/derecha, dos fases) es **demasiado simple para que el control
dependiente del estado aporte ventaja sobre una regla fija**: cambiar de fase en cuanto
se pueda ya alcanza el mejor resultado encontrado. Un escenario con **demanda asimétrica
o variable en el tiempo** es necesario para que la comparación distinga los métodos por
calidad de control, y no solo por costo de aprendizaje. También es la condición para
poder medir el ahorro de interacciones que predice la propuesta: requiere un problema
donde el RL directo necesite sustancialmente más interacciones reales que las 2,400 del
dataset del World Model.

### 7. Nota técnica adicional (sin corregir)

`ProjectRewardFunction` resta `delta × phase_change` (`delta=0.1`), y
`TrafficEnvironment.step()` define `info["phase_change"] = float(action == 1)`. Es decir,
se penaliza **pedir la fase 1**, no cambiar efectivamente de fase. Tiene la misma raíz
que la confusión de semántica de acción ya documentada (la acción es el índice de fase
verde destino, no "mantener/cambiar"). Es la única señal que distingue entre sí las
acciones en pasos bloqueados por `min_green`, y es la explicación más probable (no
verificada paso a paso) de las diferencias de 0.0–0.1 por episodio entre v1 y la regla.

## ✅ Controlador PPO contra SUMO real: historia completa y resultado oficial (v1)

> **Actualización posterior**: el bloque del baseline de RL directo (sección de arriba)
> mostró que v1 equivale en la práctica a la regla "pedir siempre la fase contraria"
> (100% de acuerdo en los pasos donde la acción afecta al tráfico). Los números de esta
> sección siguen siendo correctos, pero describen esa regla, no control aprendido
> dependiente del estado.

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
4. `ProjectRewardFunction.phase_change` penaliza pedir la fase 1, no cambiar
   efectivamente de fase (misma raíz que el punto 3; ver la sección del baseline de RL
   directo, punto 7).
5. `scripts/collect_dataset.py` no pasa semilla a `env.reset()`: los 40 episodios del
   dataset comparten la semilla de SUMO 42, y la variedad viene solo de las acciones
   aleatorias. Decidir si esto limita al World Model.
6. Ambos PPO (escenario asimétrico) nunca sostienen el verde de la fase 1 más de 8 s,
   que coincide con la duración mínima posible de una fase. No investigado a fondo (ver
   la sección del escenario asimétrico, punto 5).
7. **Episodios catastróficos de los PPO en el escenario asimétrico: causa no identificada
   por completo** (limitación abierta; ver la misma sección, puntos 5 y 6).

## ⚪ No implementado todavía

- **Normalización de recompensas o retornos (`VecNormalize` u otra) en ambos
  entrenamientos de PPO**, para comprobar si arregla la función de valor
  (`explained_variance` ≈ 0) y elimina los episodios catastróficos. Prioridad 1 del
  trabajo futuro del escenario asimétrico.
- Análisis del momento de los cambios de fase respecto a las colas de cada brazo, si la
  normalización no explica todo.
- Demanda variable en el tiempo (el escenario asimétrico ya está implementado; la
  variación temporal no).
- Transformer y TSMixer como sustitutos del LSTM (extensiones opcionales).

## Qué se estaba haciendo justo antes de este handoff

Se cambió la demanda a un escenario asimétrico (500/150 veh/h; la propuesta de 700/150
se descartó por saturar la vía principal) y se re-ejecutó el pipeline completo desde cero
(commits `febef8e`, `6c753d2`, `f58347e`, 39/39 tests en verde). La regla trivial "pedir
siempre la fase contraria" pasa a ser la peor política, y los dos PPO ya no la siguen
(54.9% y 47.2% de acuerdo en pasos no bloqueados). Ambos superan en promedio a tiempo fijo
(21/30 episodios cada uno), pero tienen episodios catastróficos (7/30 y 5/30 peores que
-600) que tiempo fijo no tiene. Se descartaron la cola invisible y las rachas largas de
fase 1 como causa; la escala sin normalizar de la función de valor se confirmó en
magnitud pero no en causalidad. Siguiente paso recomendado: normalizar recompensas o
retornos (`VecNormalize`) en los dos entrenamientos de PPO y volver a evaluar.