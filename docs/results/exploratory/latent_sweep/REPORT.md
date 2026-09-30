# Informe: barrido exploratorio de `latent_dim`

**Exploratorio y personal.** No reemplaza el Experimento 0 oficial ni se cita en la documentación
principal salvo decisión explícita del autor. Todo lo que se fijó de antemano está en `MANIFEST.md`
(commit `b5e1049`, anterior a cualquier corrida) y no se cambió.

## 1. Paso 1: dimensiones constantes (80 episodios, 4,800 estados sin normalizar)

- One-hot de fase (índices 20–23), activaciones en `states`: 2,464 / 2,336 / **0** / **0**. En
  `next_states`: 2,426 / 2,374 / 0 / 0. Las posiciones 22 y 23 tienen varianza exactamente cero.
- Tiempo restante de fase (índice 25): 0 en **4,720 de 4,800** estados (98.33%). Los 80 restantes
  valen 5 y son el paso 0 de cada episodio. En `next_states`, 0 en los 4,800.
- Las posiciones 20 y 21 son complementarias: juntas aportan un solo bit.
- La premisa se confirmó. Estado de 24 dimensiones: el oficial normalizado sin las columnas 22 y 23
  (`VALIDATION.md`).

## 2. Paso 3: barrido (semillas 0–2; rama cruda de 24 compartida)

Reducción por semilla = mediana sobre h = 1..10 de (crudo − z)/crudo del `reward_mse` en test.
Autoencoder con `hidden_dim = latent_dim`, uno por valor (semilla 0).

| `latent_dim` | Pérdida de reconstrucción (val) | Semilla 0 | Semilla 1 | Semilla 2 | Mediana | Pares ganados |
|---|---|---|---|---|---|---|
| 4 | 0.2529 | -100.8% (0/10) | +11.2% (6/10) | -13.6% (3/10) | -13.6% | 9/30 |
| 8 | 0.0916 | +0.2% (5/10) | +39.5% (9/10) | +25.4% (10/10) | +25.4% | 24/30 |
| **12** | 0.0418 | +0.9% (6/10) | +45.3% (9/10) | +38.2% (10/10) | **+38.2%** | **25/30** |
| 16 | 0.0128 | -31.9% (0/10) | +27.5% (8/10) | +32.0% (10/10) | +27.5% | 18/30 |
| 20 | 0.0047 | -26.1% (1/10) | +45.1% (10/10) | +28.1% (9/10) | +28.1% | 20/30 |

Las 18 LSTM se cortaron por early stopping (épocas 74–194); ninguna llegó al tope de 300. No hubo
fallos técnicos.

## 3. Ganador

**`latent_dim` = 12**, por el criterio fijado: la mayor mediana, sobre las 3 semillas, de la reducción
por semilla (+38.2%). No hizo falta desempate.

## 4. Paso 4: `latent_dim` = 12 con 10 semillas por rama

| Semilla | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Reducción | +0.9% | +45.3% | +38.2% | +26.9% | +14.5% | +20.1% | +19.7% | +1.2% | +42.5% | +0.7% |
| Horizontes ganados por z | 6 | 9 | 10 | 10 | 10 | 10 | 10 | 6 | 9 | 5 |

| | z12 sobre 24 dims, 10 semillas | z12, solo las semillas nuevas 3–9 | z16 oficial sobre 26 dims, 10 semillas (exploración anterior) |
|---|---|---|---|
| Pares semilla × horizonte ganados | **85/100** | 60/70 | 60/100 |
| Mediana de las medianas por horizonte | +21.1% | — | +8.4% |
| Mediana de las reducciones por semilla | +19.9% | +19.7% | +8.6% |
| Semillas positivas | **10/10** | **7/7** | 6/10 |
| Prueba de signo (exacta, bilateral) | **p = 0.002** | **p = 0.016** | p = 0.75 |
| Wilcoxon (exacto, bilateral) | **p = 0.002** | **p = 0.016** | p = 0.19 |

- Las semillas 0–2 del ganador sirvieron para elegirlo, así que las 10 semillas llevan sesgo de
  selección. Las 7 nuevas (3–9), libres de ese sesgo, dan el mismo resultado: 7/7 positivas.
- Las 14 corridas del Paso 4 se cortaron por early stopping (z12: épocas 90–208; crudo 24: 95–163).
- "26 sin comprimir" no tiene un número aparte: es la rama cruda de cada comparación.

## 5. Calibración y control *post hoc*

**Crudo 24 frente a crudo 26, semillas 0–4 (información fijada en el MANIFEST).** Las entradas solo
difieren en dos columnas siempre en cero. Reducción por semilla: +7.4%, -48.7%, -29.0%, +2.6% y
+4.0%. Quitar dos columnas muertas cambia la inicialización y, con ella, el resultado de una semilla
en hasta 49 puntos. Las semillas 1 y 2 de crudo 24 salieron especialmente malas, y como el Paso 3
comparte esa rama entre todos los `latent_dim`, **infla por igual las semillas 1 y 2 de todos los
valores del barrido**. El barrido se distingue sobre todo por la semilla 0.

**Control *post hoc*, no preregistrado.** Como el emparejamiento por semilla es nominal, se compararon
los grupos de 10 corridas sin emparejar. Métrica: media geométrica del `reward_mse` sobre h = 1..10;
prueba de permutación exacta (`posthoc_unpaired.py`).

| Comparación | Diferencia | p |
|---|---|---|
| z12 frente a crudo 24 | -21.8% | 0.001 |
| z12 frente a crudo 26 (las 10 corridas de la exploración anterior) | -21.7% | 0.0002 |
| crudo 24 frente a crudo 26 | 0.0% | 0.997 |
| z16 oficial frente a crudo 26 | -6.7% | 0.27 |
| z12 frente a z16 oficial | -16.1% | 0.008 |

A nivel de grupo, la ventaja de z12 no depende de la rama cruda de 24 (frente a la de 26 es igual) ni
del emparejamiento. Por horizonte, z12 empata en h = 1 (55.3 frente a 54.9) y la ventaja crece con el
horizonte (h = 10: 554.5 frente a 691.8).

## 6. Interpretación

- **Sí: con `latent_dim` = 12 el Autoencoder muestra una ventaja distinguible de cero.** Es 10/10 en
  el análisis pareado, 7/7 en las semillas que no participaron en la selección, y el control sin
  emparejar da p ≤ 0.001 frente a las dos ramas crudas. Con `latent_dim` = 16 sobre 26 dimensiones
  (la configuración oficial) no se distinguía (p = 0.75 y 0.19; sin emparejar, p = 0.27).
- **Tendencia del barrido:** comprimir demasiado perjudica claramente (k = 4: reconstrucción con
  pérdida de 0.25; peor que el crudo). Entre 8 y 20, las semillas 1 y 2 no discriminan (todas
  ganan, infladas por la rama cruda compartida). La semilla 0 sí: 8 y 12 quedan en torno a cero, y
  16 y 20 en -26% a -32%. El patrón compatible con eso es un óptimo intermedio: comprimir de forma
  moderada (8–12) ayuda más que comprimir poco (16–20) o mucho (4). **Con 3 semillas por valor, esa
  forma es solo una indicación; no está establecida.** Solo k = 12 se confirmó con más semillas.
- **Salvedades:**
  - Cada `latent_dim` usa **un único Autoencoder** (semilla 0): no se midió cuánto varía el
    resultado con la semilla del Autoencoder. "k = 12 es mejor" no se separa de "este Autoencoder
    de k = 12 es bueno". Lo mismo vale para el Experimento 0 oficial.
  - El ganador se eligió entre 5 valores. Las semillas 3–9 descartan que la ventaja sea
    solo el efecto de la selección, pero el tamaño con 10 semillas (+19.9%) puede estar algo
    inflado. El de las 7 nuevas es +19.7%.
  - Solo se mide `reward_mse` en rollouts del modelo temporal. No se sabe si un World Model con k = 12
    daría mejor control en SUMO.
  - Eliminar las 2 columnas constantes no tiene efecto a nivel de grupo (crudo 24 ≈ crudo 26). La
    diferencia viene del tamaño del cuello de botella, no de quitar dimensiones muertas.

## 7. Integridad

- Los 77 md5 de `official_md5_before.txt` coinciden al terminar: `datasets/processed/*`, Autoencoder
  y LSTM oficiales, `exp0_multiseed_300ep/`, `experiment_0_multiseed_300ep.json` y todos los `.md`
  versionados.
- `git diff main` fuera de `docs/results/exploratory/` está vacío. `main` no se tocó.

## 8. Cómputo

- 5 Autoencoders: 498 s de proceso.
- 32 LSTM (18 del barrido y 14 del refuerzo) más 2 de validación: unas 4.4 h de proceso en total.
  Bajo presión de memoria, las corridas iban 3–5 veces más lentas (unos 1,300 s frente a unos 300 s).
- Tiempo de pared: unas 2.5 h en total. La validación tomó unos 12 min; el primer lanzamiento del
  barrido, detenido por falta de memoria, unos 30 min; el relanzamiento con 3 en paralelo, 29 min; el
  Paso 4, 30 min; las comparaciones, unos 10 min.
- Incidencias: el sistema detuvo dos veces la tarea padre por falta de memoria. Los scripts siguieron
  corriendo huérfanos. La primera vez se detuvieron por instrucción del autor y se relanzaron con 3
  en paralelo, conservando las 3 corridas ya completas (el entrenamiento es determinista). La segunda
  vez se dejaron terminar. Ninguna corrida se repitió ni se excluyó por su resultado.

## 9. Addendum: `latent_dim` = 8 reforzado a 10 semillas (pedido del autor; criterio en `ADDENDUM_z8.md`)

Mismo procedimiento que para 12 (`run_reinforce.sh 8`): 7 corridas nuevas de z8 (semillas 3–9),
todas cortadas por early stopping (épocas 61–137). La rama cruda de 24 con semillas 3–9 es la misma
que la del refuerzo de 12 (se reutilizó; el script lo verifica).

| Semilla | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| z8: reducción | +0.2% | +39.5% | +25.4% | +11.8% | -1.6% | +0.2% | -17.5% | +14.0% | +36.7% | -8.5% |
| z8: horizontes ganados | 5 | 9 | 10 | 9 | 4 | 5 | 0 | 7 | 9 | 2 |
| z12: reducción | +0.9% | +45.3% | +38.2% | +26.9% | +14.5% | +20.1% | +19.7% | +1.2% | +42.5% | +0.7% |

| | z8, 10 semillas | z8, solo 3–9 | z12, 10 semillas | z12, solo 3–9 |
|---|---|---|---|---|
| Pares ganados | 60/100 | 35/70 | 85/100 | 60/70 |
| Semillas positivas | 7/10 | 4/7 | 10/10 | 7/7 |
| Prueba de signo | p = 0.34 | p = 1.00 | p = 0.002 | p = 0.016 |
| Wilcoxon | p = 0.19 | p = 0.69 | p = 0.002 | p = 0.016 |
| Reducción mediana por semilla | +6.0% | +0.2% | +19.9% | +19.7% |
| Rango de las reducciones por semilla | -17.5% a +39.5% | -17.5% a +36.7% | +0.7% a +45.3% | +0.7% a +42.5% |

**Sin emparejar** (media geométrica del `reward_mse` sobre h = 1..10; permutación exacta):

| Comparación | Diferencia | p | Estado |
|---|---|---|---|
| **z8 frente a z12** | **+13.3%** (z8 peor) | **0.014** | preregistrada (`ADDENDUM_z8.md`) |
| z8 frente a crudo 24 | -11.4% | 0.090 | *post hoc* |
| z8 frente a crudo 26 | -11.3% | 0.022 | *post hoc* |
| z12 frente a crudo 24 / crudo 26 | -21.8% / -21.7% | 0.001 / 0.0002 | *post hoc* |

Mediana por horizonte (10 corridas): z8 es peor que z12 en h = 1..9 (h = 1: 64.6 frente a 55.3,
peor incluso que el crudo, 54.9) y mejor solo en h = 10 (497.1 frente a 554.5).

**Interpretación.**

- **El efecto de z8 se diluye con semillas nuevas, como el de z16:** en las semillas 3–9 queda en
  +0.2% de mediana y 4/7 positivas. Sus semillas 1 y 2 del barrido (+39.5% y +25.4%) coincidían con
  las dos semillas malas de la rama cruda de 24 (sección 5).
- **Con el criterio preregistrado, 8 y 12 se distinguen** (p = 0.014): 12 tiene un error un 13%
  menor. Los rangos por semilla se traslapan mucho (z8 de -17.5% a +39.5%; z12 de +0.7% a +45.3%),
  pero la diferencia está en la cola inferior: 4 de 10 semillas de z8 son negativas y ninguna de z12
  lo es. Los datos **no** apoyan una meseta de valores igualmente buenos entre 8 y 12.
- **Salvedad principal:** cada `latent_dim` tiene un único Autoencoder (semilla 0). Las 10 semillas
  replican el ruido de la LSTM, no el del Autoencoder. "12 es mejor que 8" puede ser en parte "este
  Autoencoder de 12 es mejor que este de 8". Separarlo exigiría varias semillas del Autoencoder por
  valor.
- Con z8, z16 y el z16 oficial diluyéndose, **z12 queda como el único resultado positivo robusto del
  barrido**. Que sea un óptimo real en k = 12, o una buena inicialización de un Autoencoder concreto,
  no se puede decidir con estos datos.

Cómputo del addendum: 7 LSTM, 2,073 s de proceso y 736 s de pared.

## 10. Addendum: ¿depende la ventaja de `latent_dim` = 12 del Autoencoder concreto? (criterio en `ADDENDUM_ae_seeds.md`)

Autoencoders de `latent_dim` = 12 con semillas 0 (`ae12/`, reutilizado), 1 y 2, cada uno con la LSTM
de semillas 0–2 (6 corridas nuevas; todas cortadas por early stopping, épocas 128–167). Se verificó
antes que `train-ae --seed 0` reproduce `ae12/` tensor a tensor. La rama cruda es `raw24/seed{0,1,2}`.

**Tabla principal: reducción mediana (crudo − z)/crudo**

| | LSTM s0 | LSTM s1 | LSTM s2 | Media de fila | Reconstrucción (val) |
|---|---|---|---|---|---|
| AE s0 | +0.9% | +45.3% | +38.2% | **+28.1%** | 0.0418 |
| AE s1 | -32.6% | +37.6% | +23.5% | +9.5% | 0.0321 |
| AE s2 | -28.0% | +40.8% | +14.1% | +9.0% | 0.0312 |
| Media de columna | -19.9% | +41.2% | +25.2% | | |

| Descomposición | Métrica principal (reducción) | Métrica secundaria (solo z: 100 × log de la media geométrica del `reward_mse`) |
|---|---|---|
| Varianza entre AE (con la LSTM fija), media de 3 columnas | 164.4 | 185.9 |
| Varianza entre semillas de LSTM (con el AE fijo), media de 3 filas | 1,050.0 | 71.7 |
| ANOVA: σ²_AE / σ²_LSTM (desviación) | 96.9 (9.8) / 982.6 (31.3) | 129.7 (11.4) / 15.6 (3.9) |
| F del AE / F de la LSTM (2 y 4 gl) | 5.3 / 44.7 | 7.9 / 1.8 |
| ¿AE s0 atípico? (criterio fijado) | **sí**: +28.1 frente a +9.2 de las otras dos; umbral 11.6 | **sí**: 543.8 frente a 564.5; umbral 10.6 |

Varianza entre semillas de LSTM con el AE s0 y 10 semillas: 293.0 (desviación de 17.1 puntos).

**Lectura.**

- **En la métrica principal domina la semilla de la LSTM**, pero casi toda esa varianza es ruido de
  la **rama cruda**, no de la LSTM sobre z. La columna s comparte `raw24/seed s`, cuya media
  geométrica vale 229.7, 392.8 y 354.3 para s = 0, 1, 2. Por eso la columna 0 es negativa en las
  tres filas y la 1 positiva en las tres.
- **En la métrica que solo depende de z, domina el Autoencoder:** su desviación es del 11.4% frente
  al 3.9% de la LSTM. Con la misma LSTM (mismos pesos iniciales y mismo orden de lotes, porque las
  tres entradas tienen 12 dimensiones), cambiar el Autoencoder mueve el error más que cambiar la
  semilla de la LSTM.
- **El Autoencoder de semilla 0 es atípico, y en la dirección favorable**, según el criterio fijado
  y en las dos métricas. Sus LSTM tienen un error de recompensa unos 19% menor que las de los
  Autoencoders 1 y 2 (medias geométricas por corrida de 224–234, frente a 247–322). Los otros dos
  coinciden entre sí (566.9 y 562.2). Además, el Autoencoder 0 es el que **peor reconstruye**
  (0.042 frente a 0.032 y 0.031): la calidad de reconstrucción no predice la utilidad para la
  dinámica.
- **Descriptivo, *post hoc*, sin prueba:** con los Autoencoders 1 y 2, la media geométrica de sus 6
  corridas (247–322) cae dentro de la de las 10 corridas crudas de 24 (mediana 291.3, rango
  229.7–395.3) y de 26 (mediana 299.4), con una mediana de unos 289. **Con esos Autoencoders, z12
  no se distingue del estado crudo.**

**Conclusión.** La ventaja significativa de `latent_dim` = 12 (secciones 4 y 9) **no es robusta al
Autoencoder: depende en gran parte de esa lotería.** Las pruebas de las secciones 4 y 9 son
válidas para el Autoencoder 0, pero **no generalizan a `latent_dim` = 12 como valor.** Con 2 de 3
Autoencoders el efecto desaparece, y el que lo produce es precisamente el que se usó en todo el
barrido. Eso debilita también el orden del barrido (sección 2), porque cada `latent_dim` usó un único
Autoencoder de semilla 0: la comparación entre valores confunde el tamaño del cuello de botella con
la semilla del Autoencoder. Con solo 3 Autoencoders no se puede estimar si el 0 es "afortunado" o
los otros dos "típicos", más allá de que el 0 se aparte y los otros dos coincidan. Una conclusión
sobre `latent_dim` en general exigiría varias semillas de Autoencoder por valor, y quizás un
criterio para elegir el Autoencoder que no sea su pérdida de reconstrucción, que aquí no predijo
nada.

Cómputo del addendum: 3 Autoencoders (incluida la comprobación de la semilla 0) y 6 LSTM, unos
14 min de pared y unos 2,240 s de proceso en las LSTM.

## 11. Addendum: grilla de 5×5 (semilla del AE × semilla de la LSTM) para `latent_dim` = 12 (criterio en `ADDENDUM_grid5.md`)

Autoencoders con semillas 0–4 (los 3 y 4, nuevos) y LSTM con semillas 0–4 por Autoencoder. Se
entrenaron 2 Autoencoders y 14 LSTM nuevas (no 16: las semillas 3 y 4 del AE 0 ya existían). Las 14
se cortaron por early stopping (épocas 96–153). Rama cruda: `raw24/seed0..9`, sin reentrenar.

**Métrica principal (descriptiva): reducción mediana (crudo − z)/crudo contra `raw24/seed s`**

| | LSTM s0 | LSTM s1 | LSTM s2 | LSTM s3 | LSTM s4 | Media de fila |
|---|---|---|---|---|---|---|
| AE s0 | +0.9% | +45.3% | +38.2% | +26.9% | +14.5% | +25.1% |
| AE s1 | -32.6% | +37.6% | +23.5% | +25.2% | +8.8% | +12.5% |
| AE s2 | -28.0% | +40.8% | +14.1% | -6.3% | +1.3% | +4.4% |
| AE s3 | -9.6% | +35.4% | +37.0% | +12.7% | -2.5% | +14.6% |
| AE s4 | -112.9% | +18.8% | -27.3% | -4.6% | -11.0% | -27.4% |
| Media de columna | -36.4% | +35.6% | +17.1% | +10.8% | +2.2% | |

16 de 25 celdas son positivas.

**Métrica de decisión: media geométrica del `reward_mse` de z sobre h = 1..10 (menor es mejor)**

| | LSTM s0 | LSTM s1 | LSTM s2 | LSTM s3 | LSTM s4 | Media geométrica del AE | Reconstrucción (val) |
|---|---|---|---|---|---|---|---|
| AE s0 | 224.1 | 234.3 | 231.7 | 205.8 | 257.7 | **230.1** | 0.0418 |
| AE s1 | 322.0 | 260.2 | 289.9 | 213.5 | 281.6 | 271.0 | 0.0321 |
| AE s2 | 288.7 | 247.0 | 296.1 | 301.8 | 290.9 | 284.2 | 0.0312 |
| AE s3 | 258.3 | 269.9 | 235.7 | 250.0 | 299.6 | 261.8 | 0.0428 |
| AE s4 | 446.7 | 343.8 | 489.7 | 268.9 | 328.7 | 366.9 | 0.0452 |

Crudo de 24 dimensiones (10 corridas): media geométrica 300.9.

**ANOVA de dos factores sin réplica (5 × 5; F con 4 y 16 gl)**

| Métrica | Autoencoder | Semilla de LSTM | Residuo |
|---|---|---|---|
| Decisión (log de la media geométrica de z) | F = 8.61, **p = 0.0007**; σ_AE = 0.161 (~16%) | F = 2.07, p = 0.13; σ_LSTM = 0.060 (~6%) | σ = 0.131 |
| Principal (reducción; descriptiva) | F = 6.80, p = 0.002; σ_AE = 18.5 puntos | F = 12.06, p = 0.0001; σ_LSTM = 25.5 puntos | σ = 17.1 puntos |

**Comparación contra el crudo (criterio fijado)**

- Δ = media de g en z (25 corridas) − media de g en el crudo de 24 (10 corridas) = −0.074, es decir,
  z tiene un 7.1% menos de `reward_mse` geométrico.
- **IC del 95% por bootstrap por conglomerados (B = 100,000): de −21.7% a +12.0%. Incluye 0.**
- Permutación por corrida (no decisoria, anticonservadora): p = 0.33.
- **Conclusión según el criterio fijado: SIN EVIDENCIA de que `latent_dim` = 12 prediga la
  recompensa mejor (ni peor) que el estado crudo.**

**Lectura.**

- **La semilla del Autoencoder es la fuente de varianza dominante y significativa** en la métrica
  que solo depende de z: σ_AE ≈ 16% frente a σ_LSTM ≈ 6%, con p = 0.0007 frente a p = 0.13. Con 5
  niveles se confirma lo que el diseño de 3×3 sugería. En la métrica de reducción domina la columna,
  pero eso es ruido de la rama cruda compartida (sección 10), no de la LSTM sobre z.
- **El Autoencoder de semilla 0 es el mejor de los cinco** (230.1, frente a 262–367). Todo el
  resultado significativo anterior (secciones 4 y 9) dependía de él. El Autoencoder 4 es el peor por
  mucho (366.9, peor que el crudo). La pérdida de reconstrucción no ordena esto: el AE 0 (0.042) y el
  AE 4 (0.045) reconstruyen casi igual.
- **Respuesta directa:** con las dos fuentes de varianza controladas, **no hay evidencia** de que
  comprimir el estado con `latent_dim` = 12 ayude a predecir mejor la recompensa que el estado
  crudo. La estimación puntual favorece ligeramente al latente (−7%), pero el intervalo va desde una
  mejora del 22% hasta un empeoramiento del 12%. El resultado es **ambiguo**: tampoco hay evidencia
  de que perjudique. Lo que sí queda establecido es que el Autoencoder concreto que se entrene
  importa más que la semilla de la LSTM, y que un Autoencoder puede salir claramente útil (AE 0) o
  claramente perjudicial (AE 4).
- Salvedad fijada: con 5 Autoencoders, el bootstrap por conglomerados tiene pocos grupos. Aquí no
  cambia nada: la conclusión no depende de un margen pequeño, porque el 0 queda bien dentro del IC.

**Cómputo.** Addendum: 2 Autoencoders (28 s) y 14 LSTM (4,337 s de proceso), 26 min de pared. Toda la
exploración `latent_sweep` hasta aquí: 61 LSTM (7.4 h de proceso) y 11 Autoencoders (582 s), sin
contar las 2 corridas de validación iniciales.

## 12. Fase 2, Parte A: LSTM y Transformer a `latent_dim` = 16 (criterio en `ADDENDUM_transformer_tsmixer.md`)

Los mismos 5 Autoencoders de `latent_dim` = 16 (estado de 24 dims, semillas 0–4) para las dos
arquitecturas. Tope de 300 épocas y paciencia 15 en las dos ramas. Nuevas: 4 Autoencoders, 22 LSTM
y 35 Transformer, 2.46 h de pared.

**Convergencia.** Las 70 corridas se cortaron por early stopping; ninguna llegó al tope.

| Grupo | Época de corte, mediana (rango) | Mejor época, mediana |
|---|---|---|
| LSTM latente | 124 (86–227) | 109 |
| LSTM cruda | 110 (95–163) | 96 |
| Transformer latente | 62 (38–104) | 47 |
| Transformer crudo | 75 (57–116) | 60 |

La fila del AE 0 de la LSTM coincide con el barrido (sección 2: −31.9%, +27.5%, +32.0%).

**Media geométrica del `reward_mse` (menor es mejor), por celda y por Autoencoder**

| | s0 | s1 | s2 | s3 | s4 | AE |
|---|---|---|---|---|---|---|
| LSTM, AE 0 | 300.3 | 307.1 | 239.2 | 365.6 | 308.4 | 301.4 |
| LSTM, AE 1 | 278.2 | 198.3 | 253.2 | 290.2 | 326.6 | 265.7 |
| LSTM, AE 2 | 266.0 | 294.2 | 242.2 | 239.4 | 243.3 | 256.2 |
| LSTM, AE 3 | 343.6 | 330.3 | 325.4 | 209.2 | 224.2 | 280.4 |
| LSTM, AE 4 | 243.1 | 285.6 | 263.4 | 212.0 | 270.4 | 253.6 |
| **LSTM cruda (10 corridas)** | | | | | | **300.9** |
| Transformer, AE 0 | 464.8 | 400.2 | 383.9 | 423.0 | 411.6 | 415.8 |
| Transformer, AE 1 | 423.1 | 474.8 | 554.6 | 673.7 | 589.4 | 536.0 |
| Transformer, AE 2 | 432.6 | 465.2 | 448.4 | 579.8 | 659.6 | 510.0 |
| Transformer, AE 3 | 321.9 | 374.9 | 496.5 | 491.0 | 524.0 | 434.1 |
| Transformer, AE 4 | 436.3 | 280.3 | 294.2 | 313.4 | 348.5 | 330.3 |
| **Transformer crudo (10 corridas)** | | | | | | **329.1** |

Reducción mediana por celda (descriptiva): LSTM, 16/25 celdas positivas; Transformer, 7/25. Tabla
completa en `grid_arch_A.out`.

**ANOVA (5 × 5) y decisión**

| | Autoencoder | Semilla del modelo | Δ (z − crudo) | IC del 95% por conglomerados | Permutación por corrida | Conclusión (criterio fijado) |
|---|---|---|---|---|---|---|
| LSTM | F = 0.80, p = 0.54 (σ ≈ 0) | F = 0.27, p = 0.90 | **−10.0%** | [−21.3%, +2.8%] | p = 0.11 | **SIN EVIDENCIA** |
| Transformer | F = 7.38, **p = 0.001** (σ ≈ 18%) | F = 1.97, p = 0.15 | **+33.4%** | **[+9.8%, +60.4%]** | p = 0.001 | **EVIDENCIA: el latente predice PEOR** |

**Entre arquitecturas** (bootstrap conjunto sobre los mismos Autoencoders): Δ_LSTM − Δ_Transformer =
−0.393 en log, IC del 95% de −0.627 a −0.164. **Hay evidencia de que la ayuda del Autoencoder
difiere entre la LSTM y el Transformer**: es neutra con la LSTM y perjudicial con el Transformer.

**Lectura (intermedia; TSMixer, pendiente de la Parte B).**

- **LSTM a 16:** la misma conclusión que a 12 (sección 11). No hay evidencia de ventaja ni de
  desventaja, con una estimación puntual algo favorable al latente (−10%). Aquí, a diferencia de
  12, la semilla del Autoencoder **no** explica una varianza significativa (p = 0.54): con 16
  dimensiones, los cinco Autoencoders se comportan de forma parecida para la LSTM.
- **Transformer a 16:** el latente **empeora** la predicción de la recompensa en un 33%, con un IC
  que excluye 0. Cuatro de los cinco Autoencoders quedan por encima del crudo (415–536 frente a
  329); solo el AE 4 lo iguala (330). En el Transformer, la semilla del Autoencoder vuelve a ser la
  fuente dominante de varianza (p = 0.001).
- Coherente con el Experimento 3: el Transformer predice la recompensa peor que la LSTM. Además,
  sobre el estado crudo el Transformer (329) queda cerca de la LSTM (301): buena parte de su
  desventaja en el Experimento 3 aparece al trabajar sobre el latente.
- Salvedades: una sola `latent_dim` (16); el Transformer usa los hiperparámetros del Experimento 3
  sin ajustar al latente ni al crudo; y es una sola comparación preregistrada entre arquitecturas
  (quedan dos más cuando esté TSMixer).

Cómputo de la Parte A: 2.46 h de pared (3 en paralelo); las 22 LSTM nuevas, 9,292 s de proceso.

## 13. Fase 2, Parte B y cierre: TSMixer y la comparación de las tres arquitecturas (criterio en `ADDENDUM_transformer_tsmixer.md`)

Los mismos 5 Autoencoders de `latent_dim` = 16 que en la sección 12, y el mismo protocolo: tope de 300
épocas, paciencia 15, `train()` de `training/train_world_model_tsmixer_raw.py` para las dos ramas.
TSMixer: 25 corridas latentes (5 Autoencoders × 5 semillas) y 10 crudas (semillas 0–9), 1.23 h de
pared. El análisis es el preregistrado, sin cambios. Salida completa en `grid_arch_B.out`; números en
`grid_arch_lstm_transformer_tsmixer.json` y `gridarch_tsmixer.json`.

**Cómo se calculó.** `grid_arch.py` no pudo ejecutarse en este equipo: Smart App Control bloquea la
DLL de `kiwisolver`, de la que depende matplotlib, y la evaluación oficial importa matplotlib aunque
no dibuje. Por eso la evaluación de TSMixer se hizo con una copia literal de esa evaluación, en
`_DESCARTABLE_bloqueo_windows/` (ver su README). Antes de usarla se recalcularon las 70 corridas de
LSTM y Transformer desde los checkpoints: reproduce **exactamente**, con igualdad de floats,
`gridarch_lstm.json`, `gridarch_transformer.json` y `grid_arch_lstm_transformer.json`. Las filas de
LSTM y Transformer de abajo son idénticas a las de la sección 12.

**Convergencia.** Las 35 corridas se cortaron por early stopping; ninguna llegó al tope.

| Grupo | Época de corte, mediana (rango) | Mejor época, mediana |
|---|---|---|
| TSMixer latente | 126 (88–226) | 111 |
| TSMixer crudo | 107 (93–141) | 92 |

**Media geométrica del `reward_mse` (menor es mejor), TSMixer**

| | s0 | s1 | s2 | s3 | s4 | AE |
|---|---|---|---|---|---|---|
| TSMixer, AE 0 | 606.5 | 438.4 | 487.8 | 600.6 | 637.8 | 548.6 |
| TSMixer, AE 1 | 735.1 | 808.9 | 637.0 | 821.6 | 976.5 | 788.0 |
| TSMixer, AE 2 | 558.7 | 675.2 | 1085.3 | 579.2 | 698.3 | 697.9 |
| TSMixer, AE 3 | 720.6 | 571.6 | 595.4 | 816.2 | 856.2 | 702.8 |
| TSMixer, AE 4 | 623.3 | 493.9 | 652.2 | 736.3 | 974.1 | 678.7 |
| **TSMixer crudo (10 corridas)** | | | | | | **477.9** |

Crudo por semilla: 598.5, 475.7, 422.6, 641.7, 463.7, 505.4, 462.6, 443.5, 390.2 y 429.0. Los cinco
Autoencoders quedan por encima del crudo (549–788 frente a 478). Reducción mediana por celda
(descriptiva): 5/25 celdas positivas.

**Tabla final: efecto del Autoencoder (`latent_dim` = 16) por arquitectura**

Δ = media de g (log del `reward_mse` geométrico) de las 25 corridas latentes − media de g de las 10
crudas de la misma arquitectura. Se reporta como exp(Δ) − 1; un valor negativo significa que el
latente predice mejor.

| Arquitectura | Latente (25) | Crudo (10) | exp(Δ) − 1 | IC del 95% por conglomerados | Permutación por corrida | Conclusión (criterio fijado) |
|---|---|---|---|---|---|---|
| LSTM | 270.9 | 300.9 | **−10.0%** | [−21.3%, +2.8%] | p = 0.11 | **SIN EVIDENCIA** |
| Transformer | 439.0 | 329.1 | **+33.4%** | [+9.8%, +60.4%] | p = 0.001 | **EVIDENCIA: el latente predice PEOR** |
| TSMixer | 678.6 | 477.9 | **+42.0%** | [+21.1%, +65.0%] | p = 0.0001 | **EVIDENCIA: el latente predice PEOR** |

**ANOVA de dos factores (Autoencoder × semilla del modelo temporal, 5 × 5, sobre g)**

| | Autoencoder | Semilla del modelo | Residuo | Fuente dominante |
|---|---|---|---|---|
| LSTM | F = 0.80, p = 0.54 (σ ≈ 0) | F = 0.27, p = 0.90 (σ ≈ 0) | σ = 0.178 | ninguna |
| Transformer | F = 7.38, **p = 0.001** (σ ≈ 18%) | F = 1.97, p = 0.15 (σ ≈ 7%) | σ = 0.157 | el Autoencoder |
| TSMixer | F = 2.50, p = 0.084 (σ ≈ 10%) | F = 2.20, p = 0.12 (σ ≈ 9%) | σ = 0.187 | ninguna clara |

En TSMixer ningún factor es significativo, y sus componentes estimadas son parecidas (10% y 9%): no
domina ni la varianza del Autoencoder ni la de la inicialización. La mayor parte de la varianza es
residual, es decir, la interacción entre los dos factores más el ruido de cada corrida. El patrón de
la LSTM a 12 (sección 11) y del Transformer, donde el Autoencoder domina, no se repite con claridad
aquí. Tampoco se contradice: con 5 niveles por factor, un p = 0.084 no permite afirmar ninguna de
las dos cosas.

**¿La ayuda del Autoencoder difiere entre arquitecturas?** Bootstrap conjunto, Δ_A − Δ_B en log:

| Par | Δ_A − Δ_B | IC del 95% | Conclusión |
|---|---|---|---|
| LSTM − Transformer | −0.393 | [−0.627, −0.164] | **EVIDENCIA de diferencia** |
| LSTM − TSMixer | −0.456 | [−0.672, −0.224] | **EVIDENCIA de diferencia** |
| Transformer − TSMixer | −0.063 | [−0.286, +0.145] | SIN EVIDENCIA de diferencia |

**Comparaciones múltiples.** Hay tres pruebas por arquitectura y tres entre arquitecturas, cada una
con un 5% nominal y sin corrección, como fijó el addendum. Si las tres pruebas por arquitectura
fueran independientes y ninguna tuviera efecto real, la probabilidad de al menos un falso positivo
sería de hasta 1 − 0.95³ ≈ 14%, no del 5%. Como comprobación complementaria, **no preregistrada**
(`multiple_comparisons.py`, `multiple_comparisons.json`), se recalcularon los IC con corrección de
Bonferroni para 3 pruebas (nivel del 98.33%, mismo bootstrap):

| | IC del 95% | IC de Bonferroni (98.33%) |
|---|---|---|
| LSTM | [−21.3%, +2.8%] | [−23.5%, +5.8%] (incluye 0) |
| Transformer | [+9.8%, +60.4%] | [+5.3%, +66.4%] (excluye 0) |
| TSMixer | [+21.1%, +65.0%] | [+16.8%, +70.0%] (excluye 0) |
| LSTM − Transformer | [−0.627, −0.164] | [−0.675, −0.117] (excluye 0) |
| LSTM − TSMixer | [−0.672, −0.224] | [−0.716, −0.172] (excluye 0) |
| Transformer − TSMixer | [−0.286, +0.145] | [−0.336, +0.188] (incluye 0) |

Ninguna conclusión cambia con la corrección. El resultado del Transformer es el más cercano al
límite: su borde inferior baja de +9.8% a +5.3%.

**Lectura final de la Fase 2.**

- **LSTM:** sin evidencia de que el Autoencoder ayude ni de que perjudique. La estimación puntual es
  −10%, con un IC que va de una mejora del 21% a un empeoramiento del 3%.
- **Transformer y TSMixer:** el latente **empeora** la predicción de la recompensa, un 33% y un 42%
  respectivamente. Los dos IC excluyen 0, también con Bonferroni.
- **La ayuda del Autoencoder depende de la arquitectura:** es neutra con la LSTM, perjudicial con las
  otras dos, y no se distingue entre el Transformer y TSMixer.
- **Sobre el estado crudo el orden es el mismo que en el Experimento 3:** LSTM 301, Transformer 329,
  TSMixer 478. Sobre el latente la distancia crece (271, 439 y 679), así que buena parte de la
  desventaja del Transformer y de TSMixer en el Experimento 3, que se evaluó sobre z, viene de
  trabajar sobre el latente.
- **Qué Autoencoder es "bueno" cambia con la arquitectura** (descriptivo, 5 puntos, sin prueba):
  para la LSTM el AE 0 es el peor, para el Transformer el mejor es el AE 4 y para TSMixer el AE 0.
  No hay un Autoencoder universalmente favorable.
- **Salvedades:**
  - Una sola `latent_dim` (16), el valor oficial.
  - Hiperparámetros del Experimento 3 sin ajustar a ninguna rama. Un Transformer o un TSMixer
    ajustado al latente podría acortar la brecha, y esto no lo descarta.
  - Solo 5 Autoencoders, así que el bootstrap por conglomerados tiene pocos grupos. Aquí no cambia
    nada: los dos IC que excluyen 0 lo hacen con margen.
  - La métrica es la predicción de la recompensa, no el desempeño de control.

**Integridad.** Los 77 md5 de `official_md5_before.txt` coinciden, salvo los de `CLAUDE.md`,
`PROJECT_STATUS.md`, `README.md` y `TODO.md`. Esos cuatro los cambió a propósito el commit `1b63b10`,
y sus md5 en `main` coinciden con el registro. Diferencias con `main` fuera de
`docs/results/exploratory/`:

- esos cuatro documentos y `docs/EXPLORACION_LATENT_DIM.md`;
- los tres archivos nuevos de `36ad129`: `training/train_world_model_{transformer,tsmixer}_raw.py` y
  `tests/test_transformer_tsmixer_raw_protocol.py`. Son añadidos; ningún archivo existente cambia.

**Cómputo de la Fase 2.** Parte A, 2.46 h de pared; Parte B, 1.23 h. La evaluación de los 35
checkpoints de TSMixer tomó 1 min 44 s en CPU.

## 14. Reanálisis del orden entre arquitecturas con 25 corridas cada una (criterio en `ADDENDUM_arch_ranking.md`)

El Experimento 3 oficial afirma que la LSTM predice la recompensa mejor que el Transformer y que
TSMixer en los 10 horizontes, con reducciones medianas del 38.1% y el 56.1%, a partir de **una
corrida por arquitectura**. Aquí se repite esa comparación con las 25 corridas de cada una sobre z
de `latent_dim` = 16 (sección 13), en bloques por Autoencoder.

- **Sin entrenamiento:** solo se leen `gridarch_{lstm,transformer,tsmixer}.json` y los reportes
  oficiales de `results/`.
- **Criterio fijado antes de calcular:** `ADDENDUM_arch_ranking.md`, commit `c3685e0`, con
  Bonferroni desde el inicio.
- **Archivos:** análisis en `arch_ranking.py`; salida en `arch_ranking.out` y `arch_ranking.json`.

**No es una réplica exacta** del Experimento 3: estado de 24 dims en vez de 26, cinco Autoencoders
nuevos en vez del oficial, y tope de 300 épocas con early stopping en vez de 100. La dirección
agregada ya se conocía por la sección 13; lo nuevo son los intervalos.

**A. Tres pares: bootstrap emparejado por Autoencoder, IC con Bonferroni para 3 pruebas (98.33%)**

Diferencia en el `reward_mse` geométrico, exp(D) − 1 (negativo significa que la primera
arquitectura tiene menos error):

| Par | exp(D) − 1 | IC del 98.33% (decide) | IC del 95% | Por Autoencoder (0–4) | Conclusión |
|---|---|---|---|---|---|
| LSTM − Transformer | **−38.3%** | [−50.5%, −24.0%] | [−48.6%, −26.4%] | −28, −50, −50, −35, −23% | **la LSTM predice mejor** |
| LSTM − TSMixer | **−60.1%** | [−66.9%, −49.9%] | [−66.0%, −52.0%] | −45, −66, −63, −60, −63% | **la LSTM predice mejor** |
| Transformer − TSMixer | **−35.3%** | [−48.1%, −22.5%] | [−45.9%, −24.5%] | −24, −32, −27, −38, −51% | **el Transformer predice mejor** |

**El orden LSTM > Transformer > TSMixer se sostiene con evidencia:** los tres IC corregidos excluyen
0, con margen, y la dirección se repite en cada uno de los 5 Autoencoders.

**B. Formato del artículo: reducción de la LSTM por horizonte**

r_h = 1 − (media geométrica de la LSTM / media geométrica de X), con 25 corridas de cada una. IC
del 95% por horizonte, descriptivo: son 20 intervalos sin corrección. La columna "oficial" es la
corrida única del Experimento 3.

| h | r_h vs Transformer | IC del 95% | Oficial | r_h vs TSMixer | IC del 95% | Oficial |
|---|---|---|---|---|---|---|
| 1 | +14.3% | [+5.1%, +24.2%] | +3.8% | +24.2% | [+17.5%, +31.6%] | +27.9% |
| 2 | +25.3% | [+11.6%, +38.2%] | +23.9% | +37.6% | [+30.0%, +44.1%] | +41.5% |
| 3 | +29.5% | [+13.3%, +43.7%] | +33.8% | +50.8% | [+41.4%, +57.9%] | +54.1% |
| 4 | +32.2% | [+16.6%, +45.5%] | +42.9% | +55.8% | [+44.0%, +64.0%] | +50.5% |
| 5 | +33.8% | [+17.6%, +47.2%] | +48.1% | +59.4% | [+48.1%, +67.4%] | +52.0% |
| 6 | +37.8% | [+21.6%, +51.5%] | +50.0% | +63.1% | [+52.3%, +70.4%] | +58.1% |
| 7 | +44.5% | [+31.0%, +56.4%] | +44.7% | +68.6% | [+60.4%, +74.4%] | +63.9% |
| 8 | +49.7% | [+37.8%, +59.8%] | +42.4% | +72.2% | [+65.5%, +77.0%] | +66.0% |
| 9 | +51.7% | [+40.7%, +60.9%] | +30.4% | +72.9% | [+66.7%, +77.5%] | +64.3% |
| 10 | +52.9% | [+42.4%, +61.5%] | +14.0% | +71.8% | [+65.5%, +76.6%] | +60.7% |
| **Mediana (R)** | **+35.8%** | **[+17.8%, +50.8%]** (97.5%, Bonferroni 2) | **38.1%** | **+61.2%** | **[+48.4%, +69.6%]** (97.5%, Bonferroni 2) | **56.1%** |
| Horizontes con IC a favor de la LSTM | 10/10 | | 10/10 | 10/10 | | 10/10 |

**C. ¿Qué habría dado una sola semilla?** Se repitió el procedimiento del artículo con una corrida de
cada lado, sobre los 125 pares (LSTM i, X j) dentro de cada Autoencoder:

| | Reducción mediana: P5 / P25 / P50 / P75 / P95 | Percentil del valor publicado | La LSTM gana 10/10 | La LSTM gana ≥ 6/10 | Reducción < 0 |
|---|---|---|---|---|---|
| vs Transformer | −1.0 / +20.8 / +33.0 / +49.4 / +62.5% | 38.1% → P57 | 72.0% | 92.0% | 5.6% |
| vs TSMixer | +25.8 / +48.5 / +61.3 / +69.4 / +79.3% | 56.1% → P36 | 98.4% | 100% | 0% |

**Valores publicados, según el criterio fijado**

- **38.1% frente al Transformer: COMPATIBLE.** Queda dentro del IC de R ([+17.8%, +50.8%]; la
  estimación es +35.8%) y cae en el percentil 57 de lo que da una sola semilla, es decir, en el
  centro. **No estaba inflado ni desinflado.** Pero la dispersión con una sola semilla es grande: el
  90% central va de −1% a +63%. Con una semilla desafortunada, el Experimento 3 podría haber
  reportado una ventaja casi nula (en el 5.6% de los pares la reducción mediana es negativa), y en
  el 28% de los pares la LSTM no habría ganado los 10 horizontes.
- **56.1% frente a TSMixer: COMPATIBLE.** Queda dentro del IC ([+48.4%, +69.6%]; la estimación es
  +61.2%) y cae en el percentil 36, algo por debajo del centro. Si acaso, **subestimaba levemente** la
  ventaja, sin salir del rango típico. Con una sola semilla el resultado es muy estable: la LSTM gana
  10/10 en el 98.4% de los pares.
- **"10/10 horizontes":** se sostiene para los dos. En los 20 horizontes el IC de r_h excluye 0 a
  favor de la LSTM.

**Salvedad sobre la forma por horizonte.** La mediana del Transformer coincide, pero el perfil de la
corrida oficial no: sube hasta +50% en h = 6 y cae a +14% en h = 10. En la grilla, la ventaja crece de
forma monótona hasta +53% en h = 10. Cuatro de los diez valores oficiales frente al Transformer (h = 1,
5, 9 y 10) y dos frente a TSMixer (h = 9 y 10) quedan fuera del IC por horizonte. Esto **no es una
contradicción**:

- esos IC son para la media geométrica de 25 corridas, no intervalos de predicción para una sola
  corrida, que naturalmente cae fuera con frecuencia;
- son 20 intervalos sin corrección;
- el protocolo difiere.

Lo que sí indica es que la caída de la ventaja en h = 9–10 frente al Transformer, visible en el
Experimento 3, es probablemente propia de esa corrida y no un rasgo general.

**Lectura.**

- A diferencia de la ventaja inicial del Autoencoder (secciones 4 y 10–11), que no resistió más
  semillas, **la ventaja de la LSTM sobre el Transformer y sobre TSMixer se sostiene con 25 corridas**,
  con corrección por comparaciones múltiples y en cada Autoencoder. El orden completo LSTM >
  Transformer > TSMixer queda sostenido.
- Los valores publicados (38.1% y 56.1%) son **representativos**: ninguno cae en las colas de la
  distribución de una semilla. La conclusión del Experimento 3 (mantener la LSTM) sale reforzada.
- **Alcance:**
  - Es sobre **z**. Sobre el estado crudo (sección 13), la distancia entre la LSTM (301) y el
    Transformer (329) es mucho menor, y aquí no se pone a prueba.
  - Una sola `latent_dim` (16).
  - Hiperparámetros del Experimento 3 sin ajustar.
  - Protocolo de 24 dims y tope de 300, no idéntico al oficial.

**Integridad.** Los md5 oficiales coinciden (salvo los cuatro documentos cambiados a propósito en
`1b63b10`, y `CLAUDE.md` también en `832d534`). El inventario de `.pt`/`.zip`/`.npz` de
`models/checkpoints/` es idéntico antes y después (525 archivos), y no hay ningún archivo en
`models/checkpoints/` ni en `results/` posterior al addendum. El cálculo tomó 1.3 s.
