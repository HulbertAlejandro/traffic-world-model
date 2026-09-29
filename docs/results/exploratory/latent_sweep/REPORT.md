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
