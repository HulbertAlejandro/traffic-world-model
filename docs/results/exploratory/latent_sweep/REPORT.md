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
