# Addendum (escrito y commiteado antes de calcular): ¿se sostiene el orden LSTM > Transformer > TSMixer con 25 corridas?

Pedido del autor. El Experimento 3 oficial concluyó que la LSTM predice la recompensa mejor que el
Transformer y que TSMixer en los 10 horizontes, con reducciones medianas del 38.1% y el 56.1%, pero
con **una sola corrida por arquitectura**. La Fase 2 (secciones 12 y 13 de REPORT.md) dejó 25
corridas de cada arquitectura sobre z de `latent_dim` = 16. Pregunta: ¿el orden entre arquitecturas
se sostiene con esa potencia? ¿Los valores publicados son representativos de lo que da una sola
semilla, o estaban inflados o desinflados?

**No se entrena nada.** Solo se leen `gridarch_{lstm,transformer,tsmixer}.json` (clave
`z_reward_mse`, forma 5 × 5 × 10: Autoencoder × semilla del modelo × horizonte). No se usan las
ramas crudas.

## Lo que ya se sabe antes de calcular (transparencia)

- **La dirección agregada ya es conocida.** La sección 13 publicó la media geométrica global del
  `reward_mse` sobre z: LSTM 271, Transformer 439 y TSMixer 679. Todavía no se calcularon los IC
  pareados por Autoencoder, las reducciones por horizonte ni la distribución por pares de corridas.
- **El protocolo no es el del Experimento 3**, así que esto es una réplica aproximada, no una
  repetición exacta:

| | Experimento 3 oficial | Esta grilla |
|---|---|---|
| Estado | 26 dims | 24 dims (sin las columnas 22 y 23, siempre en cero) |
| Autoencoder | el oficial (semilla 0, entrenado sobre 26 dims) | 5 nuevos (semillas 0–4, entrenados sobre 24 dims); el oficial no está entre ellos |
| Tope de épocas | 100 (LSTM, mejor época 93; TSMixer llegó al tope) | 300 con paciencia 15 (todas cortaron por early stopping) |
| Corridas | 1 por arquitectura | 25 por arquitectura (5 por Autoencoder) |

  Si los valores publicados quedan fuera del IC nuevo, la causa puede ser la semilla **o** estas
  diferencias de protocolo. El análisis por pares de corridas (punto C) busca separar las dos.

## Métrica (fijada)

Por corrida, g = media sobre h = 1..10 de log(`reward_mse`), la de toda la Fase 2. Por corrida y
horizonte, l_h = log(`reward_mse` en h).

## A. Comparaciones principales: tres pares, Bonferroni desde el inicio

- **Pares:** LSTM − Transformer, LSTM − TSMixer y Transformer − TSMixer.
- **Estadístico:** D = media de g de las 25 corridas de A − media de g de las 25 de B. Con el diseño
  balanceado equivale a la media, sobre los 5 Autoencoders, de la diferencia dentro de cada
  Autoencoder. Se reporta también exp(D) − 1 (negativo significa que A tiene menos error que B).
- **Diseño en bloques por Autoencoder:** las tres arquitecturas se entrenaron sobre los mismos 5 z,
  así que el Autoencoder es un bloque común. Las semillas del modelo **no** se emparejan entre
  arquitecturas: la semilla s de la LSTM y la del Transformer no comparten nada.
- **Bootstrap por conglomerados emparejado por bloque**, B = 100,000, un generador
  `np.random.default_rng(12345)` nuevo por par:
  1. se remuestrean 5 índices de Autoencoder, **los mismos para A y B**;
  2. dentro de cada Autoencoder remuestreado, se remuestrean 5 semillas de A y, de forma
     independiente, 5 de B;
  3. se calcula D en la réplica.

  En código: `ai = rng.integers(0, 5, (B, 5))`, luego `siA = rng.integers(0, 5, (B, 5, 5))` y luego
  `siB` igual, en ese orden.
- **IC con Bonferroni para 3 pruebas**, nivel del 98.33% (percentiles 0.8333 y 99.1667). Es el IC que
  decide. El del 95% se reporta solo como referencia.
- **Criterio por par:**
  - IC por debajo de 0: **evidencia de que A predice mejor que B**.
  - IC por encima de 0: **evidencia de que B predice mejor**.
  - Si incluye 0: **SIN EVIDENCIA** de diferencia, sin buscar otro corte.
- **Criterio para el orden:**
  - Se considera **sostenido con evidencia** solo si los tres IC excluyen 0 en la dirección
    LSTM < Transformer < TSMixer (en error).
  - Si falla alguno, se reporta qué partes del orden se sostienen y cuáles no.

## B. Formato del artículo: reducción por horizonte, con IC

- **Reducción de la LSTM frente a X ∈ {Transformer, TSMixer} en el horizonte h:**
  r_h = 1 − exp(media de l_h en las 25 corridas de la LSTM − media de l_h en las 25 de X). Es
  1 − (media geométrica de la LSTM / media geométrica de X): la definición del artículo,
  (X_h − LSTM_h)/X_h, con medias geométricas sobre corridas en lugar de un solo valor.
- **Resumen comparable con 38.1% y 56.1%:** R = mediana de r_1..r_10.
- **IC:** del mismo bootstrap que en A (mismo generador y réplicas, par LSTM − X), aplicado a la
  matriz de 10 horizontes, así que los horizontes se remuestrean conjuntamente.
  - Por horizonte, IC del 95% **descriptivo**: son 20 intervalos, sin corrección, y no deciden nada.
  - Para R, IC con **Bonferroni para 2 comparaciones** (Transformer y TSMixer contra lo publicado),
    nivel del 97.5% (percentiles 1.25 y 98.75).
- **Se reporta también:**
  - en cuántos horizontes el IC del 95% de r_h excluye 0 a favor de la LSTM (el análogo de "gana en
    10/10");
  - los valores oficiales por horizonte, lado a lado.

## C. ¿Qué habría dado una sola semilla? (descriptivo)

- Se reproduce el procedimiento del artículo con una corrida de cada lado: para cada Autoencoder y
  cada par (corrida i de la LSTM, corrida j de X) del mismo Autoencoder, se calcula la mediana
  sobre h de (X_h − LSTM_h)/X_h y cuántos horizontes gana la LSTM.
- Son 5 × 5 × 5 = 125 pares por comparación. Se reportan:
  - su mediana y sus percentiles 5, 25, 75 y 95;
  - el percentil en que caen el 38.1% y el 56.1% publicados;
  - la fracción de pares en que la LSTM gana los 10 horizontes.

## Criterio para los valores publicados (fijado)

- **Compatible:** el valor publicado (38.1% o 56.1%) queda dentro del IC del 97.5% de R.
- **Fuera por arriba:** queda por encima del IC. El artículo **sobreestima** la ventaja respecto de
  lo que da esta grilla.
- **Fuera por abajo:** queda por debajo. La **subestima**.
- En los dos últimos casos se atribuye a "la semilla única" solo si el valor publicado cae además en
  las colas de la distribución de C (por debajo del percentil 5 o por encima del 95). Si no, se
  declara que la discrepancia puede deberse a las diferencias de protocolo y no se puede separar.
- "Gana en 10/10": se compara con el número de horizontes con IC a favor de la LSTM y con la fracción
  de pares de C que ganan 10/10.

## Reglas

- Ningún número del Experimento 3 oficial cambia. Esto no modifica su decisión (el criterio fijado de
  reemplazo se evaluó sobre la corrida oficial) y no se reescribe el artículo sin decisión explícita
  del autor.
- Se verifican los md5 oficiales antes y después, y que no aparezca ningún checkpoint nuevo.
- El script de análisis (`arch_ranking.py`) solo usa numpy y los JSON citados. No importa matplotlib
  ni la evaluación oficial.
