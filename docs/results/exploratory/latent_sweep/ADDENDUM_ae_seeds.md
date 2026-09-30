# Addendum (escrito y commiteado antes de entrenar): semillas del Autoencoder con `latent_dim` = 12

Pedido del autor. Pregunta: ¿cuánto de la ventaja de `latent_dim` = 12 depende del Autoencoder
concreto (semilla 0) y cuánto de la semilla de la LSTM? El MANIFEST y los addenda anteriores no
cambian.

## Diseño (fijado)

- **Autoencoders** de `latent_dim` = 12 (`hidden_dim` = 12, protocolo oficial: Adam, lr 1e-3, lotes
  de 32, 100 épocas, mejor checkpoint por validación) con semillas **0, 1 y 2**, sobre el estado de
  24 dimensiones. El de semilla 0 es `ae12/`, ya existente, y **no se reentrena**. Los nuevos van en
  `ae12_seed1/` y `ae12_seed2/`. `sweep.py train-ae` recibe `--seed`; con 0, el comportamiento es el
  de siempre. Se comprueba entrenando la semilla 0 en una carpeta temporal y comparando tensor a
  tensor con `ae12/`.
- **LSTM** con semillas **0, 1 y 2** por Autoencoder, con el protocolo de siempre (tope de 300
  épocas, paciencia 15). Para el Autoencoder 0 se reutilizan `z12/seed{0,1,2}`. Son 6 corridas
  nuevas, en `z12_ae1/` y `z12_ae2/`, 3 en paralelo.
- **Rama cruda:** `raw24/seed{0,1,2}`, ya existente, sin reentrenar.
- Como los tres latentes tienen 12 dimensiones, la LSTM de semilla s arranca con los **mismos pesos
  y el mismo orden de lotes** sobre los tres Autoencoders. Fijar la columna fija de verdad la
  aleatoriedad de la LSTM, no solo la etiqueta.

## Métrica y análisis (fijados)

Celda r(a, s) = reducción mediana sobre h = 1..10 de (crudo24_s − z_{a,s})/crudo24_s, la métrica de
siempre. Tabla de 3×3: filas = semilla del Autoencoder, columnas = semilla de la LSTM.

1. **Entre Autoencoders:** para cada columna s, la varianza muestral (ddof = 1) y el rango de
   r(·, s). Se promedia la varianza sobre las 3 columnas.
2. **Entre semillas de LSTM:** para cada fila a, la varianza muestral y el rango de r(a, ·). Se
   promedia sobre las 3 filas. Además, la varianza de las 10 reducciones del Autoencoder 0
   (semillas 0–9 de `winner_z12_10seeds.json`).
3. **ANOVA de dos factores sin réplica** sobre la tabla de 3×3: cuadrados medios de filas, columnas
   y residuo, y componentes de varianza σ²_AE = (CM_filas − CM_res)/3 y σ²_LSTM =
   (CM_col − CM_res)/3, truncadas en 0.
4. **Comparación principal:** qué fuente es mayor, según (1) frente a (2, dentro de la tabla de 3×3)
   y según σ²_AE frente a σ²_LSTM. No hay prueba de significancia decisiva con 3×3; se reporta la
   F del ANOVA solo como indicación.
5. **Análisis secundario (fijado):** en la métrica principal, el efecto de la columna s incluye el
   ruido de `raw24/seed s`, compartido por las tres filas. Por eso se repite la descomposición sobre
   una métrica que solo depende de z: el log de la media geométrica del `reward_mse` de z sobre
   h = 1..10.
6. **¿Es atípico el Autoencoder 0?** Sí, solo si se cumplen las dos condiciones:
   - su media de fila es la mayor o la menor de las tres;
   - su distancia a la media de las otras dos filas supera 2·√(CM_res/2), el error estándar de esa
     diferencia bajo el ruido residual.

   Si no, es representativo. Además, de forma descriptiva: cuántas de las 9 celdas son positivas y
   la media de fila de cada Autoencoder.
