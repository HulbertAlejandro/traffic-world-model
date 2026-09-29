# Exploratorio: barrido de `latent_dim` sobre el estado de 24 dimensiones

**Experimento personal y exploratorio.** No forma parte del Experimento 0 oficial
(`docs/results/experiment_0_multiseed_300ep.json`), no lo reemplaza y no se cita en la
documentación principal ni en el artículo salvo decisión explícita del autor. Rama:
`exploratory/latent-dim-sweep`, sin push.

Este archivo se escribió y se commiteó **antes de ejecutar la primera corrida**. Los valores de
`latent_dim`, las semillas y el criterio de ganador que fija no se cambian después de ver resultados.

## Hipótesis

Con 16 dimensiones latentes frente a 26 (de las cuales 2 son siempre cero), el Autoencoder
comprime poco, y eso podría explicar por qué su ventaja no se distingue del ruido de inicialización
(10 semillas: prueba de signo p = 0.75, Wilcoxon p = 0.19). Otro `latent_dim` podría mostrar una
señal más clara, en cualquier dirección.

## Paso 1: verificación de las dimensiones constantes (hecha antes de este MANIFEST)

Sobre los 80 episodios de `datasets/raw/` (4,800 estados sin normalizar):

- One-hot de fase (índices 20–23), activaciones en `states`: 2,464 / 2,336 / **0** / **0**. En
  `next_states`: 2,426 / 2,374 / 0 / 0. Las posiciones 22 y 23 tienen varianza exactamente cero.
- Tiempo restante de fase (índice 25): vale 0 en **4,720 de 4,800** estados (98.33%). Los otros 80
  valen 5 y son exactamente el paso 0 de cada episodio. En `next_states` vale 0 en los 4,800.
- Las posiciones 20 y 21 son complementarias (2,464 + 2,336 = 4,800): juntas aportan un solo bit.
- En los splits normalizados (`datasets/processed/*.npz`), las columnas de varianza cero son solo la
  22 y la 23, en los tres splits.

La premisa se confirma: 2 dimensiones siempre en cero y 1 dimensión en cero salvo en el primer paso.

## Paso 2: estado de 24 dimensiones

Se eliminan las columnas 22 y 23 de `states` y `next_states` de `datasets/processed/{train,
validation,test}.npz` (ya normalizados; el escalador se ajustó solo con train y no se recalcula).
Todo lo demás se copia sin cambios. La columna 25 (tiempo restante) se conserva. Se guarda en
`models/checkpoints/exploratory_latent_sweep/data/` (no versionado); la versión oficial de 26 no se
toca.

## Protocolos (fijados)

**Autoencoder** (uno por `latent_dim`; decisión del autor: `hidden_dim = latent_dim`). Es la
arquitectura oficial, `Linear(24→h)+ReLU+Linear(h→k)` en el encoder y el decoder simétrico, con
`h = k`. Con k = 16 es exactamente la forma del Autoencoder oficial (16/16). El protocolo es el
oficial de `training/train_autoencoder.py`, con sus mismas funciones: semilla 0 (`torch` y `numpy`,
como el oficial), Adam con lr 1e-3, lotes de 32 y 100 épocas, sin early stopping. El mejor
checkpoint es el de menor pérdida de reconstrucción en validación, y se codifica con él.

**LSTM temporal** (las dos ramas). El protocolo se importa de `training/train_world_model.py`, igual
que la rama cruda oficial: `_set_seeds`, `train_one_epoch`, `validate`, `compute_reward_scaler` y
las constantes. Tope de 300 épocas, early stopping con paciencia 15, lr 1e-3, weight_decay 1e-4,
lotes de 32, hidden 128, secuencia 16 y escalador de recompensa ajustado con el split de train.

**Evaluación.** La función `evaluate()` de `scripts/compare_experiment_0_multiseed.py`, sin
cambios: split de test (12 episodios), rollouts autorregresivos con las acciones reales, h = 1..10 y
`reward_mse` desnormalizado.

**Reducción por semilla:** mediana sobre h = 1..10 de (crudo − z)/crudo (la misma definición que la
tabla oficial). Positiva significa que el Autoencoder ayuda.

## Verificación de los ejecutores (antes del barrido)

Los ejecutores exploratorios se validan reproduciendo resultados oficiales:

- Autoencoder, con los datos oficiales de 26 dimensiones y k = 16: sus pesos deben ser idénticos a
  `models/checkpoints/autoencoder_best.pt`.
- LSTM, semilla 0, sobre `train_raw_seq.npz` y sobre `train_latent.npz`: sus pesos deben ser
  idénticos a `exp0_multiseed_300ep/{raw,z}/seed0`.

Si alguna no coincide, se reporta y el barrido no se ejecuta hasta aclararlo.

## Paso 3: barrido (valores y semillas fijados)

- `latent_dim` ∈ **[4, 8, 12, 16, 20]**.
- Semillas de la LSTM: **[0, 1, 2]** para cada `latent_dim` (15 corridas z).
- Rama cruda de 24 dimensiones: semillas **[0, 1, 2]**, entrenadas **una sola vez** (3 corridas) y
  compartidas por todos los valores de `latent_dim`.

**Criterio de ganador (fijado):** el `latent_dim` con la **mayor mediana, sobre sus 3 semillas, de
la reducción por semilla**. Primer desempate (valores exactamente iguales): más pares semilla ×
horizonte ganados por z (de 30). Segundo desempate: el `latent_dim` menor. El criterio elige el valor
más favorable al Autoencoder aunque su mediana sea negativa.

## Paso 4: refuerzo del ganador (semillas fijadas)

- Semillas adicionales: **[3, 4, 5, 6, 7, 8, 9]** para la rama z con el `latent_dim` ganador y para
  la rama cruda de 24 dimensiones. En total, 10 por rama (0–9).
- Se calcula exactamente lo mismo que para `latent_dim` = 16 con 10 semillas:
  - pares semilla × horizonte ganados (de 100);
  - mediana de las medianas por horizonte y mediana sobre los 100 pares;
  - semillas con reducción positiva y prueba de signo binomial exacta bilateral;
  - Wilcoxon de rangos con signo exacto bilateral sobre las 10 reducciones por semilla (enumeración
    de las 2^10 asignaciones de signo, el mismo método de la exploración anterior).
- **Salvedad fijada de antemano:** las semillas 0–2 del ganador sirvieron para elegirlo, así que las
  10 semillas llevan un sesgo de selección a favor de z. Por eso se reportan también, por separado,
  las mismas pruebas sobre las **7 semillas nuevas (3–9)** solas, la verificación sin sesgo.
- **Referencia con la que se compara** (exploración anterior: estado de 26 dimensiones, Autoencoder
  oficial de k = 16, semillas 0–4, 10, 20, 30, 40 y 50): z gana 60/100 pares; mediana de las
  medianas por horizonte +8.4%; mediana sobre los 100 pares +5.4%; 6/10 semillas positivas; signo
  p = 0.75; Wilcoxon p = 0.19. No existe un número aparte para "26 sin comprimir": la rama cruda es la
  referencia de cada comparación.
- **Información adicional fijada:** crudo de 24 frente a crudo de 26 en las semillas 0–4, las mismas
  entradas salvo dos columnas constantes en cero. Sirve como calibración del ruido entre
  inicializaciones.

## Reglas

- No se repite ni se excluye ninguna corrida por su resultado.
- Un fallo técnico (error, o llegar al tope de 300 épocas sin que corte el early stopping) se
  reporta tal cual.
- Archivos oficiales: su md5 se registra antes de empezar (`official_md5_before.txt`) y se compara
  al final.
- Pesos y datos en `models/checkpoints/exploratory_latent_sweep/` (no versionados). Scripts, logs,
  JSON y resumen, en esta carpeta.
