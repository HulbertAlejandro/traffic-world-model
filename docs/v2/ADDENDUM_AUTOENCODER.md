# v2 — Fase 2: ¿comprimir el estado de 104 dimensiones ayuda a la LSTM? (pre-registro)

Escrito y commiteado **antes de entrenar ningún modelo**. Alcance: **solo la LSTM**. La pregunta
es si "LSTM sobre el estado comprimido por un Autoencoder" predice la recompensa mejor que "LSTM
sobre el estado normalizado de 104 dimensiones", y qué tamaño de compresión es el más prometedor.

El diseño incorpora desde el principio lo que en la v1 se descubrió tarde:

| Problema en la v1 | Cómo se evita aquí |
|---|---|
| Protocolo desigual entre las dos ramas | Una sola función de entrenamiento para las dos ramas, verificada por tests (sección 1) |
| Una sola repetición | Varias semillas de LSTM por rama desde la primera ronda (sección 3) |
| Varianza del propio Autoencoder | Diseño cruzado (Autoencoder × semilla de LSTM) y bootstrap por conglomerados (sección 3) |
| Conjunto de prueba chico | 24 episodios de prueba, el doble que en la v1, y reporte de la concentración del error por episodio (sección 4) |

Código:

| | |
|---|---|
| Experimento | `training/v2_compression_experiment.py` |
| Orquestación | `scripts/v2/run_compression_experiment.py` |
| Análisis | `scripts/v2/analyze_compression_experiment.py` |
| Datos | `scripts/v2/prepare_dataset_v2.py` |
| Integridad | `scripts/v2/md5_official.py` |
| Test de paridad | `tests/test_v2_compression_protocol.py` |

## 1. Paridad de protocolo entre las dos ramas

**Las dos ramas entrenan la LSTM con la misma función** (`train_lstm`), a través de `train_branch`,
y con **el mismo objeto de protocolo** (`LSTM_PROTOCOL`). Lo único que cambia es la dimensión de
entrada: k en la rama comprimida, 104 en la rama sin comprimir.

| Parámetro | Valor | Origen |
|---|---|---|
| Optimizador | Adam | v1 |
| Tasa de aprendizaje | 1e-3 | `training/train_world_model.py` (v1) |
| *Weight decay* | 1e-4 | v1 |
| Tamaño de lote | 32 | v1 |
| *Early stopping* | paciencia de 15 épocas sobre la pérdida de validación; se guarda la mejor época | v1 |
| Máximo de épocas | 300 | Experimento 0 rehecho de la v1: un tope que el *early stopping* no debería alcanzar |
| Pérdida | MSE del estado siguiente + 1.0 × MSE de la recompensa normalizada | v1 |
| Normalización de la recompensa | media y desviación del split `train` | v1 |
| Ventana | 16 pasos | v1 |
| Tamaño oculto de la LSTM | 128 | v1 |
| Acción | one-hot por semáforo, 4 × 2 = 8 dimensiones | ver abajo |

El ciclo por época, la pérdida de validación, el escalador de la recompensa y la fijación de
semillas **son las funciones de la v1, importadas y no copiadas**: `train_one_epoch`, `validate`,
`compute_reward_scaler` y `_set_seeds` de `training/train_world_model.py`.

**Verificación automática** (`tests/test_v2_compression_protocol.py`, ya en verde):

1. Las dos ramas llaman a la misma función con el mismo objeto de protocolo, el mismo split y la
   misma semilla. Solo difiere la dimensión de entrada.
2. En el código de `train_lstm`, el *weight decay* y el *early stopping* se aplican desde el
   protocolo, que es justo lo que le faltó a la rama sin comprimir de la v1.
3. Los valores del protocolo son los de la v1.
4. Una corrida real pequeña de cada rama, con datos sintéticos y el mismo protocolo reducido,
   guarda en su `.json` el mismo protocolo, termina por *early stopping* y difiere de la otra solo
   en la dimensión de entrada.

**Codificación de la acción.** La acción de la v2 son 4 decisiones de mantener/cambiar. Se
codifica con un one-hot de 2 clases por semáforo, concatenado: 8 dimensiones, la convención de la
v1 ("acción one-hot, `action_dim = 2`") aplicada a cada semáforo. Para no duplicar la lógica de la
métrica, se agregó a la v1 un único helper, `encode_actions` en `datasets/latent_sequence_dataset.py`,
que usan el dataset de ventanas y `rollout_episode`. Con acciones de una sola dimensión (las de la
v1) el resultado es idéntico a antes, y lo comprueban los tests de la v1.

## 2. Candidatos de tamaño comprimido

**k ∈ {16, 32, 48, 64, 80}.** Son exactamente los tamaños de la v1 (4, 8, 12, 16 y 20 sobre un
estado de 26) multiplicados por 4: el mismo rango de razones de compresión (de 0.15 a 0.77) sobre
un estado 4 veces mayor. Así los resultados son comparables con la exploración de la v1
(`docs/EXPLORACION_LATENT_DIM.md`).

En el extremo bajo, 16 son unas 4 dimensiones por semáforo, compresión agresiva. En el alto, 80
queda por debajo de las 96 columnas no constantes del estado: siempre hay algo que comprimir.

**Arquitectura del Autoencoder:** la misma de la v1, una capa oculta en el codificador y otra en el
decodificador (`models/representation/Autoencoder`), con un ancho oculto de **104**, igual a la
entrada. Así la única restricción es la capa latente, y el ancho oculto es el mismo para los cinco
candidatos: solo cambia k. Entrenamiento como en la v1: Adam con tasa de aprendizaje 1e-3, lote de
32, 100 épocas, guardando la mejor época de validación y con la reconstrucción por MSE.

## 3. Diseño del experimento

**Métrica por modelo:** `reward_mse` en el split `test` (24 episodios), con *rollouts*
autoregresivos con las acciones reales (`rollout_episode` de la v1) en los horizontes h = 1 a 10,
y la recompensa desnormalizada con el escalador de cada corrida. Es la misma métrica de decisión
del Experimento 0 de la v1: la única comparable entre los dos espacios de entrada. El puntaje de
cada modelo es

```
log GM = media sobre h = 1..10 de log(reward_mse_h)       (logaritmo de la media geométrica)
```

**Comparación de un tamaño k con la rama sin comprimir:**

```
Δ = exp( media de log GM de los modelos z  −  media de log GM de los modelos sin comprimir ) − 1
```

Δ < 0 significa que la compresión reduce el error. Su IC del 95% sale de un **bootstrap por
conglomerados en dos niveles**: 10,000 remuestreos con semilla de numpy 0. En cada remuestreo se
sacan Autoencoders con reposición y, dentro de cada uno, sus semillas de LSTM con reposición; la
rama sin comprimir saca sus semillas con reposición.

### 3.1 Fase de selección

| Rama | Diseño | Modelos |
|---|---|---|
| Comprimida | 5 tamaños × 3 Autoencoders (semillas 0–2) × 3 LSTM (semillas 0–2) | 45 |
| Sin comprimir | 3 LSTM (semillas 0–2) | 3 |

- **Regla de selección:** el tamaño con la menor media de log GM. Si hay un empate exacto, el
  menor k.
- Se reportan el Δ y el IC de cada tamaño como descripción. Se considera que un tamaño "muestra
  ventaja clara" en la selección solo si su IC queda entero por debajo de 0, pero **esto no
  decide nada**: la confirmación se corre con el tamaño seleccionado en cualquier caso, para que
  la respuesta final siempre salga del diseño de 5 × 5.

### 3.2 Fase de confirmación (decisoria)

| Rama | Diseño | Modelos |
|---|---|---|
| Comprimida | el tamaño seleccionado × 5 Autoencoders (semillas 100–104) × 5 LSTM (semillas 100–104) | 25 |
| Sin comprimir | 10 LSTM (semillas 100–109) | 10 |

**Las semillas son nuevas:** no se reutiliza ningún modelo de la selección, para no heredar el
sesgo de haber elegido al ganador.

**Decisión, con el IC del 95% de Δ:**

- IC entero por debajo de 0 → **la compresión ayuda** a la LSTM.
- IC entero por encima de 0 → **la compresión empeora**.
- IC que incluye el 0 → **no hay evidencia** de que ayude ni de que empeore. Se reporta así, con
  el valor puntual.

### 3.3 Reportes secundarios (no deciden)

- Δ y su IC por horizonte.
- Las estadísticas de la v1 por horizonte: la fracción de pares (modelo z, modelo sin comprimir)
  que gana z, y la mediana de la reducción relativa.
- La descomposición de la varianza de log GM: entre Autoencoders, dentro de un Autoencoder (entre
  semillas de LSTM) y entre semillas de la rama sin comprimir.
- El error de reconstrucción de validación de cada Autoencoder.
- El *baseline* persistente: la referencia del Experimento 1 de la v1.
- Las épocas de parada y cuántos modelos llegaron al tope de 300.

## 4. Datos

- Los splits `train`, `validation` y `test` del dataset pre-registrado
  (`docs/v2/ADDENDUM_DATASET.md`): 112, 24 y 24 episodios, con 6,720, 1,440 y 1,440
  transiciones. No se regenera nada.
- `scripts/v2/prepare_dataset_v2.py` verifica el SHA-256 de cada episodio contra el manifiesto y
  después los une (`merge_files` de la v1) y los normaliza (`normalize_dataset` de la v1, con el
  escalador ajustado **solo con `train`**). Las salidas van a `datasets/v2/processed/`, fuera de
  git.
- **El split `ood` no se lee en esta fase.** El script no lo incluye, y los datos procesados no
  tienen ningún `episode_id` ≥ 23000.
- **Concentración del error por episodio** (el patrón de la v1): para cada modelo se reporta qué
  fracción del error de recompensa en h = 10 se llevan los 3 peores de los 24 episodios de prueba,
  la mediana por episodio y qué episodios son los peores.

## 5. Las 8 columnas constantes

Se mantienen las 104 columnas, como ya se decidió. `normalize_dataset` sustituye su desviación
estándar nula por 1, así que en los datos normalizados valen 0 siempre. Como dato informativo se
reporta el error de reconstrucción de cada Autoencoder en esas 8 columnas.

## 6. Integridad

- **MD5 antes y después** (`scripts/v2/md5_official.py`) de los archivos oficiales: red y dataset
  de la v1 (crudo y procesado), checkpoints oficiales de la v1 (Autoencoder, LSTM,
  Transformer/TSMixer y los tres PPO), y red, rutas y dataset crudo de la v2. La foto "antes" se
  tomó antes de este commit: `docs/results/v2/autoencoder/md5_before.json`, con 314 archivos.
- Los pesos van a `models/checkpoints/v2/compression/`, una carpeta nueva y fuera de git; sus
  `.json` sí se versionan.
- La suite completa de tests se corre al final.

## 7. Lo que este pre-registro prohíbe

- Cambiar los candidatos, las semillas, el protocolo, la métrica, la regla de selección o el
  criterio de decisión después de ver resultados.
- Usar el split `ood`.
- Descartar modelos. Si alguno termina en un estado anómalo (por ejemplo, llega al tope de
  épocas), se reporta.
- Transformer y TSMixer quedan fuera de esta fase.

## 8. Enmienda posterior a la fase de selección: candidato k = 96

**Escrita después de ver los resultados de la selección y antes de entrenar k = 96.** Se aparta del
pre-registro (sección 7: no cambiar los candidatos después de ver resultados), a pedido explícito
del autor, y se deja escrito.

**Qué se vio** (`docs/results/v2/autoencoder/selection_analysis.json`): Δ decrece de forma
monótona con k (+125%, +89%, +40% y +28% para 16, 32, 48 y 64), y 80 queda en −3.0%, con un IC
de [−15.6%, +9.5%]. 80 es el borde superior de la grilla, así que la tendencia podría seguir
mejorando hasta no comprimir.

**Qué se agrega:** **k = 96**, con el diseño de selección sin cambios: 3 Autoencoders (semillas
0–2) × 3 LSTM (semillas 0–2), y las mismas 3 LSTM sin comprimir de la selección, que no se
reentrenan. Se corre con `--sizes 96 --tag k96` y el resultado va a `selection_k96.json`, sin
tocar los resultados ya commiteados.

**Nota para leerlo.** 96 es exactamente el número de columnas no constantes del estado (104 − 8),
y el Autoencoder tiene una capa oculta de 104. Un Autoencoder con k = 96 podría, en principio,
reconstruir el estado sin pérdida: es una compresión solo nominal, que quita la redundancia de las
8 columnas constantes. Sirve como sonda de "casi no comprimir".

**Regla de selección enmendada** (fijada antes de correr k = 96): entre los seis tamaños
{16, 32, 48, 64, 80, 96}, el de menor media de log GM, con la misma regla de empate. Ese tamaño
pasa a la fase de confirmación, que sigue igual que en 3.2, **sujeto a la aprobación explícita del
autor**. Se reportan los resultados de 80 y de 96 lado a lado.
