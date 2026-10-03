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

### 8.1 Resultado de k = 96

`docs/results/v2/autoencoder/selection_k96.json` y `selection_k96_analysis.json`. Las 3 LSTM sin
comprimir son las mismas de la selección (no se reentrenaron), así que el Δ es directamente
comparable con el de los otros cinco tamaños.

| | k = 80 | k = 96 |
|---|---|---|
| Δ (h = 1..10) | −3.0%, IC [−15.6%, +9.5%] | −0.1%, IC [−9.7%, +11.0%] |
| Media de log GM (z); sin comprimir: 7.4247 | 7.3938 | 7.4238 |
| Δ por Autoencoder (semillas 0, 1, 2) | −14.2%, +9.6%, −3.0% | −6.2%, +0.1%, +6.3% |
| Varianza entre / dentro de Autoencoders | 0.015 / 0.014 | 0.004 / 0.023 |
| MSE de reconstrucción en validación | 0.0024–0.0031 | 0.0023–0.0026 |

- **k = 96 no mejora sobre k = 80.** Los dos IC incluyen el 0: ninguno muestra que la compresión
  ayude ni que empeore, y la diferencia entre ellos (0.03 en log GM) es mucho menor que los IC.
- **La tendencia se aplana entre 80 y 96**: Δ = +125%, +89%, +40%, +28%, −3.0% y −0.1% para
  16, 32, 48, 64, 80 y 96. Comprimir mucho (k ≤ 64) empeora; con poca compresión, el efecto es
  cercano a 0.
- k = 96 no reconstruye sin pérdida: su error de reconstrucción es del orden del de k = 80, y
  ningún Autoencoder llegó al tope de épocas. Ninguna LSTM llegó al tope (pararon entre las
  épocas 99 y 206).
- **Bajo la regla enmendada, el candidato para la confirmación sigue siendo k = 80** (menor media
  de log GM entre los seis tamaños, sin empate). El "selected latent_dim: 96" de
  `selection_k96_analysis.json` es un artefacto del script: ese archivo solo contiene k = 96.

## 9. Repetición de la fase de selección con el Transformer

**Escrita antes de entrenar ningún Transformer de la v2.** A pedido del autor, antes de la
confirmación de k = 80 con la LSTM. Es **descriptiva**: no cambia el candidato de la LSTM ni decide
nada por sí sola. Su resultado se compara con el de la LSTM antes de decidir el paso siguiente.

**Motivo.** En la v1 (exploración del `latent_dim`, `docs/EXPLORACION_LATENT_DIM.md`, Fase 2) el
Autoencoder fue neutro con la LSTM, pero empeoró la predicción de la recompensa con el Transformer
(+33%). Esta repetición pregunta si en el estado de 104 columnas pasa lo mismo.

**Diseño.**

| Rama | Diseño | Modelos nuevos |
|---|---|---|
| Comprimida | 6 tamaños (16, 32, 48, 64, 80, 96) × 3 Autoencoders (semillas 0–2) × 3 Transformers (semillas 0–2) | 54 |
| Sin comprimir | 3 Transformers (semillas 0–2) | 3 |

- **Se reutilizan** los 18 Autoencoders de la selección y sus splits codificados, y los splits sin
  comprimir (`raw_data/`). El Autoencoder se entrena sin mirar al modelo temporal, así que es el
  mismo para los dos.
- **No se reutilizan** las LSTM sin comprimir: la referencia de cada arquitectura es la misma
  arquitectura sobre el estado de 104. Se entrenan 3 Transformers sin comprimir nuevos.
- **Protocolo de entrenamiento idéntico al de la LSTM** (`LSTM_PROTOCOL`, el mismo objeto): Adam,
  lr 1e-3, weight decay 1e-4, lotes de 32, ventana de 16, tope de 300 épocas, paciencia 15, la
  misma pérdida y el mismo `reward_scaler` (solo con el split de entrenamiento). El único camino de
  entrenamiento es `train_temporal_model`, y entre las dos ramas solo cambia la dimensión de
  entrada. Lo verifica `tests/test_v2_compression_protocol.py`.
- **Hiperparámetros propios del Transformer:** los del Experimento 3 de la v1
  (`training/train_world_model_transformer.py`): d_model 128, 4 cabezas, 2 capas, feedforward
  256, dropout 0. Son los mismos que en la Fase 2 de la exploración de la v1, que también usó el
  tope de 300 y la paciencia 15. No se ajustan para la v2.
- **Evaluación y análisis:** los mismos de la sección 3 (`reward_mse` en rollouts de h = 1..10
  con las acciones reales sobre el test, log GM, Δ, IC por bootstrap en dos niveles, varianza
  entre y dentro de Autoencoders). Resultados en `selection_transformer.json` y
  `selection_transformer_analysis.json`. Pesos en `tf_*` dentro de la carpeta de la selección.
- **Se reporta además, como descripción:** el tamaño de menor media de log GM con el Transformer
  (misma regla de empate), la comparación de Δ por tamaño con la LSTM, y el log GM sin comprimir del
  Transformer frente al de la LSTM.
- Ninguna corrida se repite ni se descarta por su resultado. Si alguna llega al tope de épocas,
  se reporta.

### 9.1 Resultado

`docs/results/v2/autoencoder/selection_transformer.json` y `selection_transformer_analysis.json`.
Los 57 Transformers terminaron por early stopping (épocas 35–117); ninguno llegó al tope. La
corrida se interrumpió una vez por falta de memoria del sistema y se reanudó: las 18 corridas que
ya tenían `evaluation.json` no se reentrenaron, y las 4 que estaban a medias se entrenaron de
cero.

Δ de cada arquitectura frente a ella misma sin comprimir (Δ < 0: la compresión ayuda):

| k | LSTM: Δ, IC 95% | Transformer: Δ, IC 95% | Transformer: varianza entre / dentro de AE |
|---|---|---|---|
| 16 | +124.9% [+100.5, +151.8] | +111.6% [+77.1, +151.2] | 0.0079 / 0.026 |
| 32 | +88.9% [+67.1, +109.5] | +68.2% [+41.2, +99.6] | 0.0077 / 0.025 |
| 48 | +40.1% [+16.1, +64.3] | +47.1% [+25.5, +70.2] | 0.0005 / 0.024 |
| 64 | +27.8% [+19.3, +36.9] | +25.5% [+3.9, +52.3] | 0.0129 / 0.031 |
| 80 | −3.0% [−15.6, +9.5] | +16.9% [−1.0, +37.6] | 0.0012 / 0.034 |
| 96 | −0.1% [−9.7, +11.0] | +19.3% [+0.1, +41.8] | 0.0004 / 0.047 |

- **Con el Transformer la compresión empeora en todos los tamaños.** El daño baja con k, pero se
  estabiliza en +17–19% en 80 y 96, en lugar de llegar a 0 como con la LSTM. En k = 80 el IC
  apenas incluye el 0; en k = 96 apenas lo excluye. El Δ es positivo en los 10 horizontes para
  los dos tamaños, y los tres Autoencoders lo empeoran por separado (k = 80: +13%, +21%, +17%;
  k = 96: +18%, +18%, +22%). La varianza está casi toda dentro de cada Autoencoder (entre las
  semillas del Transformer), no entre Autoencoders.
- **Replica lo visto en la v1** (exploración del `latent_dim`, Fase 2): el latente es neutro con
  la LSTM y perjudica al Transformer.
- Tamaño de menor media de log GM con el Transformer: **k = 80** (7.4105, frente a 7.4309 en 96).
  Es descriptivo: no cambia el candidato de la LSTM.
- **Sin comprimir, el Transformer tuvo menor error que la LSTM** (media de log GM 7.2542 frente a
  7.4247; Δ = −15.7%; log GM por semilla 7.13, 7.40 y 7.22 frente a 7.40, 7.45 y 7.42). Es la
  inversa del orden de la v1. Con 3 semillas por lado y una semilla del Transformer casi al nivel
  de la LSTM, es una **brecha no significativa** (Welch sobre los log GM por semilla: t = −2.11,
  gl ≈ 2.1, p = 0.16). Esta selección no se diseñó para comparar arquitecturas; afirmarlo exigiría
  una comparación propia, preregistrada y con más semillas.

## 10. Repetición de la fase de selección con TSMixer

**Escrita antes de entrenar ningún TSMixer de la v2.** A pedido del autor, después de la sección 9
y antes de las decisiones pendientes sobre el Transformer y de la confirmación de k = 80 con la
LSTM. Es **descriptiva**, como la sección 9: no cambia el candidato de la LSTM ni decide nada por
sí sola, y no compara arquitecturas.

**Motivo.** Completar las tres arquitecturas de la v1. En la exploración del `latent_dim` de la
v1 (Fase 2), el latente empeoró la predicción de la recompensa con TSMixer (+42%), más que con el
Transformer (+33%).

**Diseño.**

| Rama | Diseño | Modelos nuevos |
|---|---|---|
| Comprimida | 6 tamaños (16, 32, 48, 64, 80, 96) × 3 Autoencoders (semillas 0–2) × 3 TSMixer (semillas 0–2) | 54 |
| Sin comprimir | 3 TSMixer (semillas 0–2) | 3 |

- **Se reutilizan** los 18 Autoencoders de la selección y sus splits codificados, y los splits sin
  comprimir (`raw_data/`). **No se reutilizan** los modelos sin comprimir de la LSTM ni del
  Transformer: la referencia de TSMixer es TSMixer sobre el estado de 104.
- **Protocolo de entrenamiento idéntico al de la LSTM y el Transformer** (`LSTM_PROTOCOL`, el
  mismo objeto): Adam, lr 1e-3, weight decay 1e-4, lotes de 32, ventana de 16, tope de 300 épocas,
  paciencia 15, la misma pérdida y el mismo `reward_scaler` (solo con el split de entrenamiento).
  Mismo camino de entrenamiento (`train_temporal_model`); entre las dos ramas solo cambia la
  dimensión de entrada. Lo verifica `tests/test_v2_compression_protocol.py`.
- **Hiperparámetros propios de TSMixer:** los del Experimento 3 de la v1
  (`training/train_world_model_tsmixer.py`): dimensión oculta 128 (el valor por defecto de
  `WorldModelConfig`), 2 bloques, dropout 0. No se ajustan para la v2. El tope de 300 importa
  aquí: en la v1, TSMixer no convergía en 100 épocas.
- **Evaluación y análisis:** los mismos de la sección 3 (`reward_mse` en rollouts de h = 1..10
  con las acciones reales sobre el test, log GM, Δ, IC por bootstrap en dos niveles, varianza
  entre y dentro de Autoencoders). Resultados en `selection_tsmixer.json` y
  `selection_tsmixer_analysis.json`. Pesos en `tsm_*` dentro de la carpeta de la selección.
- **Se reporta además, como descripción:** el tamaño de menor media de log GM con TSMixer (misma
  regla de empate) y la comparación de Δ por tamaño con la LSTM y el Transformer. No se compara
  el error absoluto entre arquitecturas: eso queda para una comparación propia, si se decide.
- Ninguna corrida se repite ni se descarta por su resultado. Si alguna llega al tope de épocas,
  se reporta. Si la corrida se interrumpe (por ejemplo, por memoria), se reanuda sin reentrenar
  las corridas que ya tengan `evaluation.json`, y se reporta.
