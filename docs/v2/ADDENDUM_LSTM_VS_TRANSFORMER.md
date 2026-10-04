# Pre-registro: LSTM frente a Transformer sin comprimir, con presupuesto de parámetros igualado (v2)

**Escrito el 3 de octubre de 2026, antes de entrenar ninguno de los 20 modelos.** Experimento
independiente de la Fase 2 (`ADDENDUM_AUTOENCODER.md`): no usa ningún Autoencoder y no cambia nada
de lo decidido o pendiente en ella.

## 1. Pregunta y motivo

¿Qué modelo temporal usa la v2: la LSTM o el Transformer? Los dos sobre el estado de 104
dimensiones, sin comprimir, **con el mismo número de parámetros** (≈ 152,000).

En la selección de la Fase 2 (sección 9.1 del addendum de la Fase 2), el Transformer sin comprimir
tuvo menor error que la LSTM (media de log GM 7.2542 frente a 7.4247; −15.7%). Con 3 semillas por
lado, es una brecha no significativa (Welch, p = 0.16). Además, ese Transformer tenía el doble de
parámetros que la LSTM (292,969 frente a 137,449), así que la brecha podía venir solo del tamaño.
Aquí se iguala el presupuesto y se usan 10 semillas por lado, para no decidir con pocas
repeticiones.

> **Nota sobre los tamaños.** Esta comparación usa presupuestos de parámetros igualados:
> **capa oculta de 137 para la LSTM y d_model de 80 para el Transformer**. Son deliberadamente
> distintos de los tamaños heredados de la v1 (128 y 128), que se usaron en toda la Fase 2 del
> Autoencoder y que esa fase conserva. Reemplazan a los de la v1 **solo en este experimento**, para
> comparar arquitecturas con el mismo presupuesto, siguiendo la misma práctica que el trabajo con 7
> intersecciones.
>
> Por eso, el resultado responde **"¿qué arquitectura es mejor con el mismo número de
> parámetros?"**, no "¿qué arquitectura es mejor en su configuración por defecto?".
>
> Si la arquitectura ganadora se usa más adelante en la confirmación de la Fase 2, esa confirmación
> debe **pre-registrar sus propios tamaños** y aclarar con qué presupuesto se hizo.

## 2. Tamaños y conteo de parámetros

Entrada de los dos modelos: 104 (estado) + 8 (acción) = 112 por paso. Fórmulas exactas, verificadas
contra `sum(p.numel())` de los modelos reales (`tests/test_v2_lstm_vs_transformer.py`):

- **LSTM** (`LatentDynamicsLSTM`, 1 capa; PyTorch usa dos vectores de sesgo por compuerta):
  4h(112 + h) + 8h (LSTM) + 104h + 104 (cabeza latente) + h + 1 (cabeza de recompensa)
  = **4h² + 561h + 105**.
- **Transformer** (`LatentDynamicsTransformer`, 4 cabezas, 2 capas, feedforward 256, sin dropout):
  proyección de entrada 112d + d; por capa, atención 4d² + 4d, feedforward 256d + 256 + 256d + d y
  dos LayerNorm 4d; cabezas 104d + 104 + d + 1. Total = **8d² + 1260d + 617**. La codificación
  posicional no tiene parámetros.

| Configuración | Parámetros | Frente a 150,000 |
|---|---|---|
| LSTM h = 128 (v1, Fase 2) | 137,449 | −8.4% |
| Transformer d = 128 (v1, Fase 2) | 292,969 | +95.3% |
| **LSTM h = 137 (este experimento)** | **152,038** | +1.4% |
| **Transformer d = 80 (este experimento)** | **152,617** | +1.7% |

- **Transformer:** 8d² + 1260d + 617 = 150,000 da d ≈ 79.0. d tiene que ser divisible entre las 4
  cabezas: los múltiplos de 4 cercanos son 76 (142,585; −4.9%) y **80** (152,617; +1.7%). Con 80,
  cada cabeza tiene 20 dimensiones (80 / 4 = 20).
- **LSTM:** 4h² + 561h + 105 = 150,000 da h ≈ 135.8. h = 136 da 150,385, el más cercano a 150,000,
  pero queda a 1.5% del Transformer. Se elige **h = 137** (152,038), que queda a **0.4%** (579
  parámetros) del Transformer: lo que importa es igualar las dos arquitecturas entre sí. La LSTM no
  tiene restricción de divisibilidad.
- El feedforward del Transformer se mantiene en 256, como en la v1: su ancho pasa de 2 × d a
  3.2 × d.
- El script de entrenamiento se niega a correr si los modelos no tienen exactamente estos conteos,
  y el análisis comprueba el conteo de los pesos guardados de cada corrida.

## 3. Diseño

| Arquitectura | Tamaño | Semillas | Carpetas |
|---|---|---|---|
| LSTM | h = 137 | 0–9 | `models/checkpoints/v2/arch_comparison/lstm_raw_s{0..9}` |
| Transformer | d = 80 | 0–9 | `models/checkpoints/v2/arch_comparison/transformer_raw_s{0..9}` |

- **Los 20 modelos se entrenan desde cero.** No se reutilizan las semillas 0–2 de la selección de
  la Fase 2: tienen los tamaños de la v1.
- **Mismo protocolo de entrenamiento que la Fase 2**, el mismo objeto `LSTM_PROTOCOL`, por el mismo
  camino (`train_branch("raw", ...)` → `train_temporal_model`): Adam, lr 1e-3, weight decay 1e-4,
  lotes de 32, ventana de 16, tope de 300 épocas, paciencia 15, la misma pérdida y el
  `reward_scaler` del split de entrenamiento. Mismos splits sin comprimir (`selection/raw_data/`).
  Solo cambian los hiperparámetros de arquitectura, que se pasan como `architecture_hparams`. Los
  valores por defecto de la Fase 2 (`LSTM_PROTOCOL.hidden_dim` = 128, `TRANSFORMER_HPARAMS`) no se
  tocan.
- El análisis comprueba que las 20 corridas registran el mismo protocolo y los tamaños de esta
  sección, y guarda el md5 de los pesos de cada una.
- Script de entrenamiento: `scripts/v2/run_lstm_vs_transformer.py` (4 procesos por defecto).
  Análisis: `scripts/v2/analyze_lstm_vs_transformer.py` → `docs/results/v2/arch_comparison/analysis.json`.

## 4. Métrica y pruebas

**Métrica por modelo:** log GM, el logaritmo de la media geométrica del `reward_mse` sobre los
horizontes 1 a 10, en el split de test (rollouts autorregresivos con las acciones reales). Es la
métrica de toda la Fase 2.

**Diferencia:** d = media de log GM (Transformer) − media de log GM (LSTM). Se reporta también como
relativa, exp(d) − 1. d < 0 significa que el Transformer predice mejor.

**Las semillas no se emparejan.** La semilla N de la LSTM y la semilla N del Transformer no
comparten ninguna fuente de aleatoriedad: son arquitecturas distintas, con inicializaciones
distintas. El número de semilla es solo una etiqueta (la misma salvedad de la v1). Por eso:

- **Welch** sobre los 10 log GM de cada lado: t, grados de libertad, p bilateral y el IC del 95% de d.
- **Bootstrap percentil** del IC del 95% de d: 10,000 remuestreos con semilla de numpy 0. En cada
  uno se sacan con reposición las 10 semillas de cada arquitectura, por separado.

La distribución t se calcula en el propio script, porque scipy no está en el entorno. Hay tests
contra valores de tabla (`tests/test_v2_lstm_vs_transformer.py`).

## 5. Criterio de decisión (fijado ahora)

- **Los dos IC del 95% de d (Welch y bootstrap) excluyen el 0 del mismo lado** → se adopta como
  modelo temporal oficial de la v2 la arquitectura de menor error, con el tamaño de este
  experimento.
- **En cualquier otro caso** (alguno de los dos incluye el 0) → **"sin evidencia suficiente"**. Se
  reporta así, con el valor puntual, y se documenta como limitación. No se fuerza una elección.

Se exigen los dos intervalos porque, con 10 semillas por lado, el bootstrap percentil tiende a dar
intervalos algo más estrechos de lo debido. Pedir que coincida con Welch evita decidir por esa
diferencia.

**Alcance de la decisión.** Decide el modelo temporal de la v2 sobre el estado sin comprimir, con
presupuesto igualado (nota de la sección 1). No responde la pregunta de la Fase 2 (si comprimir
ayuda), que sigue pendiente de sus propias decisiones.

## 6. Reportes secundarios (no deciden)

- **Por horizonte (h = 1..10):** d, Welch y bootstrap para cada horizonte, para ver si la ventaja
  es pareja o se concentra en algún rango. Son 10 pruebas sin corrección por comparaciones
  múltiples, así que se leen como descripción.
- **Como contexto:** épocas de convergencia de cada arquitectura (mejor época y época de parada), y
  la variabilidad entre semillas (desviación estándar del log GM de cada una y el cociente de
  varianzas).
- **Como contexto, fuera de la comparación:** los modelos de 128 de la selección de la Fase 2 (3
  semillas por lado), para ver cuánto cambia cada arquitectura con el nuevo tamaño.
- **Integridad:** md5 de los archivos oficiales antes y después (`scripts/v2/md5_official.py`,
  etiquetas `lstm_vs_transformer_before` / `_after`).

## 7. Reglas

- Ninguna corrida se repite ni se descarta por su resultado. Si alguna llega al tope de 300
  épocas, se reporta.
- Si el entrenamiento se interrumpe (por ejemplo, por memoria), se reanuda sin reentrenar las
  corridas que ya tengan `evaluation.json`, y se reporta.

## 8. Resultado

`docs/results/v2/arch_comparison/analysis.json`. Los 20 modelos se entrenaron sin interrupciones
(92 minutos con 4 procesos). Todos terminaron por early stopping; ninguno llegó al tope de 300
épocas. Las 20 corridas registran el mismo protocolo y los tamaños de la sección 2 (lo comprueba
el análisis sobre los pesos guardados).

### 8.1 Criterio de decisión

| | LSTM h = 137 | Transformer d = 80 |
|---|---|---|
| Media de log GM (10 semillas) | 7.5022 | 7.3863 |
| Desviación estándar del log GM | 0.182 | 0.145 |
| GM `reward_mse` por semilla | 1545, 2394, 1756, 1822, 1816, 2333, 1321, 1998, 1827, 1577 | 1880, 1625, 1532, 1519, 1563, 1565, 2198, 1680, 1309, 1425 |

- d = −0.116: el Transformer tiene un error **10.9% menor** como valor puntual.
- **Welch:** t = −1.58, gl = 17.1, **p = 0.13**, IC del 95% [−23.7%, +4.0%].
- **Bootstrap:** IC del 95% [−22.1%, +2.5%].
- **Los dos IC incluyen el 0 → "sin evidencia suficiente".** Según el criterio de la sección 5,
  no se adopta ninguna arquitectura como oficial de la v2. Es una brecha no significativa a favor
  del Transformer, y queda como limitación.

### 8.2 Por horizonte (descriptivo, sin corrección por comparaciones múltiples)

| h | Transformer − LSTM | IC Welch | IC bootstrap | p (Welch) |
|---|---|---|---|---|
| 1 | −33.7% | [−38.9, −28.0] | [−38.2, −28.5] | 1.1e-8 |
| 2 | −29.9% | [−36.3, −22.9] | [−35.6, −23.8] | 5.5e-7 |
| 3 | −18.8% | [−28.6, −7.6] | [−27.3, −9.1] | 0.0034 |
| 4 | −11.1% | [−25.1, +5.5] | [−23.3, +3.3] | 0.16 |
| 5 | −3.3% | [−20.0, +17.0] | [−17.9, +14.3] | 0.71 |
| 6 | −1.0% | [−18.5, +20.2] | [−16.3, +18.0] | 0.91 |
| 7 | +0.2% | [−18.0, +22.4] | [−15.6, +20.8] | 0.98 |
| 8 | +1.4% | [−18.4, +26.0] | [−15.5, +24.8] | 0.89 |
| 9 | −0.4% | [−20.1, +24.3] | [−17.0, +23.1] | 0.97 |
| 10 | −3.6% | [−21.9, +19.1] | [−19.3, +17.6] | 0.72 |

La ventaja del Transformer **no es pareja: se concentra en los horizontes cortos.** En h = 1–3 es
grande y clara (−34% a −19%, p ≤ 0.0034; sobreviviría a una corrección de Bonferroni por 10
pruebas, 0.005). Desde h = 5 la diferencia es prácticamente 0. La métrica de decisión promedia los
10 horizontes en log, así que la ventaja de los horizontes cortos se diluye en el promedio.

### 8.3 Contexto (no forma parte del criterio)

- **Convergencia:** el Transformer converge antes. Mejor época media 89 frente a 110; parada en
  las épocas 79–142 (mediana 96) frente a 77–188 (mediana 116.5) de la LSTM. Las dos semillas de la
  LSTM con más error (1: 2394; 5: 2333) son también las que pararon antes (épocas 78 y 77).
- **Variabilidad entre semillas:** el Transformer varía algo menos (desviación estándar 0.145
  frente a 0.182; cociente de varianzas 0.63). Con 10 semillas por lado, no es una diferencia
  clara.
- **Frente a los tamaños de la v1 (selección de la Fase 2, 3 semillas por lado):** media de log GM
  7.4247 para la LSTM de 128 y 7.2542 para el Transformer de 128. Las dos arquitecturas quedan peor
  con el tamaño igualado de este experimento (7.5022 y 7.3863). Para el Transformer, d = 80 tiene
  la mitad de parámetros. Para la LSTM, h = 137 tiene un 11% más, y aun así su media es peor, lo
  que sugiere que las 3 semillas de la selección cayeron del lado bajo de su variabilidad (sus
  log GM, 7.40–7.45, quedan dentro del rango de las 10 de aquí, 7.19–7.78). Son diseños y tamaños
  distintos: no se comparan formalmente.
- **Integridad:** el md5 de los 314 archivos oficiales es igual antes y después
  (`md5_lstm_vs_transformer_before.json` / `_after.json`). Los pesos y evaluaciones de las 6
  corridas sin comprimir de la selección de la Fase 2 tampoco cambiaron. Tests: 107 passed. Los 12
  de SUMO en vivo siguen sin poder correr por el bloqueo de Smart App Control (PROJECT_STATUS.md,
  "Verificación de tests al commit del 3 de octubre").

## 9. Extensión a TSMixer, con presupuesto igualado y corrección de Bonferroni

**Escrita el 4 de octubre de 2026, antes de entrenar ningún TSMixer de este experimento.** Las
secciones 1–8 y sus resultados (`analysis.json`) no se reescriben. `analysis.json` se regeneró con
el código actualizado y salió idéntico byte a byte.

### 9.1 Tamaño

TSMixer (`LatentDynamicsTSMixer`) con 2 bloques y dropout 0, como en la v1; solo cambia la
dimensión oculta H de la mezcla de variables. Entrada de 112 por paso (104 + 8), ventana L = 16:

- Por bloque: dos LayerNorm (2 · 2 · 112 = 448), mezcla temporal L² + L = 272 y mezcla de variables
  112H + H + 112H + 112 = 225H + 112. Total por bloque: 225H + 832.
- Cabezas: 112 · 104 + 104 (latente) y 112 + 1 (recompensa) = 11,865.
- **Total = 2(225H + 832) + 11,865 = 450H + 13,529.** Verificado contra `sum(p.numel())` del modelo
  real (`tests/test_v2_lstm_vs_transformer.py`).

| H | Parámetros | Frente a la LSTM (152,038) | Frente al Transformer (152,617) |
|---|---|---|---|
| 128 (v1) | 71,129 | −53.2% | −53.4% |
| 307 | 151,679 | −0.24% | −0.61% |
| **308** | **152,129** | **+0.06%** | **−0.32%** |
| 309 | 152,579 | +0.36% | −0.02% |

Cualquier H entre 303 y 314 queda a menos de 2% de los dos. Se elige **H = 308 (152,129
parámetros)**: es el más cercano al punto medio de los otros dos (152,328), y su mayor distancia a
cualquiera de ellos (488 parámetros) es la menor de todas las opciones. Como en la sección 2, el
script se niega a entrenar si el conteo no es exactamente 152,129, y el análisis comprueba el
conteo de los pesos guardados.

### 9.2 Diseño

- **10 semillas (0–9), todas desde cero**, en `models/checkpoints/v2/arch_comparison/tsmixer_raw_s{0..9}`.
  Las 20 corridas de LSTM y Transformer no se reentrenan ni se tocan: el script solo entrena las
  arquitecturas que se le piden (`--architectures tsmixer`) y salta toda corrida que ya tenga
  `evaluation.json`.
- **El mismo `LSTM_PROTOCOL`** por el mismo camino (`train_branch("raw", ...)` →
  `train_temporal_model` con `architecture_hparams`), los mismos splits sin comprimir, la misma
  métrica (log GM sobre h = 1..10 del test). El análisis comprueba que las 30 corridas registran el
  mismo protocolo.

### 9.3 Comparaciones, corrección y criterio

Tres comparaciones por pares, sin emparejar por semilla (como en la sección 4):

1. Transformer − LSTM (la de las secciones 1–8, reportada otra vez con la corrección).
2. TSMixer − LSTM.
3. TSMixer − Transformer.

- **Corrección de Bonferroni desde el inicio:** cada IC es del **98.33%** (1 − 0.05/3), en Welch y
  en el bootstrap percentil (10,000 remuestreos, semilla de numpy 0, cada arquitectura por separado).
- **Criterio por par, igual al de la sección 5 con los IC corregidos:** si los dos IC del 98.33%
  excluyen el 0 del mismo lado, gana el par la arquitectura de menor error. Si no, "sin evidencia
  suficiente" en ese par.
- **Criterio global:** una arquitectura se adopta como modelo temporal oficial de la v2 solo si
  gana **sus dos** pares. Si ninguna gana sus dos pares, el resultado global es "sin evidencia
  suficiente", y se reporta qué pares sí se resolvieron (por ejemplo, si una arquitectura queda
  descartada por perder sus dos pares).
- **La conclusión de las secciones 1–8 se mantiene como se registró** (al 95%). Con la corrección,
  el IC de Transformer − LSTM solo puede ensancharse, así que ese par no puede pasar a resolverse.
- La nota de la sección 1 aplica igual: el resultado responde "¿qué arquitectura es mejor con el
  mismo número de parámetros?".

### 9.4 Regla de avance a la fase de control (fijada ahora)

- **Pasan a la fase de control todas las arquitecturas que no sean significativamente peores que
  alguna otra**, con el mismo criterio por pares de 9.3: los dos IC del 98.33% (Welch y bootstrap)
  excluyen el 0 del mismo lado.
- Una arquitectura significativamente peor que otra (pierde al menos uno de sus pares) **queda
  fuera de la fase de control**, y se reporta como tal.
- Si ningún par muestra diferencia significativa, **pasan las tres**.
- Esta regla decide **quién se prueba en control**. No declara a nadie arquitectura oficial: eso se
  decide con el resultado de control real.
- Relación con el criterio global de 9.3: si una arquitectura gana sus dos pares, las otras dos son
  significativamente peores que ella, así que solo esa pasa a control. Las dos reglas coinciden.

### 9.5 Reportes secundarios (no deciden)

- Las tres arquitecturas lado a lado: media de log GM, GM por semilla, media de log `reward_mse`
  por horizonte, y los tres pares por horizonte (h = 1..10, IC del 98.33%, sin corrección
  adicional por los 10 horizontes: descriptivo).
- Épocas de convergencia (mejor época y parada) y variabilidad entre semillas (desviación estándar
  del log GM y cocientes de varianzas).
- Integridad: md5 de los archivos oficiales antes y después (etiquetas `tsmixer_matched_before` /
  `_after`), y md5 de los pesos y evaluaciones de las 20 corridas de LSTM y Transformer.
- Análisis: `scripts/v2/analyze_three_architectures.py` → `docs/results/v2/arch_comparison/analysis_three_architectures.json`.
- Ninguna corrida se repite ni se descarta por su resultado. Si alguna llega al tope de 300 épocas,
  se reporta. Si el entrenamiento se interrumpe, se reanuda sin reentrenar lo que ya tenga
  `evaluation.json`, y se reporta.
