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
