# Resultados para el artículo (v2, corredor de 4 intersecciones)

Informe de **solo lectura**, generado el 2026-10-05 con `scripts/v2/make_article_report.py` a partir de los archivos de resultados versionados de la rama `v2/four-intersections`. No se simuló ni se entrenó nada para producirlo, y no modifica ningún resultado. Cada tabla indica el archivo de donde sale. Los valores calculados aquí (por ejemplo, las pruebas por intersección de la sección 1) usan las funciones estadísticas del proyecto (`paired_t` de `scripts/evaluate_multiseed_statistical.py` y el Wilcoxon de `docs/results/ppo_10_seeds/analyze.py`). Convención: retorno = suma de la recompensa por paso; más cerca de 0 es mejor. El signo de cada diferencia se indica en su tabla.

## 1. Escenario

**Red.** Corredor de 4 intersecciones en línea (A0, B0, C0, D0, de oeste a este) sobre una arterial Este–Oeste, cada una con su calle transversal Norte–Sur; un carril por sentido, 13.89 m/s; parámetros de SUMO y sumo-rl iguales a la v1 (`delta_time` 5 s, amarillo 2 s, verde mínimo 5 s, episodios de 300 s = 60 pasos de control). Fuente: `docs/v2/DISENO_RED_4_INTERSECCIONES.md`, secciones 1 y 2.

**Demanda final (candidata `it5`), en veh/h.** Llegadas `poisson`. Patrón pulsado de dos mitades de 150 s que se repite cada 300 s. Fuente: `docs/results/v2/demand_calibration/it5_offset_300s.json`, `docs/v2/DISENO_RED_4_INTERSECCIONES.md (3.4)`.

| Segmento | Arterial desde el oeste (→E) | Arterial desde el este (→O) | A0 N/S | B0 N/S | C0 N/S | D0 N/S |
|---|---|---|---|---|---|---|
| 0–150 s | 750 | 250 | 200/120 | 40/30 | 200/150 | 40/30 |
| 150–300 s | 450 | 550 | 200/120 | 40/30 | 500/350 | 40/30 |
| **Media** | 600 | 400 | 200/120 | 40/30 | 350/250 | 40/30 |

- Demanda total media: 2,060 veh/h. Giros desde la arterial: 3.75% hacia el Norte y otro tanto hacia el Sur en cada intersección; en las transversales, 60% sigue recto y el resto gira hacia la arterial, mitad a cada lado.
- **Desfase del pulso:** cada episodio arranca el patrón en `offset = default_rng(semilla).integers(0, 300)`; la demanda en el instante t es la del patrón en (t + offset) mod 300 (`scripts/v2/corridor_demand.py`, `pulse_offset`; `DISENO_RED_4_INTERSECCIONES.md`, 7.1). Solo C0 y la arterial varían entre mitades; A0, B0 y D0 tienen transversal constante.
- **Transversal de C0 (pulsada, N/S):** dos niveles de 150 s cada uno, 200/150 y 500/350 veh/h; su media es 350/250 veh/h, que es la cifra de la fila «Media» y **no** un flujo que ocurra en ningún momento del episodio. En A0 (200/120), B0 (40/30) y D0 (40/30) el flujo es constante. Fuente: `docs/results/v2/demand_calibration/it5_offset_300s.json`. (`spec.segments[].cross.C0` y `spec.cross.C0`)

**Criterios de calibración** (fijados antes de la primera candidata; 20 semillas de calibración 11000–11019; `DISENO_RED_4_INTERSECCIONES.md`, 3.1). Valores en `docs/results/v2/demand_calibration/it5_offset_300s.json` → `criteria`:

1. Sin cola invisible: con la mejor referencia, como máximo 5 vehículos pendientes de inserción después de los primeros 30 s, en todos los episodios.
2. Demora no trivial: la mejor referencia le gana a **cada** política trivial con reducción media pareada ≥ 10% y p < 0.01 (t pareada y Wilcoxon).
3. Recompensa estilo v1 no trivial: mejora frente a cada trivial con p < 0.01 (t pareada).

**Resultado agregado con desfase aleatorio** (mejor referencia `cola_mas_larga` frente a la mejor trivial `min_verde_y_cambiar`, 20 semillas). Fuente: `docs/results/v2/demand_calibration/it5_offset_300s.json`.

| Métrica | Reducción media pareada | t pareada p | Wilcoxon p | Semillas ganadas | ¿Pasa? |
|---|---|---|---|---|---|
| Demora | +15.7% | 7.1e−4 | 0.001 | 17/20 | sí |
| Recompensa estilo v1 | +5.5% | 0.44 | 0.19 | 13/20 | no |

- Cola invisible: máximo de pendientes después de 30 s con `cola_mas_larga` = 4.0 (`summary.cola_mas_larga.max_pending_after_warmup_max`); criterio 1 cumplido: sí. `accepted` = False.
- La demanda se adoptó **con el criterio 3 evaluado por intersección**, por decisión del autor tomada **después de ver los datos** (`DISENO_RED_4_INTERSECCIONES.md`, 7.11).

**Por intersección** (criterio 3 desagregado): recompensa estilo v1 de `cola_mas_larga` menos `min_verde_y_cambiar` (> 0: la referencia es mejor), 20 semillas, desfase aleatorio. Calculado aquí a partir de `per_seed[].gap_per_signal` de `docs/results/v2/demand_calibration/analysis_reference_failures_it5_it6_it7.json` (clave `it5`).

| Intersección | Ventaja media | Mediana | Semillas ganadas | t pareada p | Wilcoxon p | Peor semilla |
|---|---|---|---|---|---|---|
| A0 | 160.4 | 179.0 | 17/20 | 1.0e−4 | 7.8e−4 | −102 |
| B0 | 63.9 | 63.0 | 14/20 | 0.33 | 0.11 | −717 |
| C0 | −327.4 | −259.0 | 3/20 | 4.5e−4 | 4.5e−4 | −1,282 |
| D0 | 179.7 | 141.0 | 17/20 | 4.1e−4 | 3.9e−4 | −147 |

## 2. Dataset

**Splits y semillas** (cada semilla fija la de SUMO, el desfase del pulso y el generador de la política aleatoria; una recolección nueva da el mismo dataset). Fuente: `docs/results/v2/dataset/manifest.json`, `docs/v2/ADDENDUM_DATASET.md`.

| Split | Semillas | Episodios | `aleatoria` | `fijo_2_3` | `cola_mas_larga` | Transiciones |
|---|---|---|---|---|---|---|
| `train` | 20000–20111 | 112 | 38 | 37 | 37 | 6,720 |
| `validation` | 21000–21023 | 24 | 8 | 8 | 8 | 1,440 |
| `test` | 22000–22023 | 24 | 8 | 8 | 8 | 1,440 |
| `ood` | 23000–23029 | 30 | 10 | 10 | 10 | 1,800 |

- Políticas de recolección: `aleatoria` (cada bit mantener/cambiar con probabilidad 1/2, generador sembrado con la semilla del episodio), `fijo_2_3` (2 pasos en la fase 0, transversal Norte/Sur, y 3 en la fase 1, arterial; el mismo programa en los 4 semáforos) y `cola_mas_larga` (verde a la fase con más vehículos detenidos en sus carriles de entrada), asignadas en turno rotativo dentro de cada split. `ood` se reservó desde el principio; `train`, `validation` y `test` se usaron para el modelo del mundo.
- Por política (los 190 episodios; `docs/results/v2/dataset/checks.json` → `per_policy`):

| Política | Episodios | Retorno medio | Mínimo | Máximo | Cambios por semáforo (A0/B0/C0/D0) |
|---|---|---|---|---|---|
| `aleatoria` | 64 | −5,383.4 | −12,475 | −2,168 | 19.2 / 19.7 / 19.4 / 19.7 |
| `fijo_2_3` | 63 | −1,619.6 | −3,593 | −827 | 23.0 / 23.0 / 23.0 / 23.0 |
| `cola_mas_larga` | 63 | −1,118.7 | −2,147 | −467 | 18.6 / 7.7 / 20.1 / 8.1 |

**Estado: 104 dimensiones = 4 semáforos × 26** (bloque del semáforo k en las columnas 26k…26k+25, orden A0, B0, C0, D0). Cada bloque es el `TrafficState` de la v1 (`environments/traffic_state.py`, `to_vector`):

| Columnas del bloque | Variable | Qué es (por carril de entrada, 4 carriles) |
|---|---|---|
| 0–3 | `vehicle_counts` | vehículos en el carril |
| 4–7 | `queue_lengths` | vehículos detenidos (cola) |
| 8–11 | `waiting_times` | tiempo de espera acumulado de los vehículos del carril (s) |
| 12–15 | `mean_speeds` | velocidad media en el carril |
| 16–19 | `occupancies` | ocupación del carril |
| 20–23 | `phase_one_hot` | fase verde actual en one-hot de 4 posiciones; solo se activan 2 (la lógica de sumo-rl tiene 2 fases verdes) |
| 24 | `elapsed_phase_time` | segundos desde el último cambio de fase |
| 25 | `remaining_phase_time` | `min_green − elapsed`, acotado en 0 |

**Las 8 columnas constantes** (`docs/results/v2/dataset/checks.json` → `constant_columns`): `A0.phase_one_hot[2]`, `A0.phase_one_hot[3]`, `B0.phase_one_hot[2]`, `B0.phase_one_hot[3]`, `C0.phase_one_hot[2]`, `C0.phase_one_hot[3]`, `D0.phase_one_hot[2]`, `D0.phase_one_hot[3]`. Son las 2 posiciones del one-hot de fase que nunca se activan, en cada uno de los 4 semáforos. El escalador les da desviación 1 (`scripts/normalize_dataset.py`).

## 3. Autoencoder (Fase 2: ¿comprimir el estado ayuda al modelo temporal?)

**Arquitectura y protocolo** (`models/representation/`; `training/v2_compression_experiment.py`, `AutoencoderProtocol`; los valores salen de `autoencoders[].protocol` en `docs/results/v2/autoencoder/selection.json`):

- Encoder: Linear(104 → 104) + RELU + Linear(104 → k); decoder simétrico. La capa latente k es el único cuello de botella.
- Entrenamiento: MSE de reconstrucción sobre el estado normalizado, Adam, tasa 0.001, lote 32, 100 épocas fijas, se conserva la mejor época de validación; semillas de Autoencoder 0, 1 y 2.
- Modelo temporal sobre z o sobre el estado crudo con el mismo protocolo (Adam 1e-3, weight decay 1e-4, lote 32, tope 300 épocas, early stopping con paciencia 15, ventanas de 16 pasos); 3 semillas por Autoencoder y 3 en la rama cruda.
- **Métrica:** log de la media geométrica del `reward_mse` de rollouts autoregresivos en h = 1…10 sobre el test del dataset (22000–22023). **Δ = exp(media(log GM z) − media(log GM crudo)) − 1** (diferencia relativa; Δ > 0: comprimir **empeora**). IC al 95% por bootstrap por conglomerados de dos niveles (Autoencoders y, dentro de cada uno, semillas del modelo temporal; 10,000 remuestreos, semilla 0). Definición en `scripts/v2/analyze_compression_experiment.py`; `delta` y `ci95` de los JSON ya son relativos.

**Error de reconstrucción del Autoencoder** (MSE de validación sobre el estado normalizado, media de las 3 semillas; `autoencoders[].reconstruction.mse` de `docs/results/v2/autoencoder/selection*.json`):

| k | MSE (semillas 0, 1, 2) | Media |
|---|---|---|
| 16 | 0.1377, 0.1361, 0.1256 | 0.1331 |
| 32 | 0.0382, 0.0379, 0.0407 | 0.0389 |
| 48 | 0.0203, 0.0193, 0.0169 | 0.0188 |
| 64 | 0.0074, 0.0049, 0.0050 | 0.0058 |
| 80 | 0.0024, 0.0031, 0.0027 | 0.0028 |
| 96 | 0.0026, 0.0023, 0.0025 | 0.0025 |

**Δ por tamaño y arquitectura** (por fila: 3 Autoencoders × 3 modelos temporales sobre z, frente a 3 modelos sobre el estado crudo). Varianzas de log GM: *entre* Autoencoders, *dentro* de un Autoencoder (semillas del modelo temporal) y entre las semillas crudas.

| Arquitectura | k | log GM z | log GM crudo | Diferencia de log GM | Δ relativo | IC 95% de Δ | ¿Excluye 0? | Var. entre AE | Var. dentro AE | Var. crudo | Fuente |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 16 | 8.2352 | 7.4247 | +0.8105 | +124.9% | [+100.5%, +151.8%] | sí | 0.00883 | 0.01848 | 0.00059 | `selection_analysis.json` |
| LSTM | 32 | 8.0610 | 7.4247 | +0.6363 | +88.9% | [+67.1%, +109.5%] | sí | 0.00638 | 0.02604 | 0.00059 | `selection_analysis.json` |
| LSTM | 48 | 7.7616 | 7.4247 | +0.3369 | +40.1% | [+16.1%, +64.3%] | sí | 0.01690 | 0.06054 | 0.00059 | `selection_analysis.json` |
| LSTM | 64 | 7.6703 | 7.4247 | +0.2456 | +27.8% | [+19.3%, +36.9%] | sí | 0.00275 | 0.00670 | 0.00059 | `selection_analysis.json` |
| LSTM | 80 | 7.3938 | 7.4247 | −0.0309 | −3.0% | [−15.6%, +9.5%] | no | 0.01505 | 0.01402 | 0.00059 | `selection_analysis.json` |
| LSTM | 96 | 7.4238 | 7.4247 | −0.0009 | −0.1% | [−9.7%, +11.0%] | no | 0.00389 | 0.02281 | 0.00059 | `selection_k96_analysis.json` |
| Transformer | 16 | 8.0038 | 7.2542 | +0.7496 | +111.6% | [+77.1%, +151.2%] | sí | 0.00790 | 0.02564 | 0.01903 | `selection_transformer_analysis.json` |
| Transformer | 32 | 7.7741 | 7.2542 | +0.5199 | +68.2% | [+41.2%, +99.6%] | sí | 0.00775 | 0.02494 | 0.01903 | `selection_transformer_analysis.json` |
| Transformer | 48 | 7.6400 | 7.2542 | +0.3859 | +47.1% | [+25.5%, +70.2%] | sí | 0.00055 | 0.02359 | 0.01903 | `selection_transformer_analysis.json` |
| Transformer | 64 | 7.4814 | 7.2542 | +0.2272 | +25.5% | [+3.9%, +52.3%] | sí | 0.01290 | 0.03077 | 0.01903 | `selection_transformer_analysis.json` |
| Transformer | 80 | 7.4105 | 7.2542 | +0.1563 | +16.9% | [−1.0%, +37.6%] | no | 0.00119 | 0.03414 | 0.01903 | `selection_transformer_analysis.json` |
| Transformer | 96 | 7.4309 | 7.2542 | +0.1767 | +19.3% | [+0.1%, +41.8%] | sí | 0.00040 | 0.04720 | 0.01903 | `selection_transformer_analysis.json` |
| TSMixer | 16 | 8.2261 | 7.9485 | +0.2776 | +32.0% | [+6.4%, +64.7%] | sí | 0.04101 | 0.00419 | 0.01739 | `selection_tsmixer_analysis.json` |
| TSMixer | 32 | 8.1881 | 7.9485 | +0.2396 | +27.1% | [+7.8%, +46.9%] | sí | 0.00380 | 0.02432 | 0.01739 | `selection_tsmixer_analysis.json` |
| TSMixer | 48 | 8.2453 | 7.9485 | +0.2968 | +34.6% | [+13.7%, +59.7%] | sí | 0.00622 | 0.03185 | 0.01739 | `selection_tsmixer_analysis.json` |
| TSMixer | 64 | 7.8813 | 7.9485 | −0.0672 | −6.5% | [−23.2%, +11.6%] | no | 0.00991 | 0.04267 | 0.01739 | `selection_tsmixer_analysis.json` |
| TSMixer | 80 | 7.9006 | 7.9485 | −0.0479 | −4.7% | [−18.5%, +8.9%] | no | 0.00276 | 0.01441 | 0.01739 | `selection_tsmixer_analysis.json` |
| TSMixer | 96 | 7.6843 | 7.9485 | −0.2641 | −23.2% | [−38.4%, −8.5%] | sí | 0.00635 | 0.06353 | 0.01739 | `selection_tsmixer_analysis.json` |

- Lectura (`docs/v2/ADDENDUM_AUTOENCODER.md`, secciones 8.1, 9.1 y 10.1): con mucha compresión (k ≤ 48) comprimir empeora a las tres arquitecturas. Con poca compresión (k = 80–96) es neutro con la LSTM (IC que incluyen 0), empeora al Transformer en k = 96 (en k = 80 el IC apenas incluye 0) y **ayuda a TSMixer en k = 96** (IC entero por debajo de 0). Aun así, TSMixer comprimido en k = 96 (7.6843) queda por encima de la LSTM sin comprimir (7.4247): su referencia sin comprimir es débil (sección 10.1, contexto fuera del pre-registro). La v2 sigue **sin Autoencoder** (`docs/v2/DISENO_CONTROL.md`, 2.1).

## 4. Arquitecturas con presupuesto de parámetros igualado (estado crudo de 104, 10 semillas cada una)

Métrica: log GM del `reward_mse` (h = 1…10) en el test del dataset. Fuente: `docs/results/v2/arch_comparison/analysis_three_architectures.json`.

| Arquitectura | Parámetros | Hiperparámetros | Media log GM | DE log GM | GM media del reward_mse | Mejores épocas (media; rango) | Épocas de parada (media) | Tocó el tope |
|---|---|---|---|---|---|---|---|---|
| lstm | 152,038 | {"hidden_dim": 137} | 7.5022 | 0.1815 | 1,812.0 | 110.4; 62–173 | 125.4 | 0 |
| transformer | 152,617 | {"d_model": 80, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.0} | 7.3863 | 0.1446 | 1,613.7 | 89.0; 64–127 | 104.0 | 0 |
| tsmixer | 152,129 | {"hidden_dim": 308, "num_blocks": 2, "dropout": 0.0} | 7.8696 | 0.1571 | 2,616.5 | 70.7; 46–104 | 85.7 | 0 |

**Comparaciones por pares** (Welch sobre log GM, IC al 98.33%, Bonferroni sobre 3 pares; diferencia = primera − segunda; negativo = la primera predice mejor):

| Par | Δ log GM | Relativa | p | IC de Δ | IC relativo | Decisión |
|---|---|---|---|---|---|---|
| transformer_minus_lstm | −0.1159 | −10.9% | 0.13 | [−0.3106, +0.0788] | [−26.7%, +8.2%] | sin evidencia suficiente |
| tsmixer_minus_lstm | +0.3674 | +44.4% | 1.4e−4 | [+0.1666, +0.5682] | [+18.1%, +76.5%] | lstm |
| tsmixer_minus_transformer | +0.4833 | +62.1% | 1.2e−6 | [+0.3049, +0.6616] | [+35.7%, +93.8%] | transformer |

Decisión global: **sin evidencia suficiente**; pasan a control: lstm, transformer. Razones de varianza: {"lstm_over_transformer": 1.576541572520825, "lstm_over_tsmixer": 1.334624204004002, "transformer_over_tsmixer": 0.8465518621687806}.

**`reward_mse` por horizonte, 10 semillas por arquitectura.** Media geométrica entre semillas = exp(`per_arch.<arq>.mean_log_reward_mse_per_horizon`), la escala de la métrica; entre paréntesis, la media aritmética de `runs.<arq>_s<i>.reward_mse` (calculada aquí). Fuente: `docs/results/v2/arch_comparison/analysis_three_architectures.json`.

| h | lstm: geométrica (aritmética) | transformer: geométrica (aritmética) | tsmixer: geométrica (aritmética) |
|---|---|---|---|
| 1 | 391.5 (393.2) | 259.7 (260.3) | 269.8 (271.0) |
| 2 | 925.7 (931.3) | 648.8 (650.9) | 763.7 (772.0) |
| 3 | 1,490.4 (1,506.2) | 1,210.8 (1,218.6) | 1,612.9 (1,636.8) |
| 4 | 1,886.5 (1,924.2) | 1,677.0 (1,693.6) | 2,537.5 (2,584.7) |
| 5 | 2,075.9 (2,128.6) | 2,007.8 (2,030.1) | 3,370.9 (3,450.2) |
| 6 | 2,294.3 (2,349.7) | 2,270.5 (2,302.7) | 4,142.5 (4,245.4) |
| 7 | 2,509.6 (2,566.8) | 2,514.8 (2,561.7) | 5,083.3 (5,172.8) |
| 8 | 2,783.4 (2,845.7) | 2,822.6 (2,903.5) | 5,934.2 (5,998.6) |
| 9 | 3,157.4 (3,233.8) | 3,146.0 (3,242.1) | 6,394.5 (6,458.1) |
| 10 | 3,565.8 (3,659.7) | 3,438.5 (3,516.5) | 6,619.6 (6,700.8) |

## 5. Controlador PPO en imaginación, sin planificar (Fase 3; prueba 22000–22023, 24 escenarios)

10 semillas por brazo aprendido; catastróficos con < −3,600 (umbral de la Fase 3). Fuente: `docs/results/v2/control/test_stage3_analysis.json`.

| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos |
|---|---|---|---|---|---|---|---|
| `sueno_lstm` | −16,168.1 | −3,225.0 | −3,394.4 | −2,247.3 | −5,432.6 | −5,093.7 | 118/240 |
| `sueno_transformer` | −2,340.4 | −1,567.0 | −581.3 | −502.9 | −943.8 | −312.4 | 18/240 |
| `fijo_2_3` | −1,591.7 | −1,617.0 | −347.9 | −209.5 | −864.2 | −170.2 | 0/24 |
| `min_verde_y_cambiar` | −1,309.7 | −1,305.5 | −451.5 | −230.4 | −365.3 | −262.5 | 0/24 |
| `cola_mas_larga` | −1,260.0 | −1,049.5 | −565.8 | −107.0 | −485.0 | −102.2 | 1/24 |
| `max_presion` | −7,565.5 | −6,555.5 | −1,585.2 | −2,916.0 | −1,086.8 | −1,977.5 | 22/24 |
| `espera_mas_larga` | −1,279.5 | −1,010.0 | −556.5 | −107.9 | −486.4 | −128.8 | 1/24 |
| `directo_10k` | −6,138.3 | −3,192.5 | −1,826.9 | −1,737.0 | −1,736.7 | −837.8 | 96/240 |
| `directo_30k` | −1,741.6 | −1,733.0 | −497.9 | −309.0 | −669.2 | −265.4 | 2/240 |

**Las 8 comparaciones** (α' = 0.05/8 = 0.00625, IC al 99.375%; test principal Welch sobre medias por semilla, o t de una muestra contra una regla; t pareada y Wilcoxon por escenario son secundarios y pseudorreplicación). Fuente: `docs/results/v2/control/test_stage3_analysis.json`.

| # | Comparación | Diferencia | p principal | IC | ¿Sig.? | t pareada p | Wilcoxon p | Gana |
|---|---|---|---|---|---|---|---|---|
| P1 | `sueno_lstm` − `sueno_transformer` (total) | −13,827.6 | 0.053 | [−35,831.4, 8,176.1] | no | 4.0e−10 | 1.2e−7 | 0/24 |
| P2a | `sueno_lstm` − `directo_10k` (total) | −10,029.8 | 0.15 | [−32,027.3, 11,967.8] | no | 1.1e−6 | 1.2e−6 | 1/24 |
| P2b | `sueno_transformer` − `directo_10k` (total) | 3,797.9 | 0.046 | [−1,867.0, 9,462.7] | no | 1.2e−6 | 1.2e−7 | 24/24 |
| P3a | `sueno_lstm` − `directo_30k` (total) | −14,426.5 | 0.045 | [−36,442.2, 7,589.2] | no | 2.6e−10 | 1.2e−7 | 0/24 |
| P3b | `sueno_transformer` − `directo_30k` (total) | −598.8 | 0.34 | [−2,680.2, 1,482.5] | no | 0.022 | 0.0072 | 5/24 |
| P4 | `sueno_lstm` − `espera_mas_larga` (total) | −14,888.5 | 0.04 | [−36,904.3, 7,127.3] | no | 1.0e−10 | 1.2e−7 | 0/24 |
| P5 | `sueno_lstm` − `espera_mas_larga` (B0) | −2,139.5 | 0.028 | [−5,037.4, 758.5] | no | 5.2e−6 | 1.2e−7 | 0/24 |
| P6 | `sueno_lstm` − `min_verde_y_cambiar` (C0) | −5,067.3 | 0.24 | [−19,513.9, 9,379.3] | no | 1.1e−4 | 1.2e−7 | 0/24 |

**Bloqueos** (controlador bloqueado: media < 3 cambios por episodio en algún semáforo; episodio bloqueado: < 3 cambios en algún semáforo). Controladores de `docs/results/v2/control/test_stage2_analysis.json` → `blocked_count` y `docs/results/v2/control/test_stage3_analysis.json` → `blocked_count`; episodios calculados aquí de `test_stage2.json` y `test_stage3.json`.

| Brazo | Controladores bloqueados | Episodios bloqueados | Por semáforo (A0/B0/C0/D0) |
|---|---|---|---|
| `sueno_lstm` | 1/10 | 75/240 | 4 / 17 / 8 / 56 |
| `sueno_transformer` | 0/10 | 2/240 | 0 / 2 / 1 / 0 |
| `directo_10k` | 0/10 | 4/240 | 0 / 3 / 0 / 1 |
| `directo_30k` | 0/10 | 0/240 | 0 / 0 / 0 / 0 |

**Fidelidad del retorno imaginado a 7 pasos** (validación; media de los 10 modelos; IC al 95% por bootstrap; sueño con la alineación `legacy`, la de todos los resultados de la Fase 3):

| Arquitectura | Acciones del dataset: Pearson [IC] | Sesgo imaginado − real [IC] | Acciones del controlador (on-policy): Pearson [IC] | Sesgo [IC] |
|---|---|---|---|---|
| lstm | 0.871 [0.827, 0.900] | 6.2 [−13.1, 26.4] | 0.433 [0.364, 0.563] | 970.1 [564.4, 1,334.5] |
| transformer | 0.897 [0.861, 0.926] | 25.0 [5.4, 44.6] | 0.667 [0.579, 0.709] | 32.2 [18.0, 44.9] |

Fuentes: `docs/results/v2/control/fidelity_validation.json` → `architectures.<arq>.clipped` (validación 21000–21023, 24 episodios × 39 ventanas por modelo); `docs/results/v2/control/fidelity_onpolicy_validation.json` → `summary.<arq>.clipped` (20 controladores en 21000–21005, 6 escenarios: IC orientativos).

**Sesgo on-policy según la fase mantenida más larga entre los 4 semáforos** (pasos desde el último cambio real al empezar a imaginar; recompensa recortada). `docs/results/v2/control/fidelity_onpolicy_validation.json` → `summary.<arq>.bias_by_longest_hold`:

| Retención | LSTM: ventanas | LSTM: retorno real medio | LSTM: sesgo | LSTM: imaginado > real | Transformer: ventanas | Transformer: retorno real | Transformer: sesgo | Transformer: imaginado > real |
|---|---|---|---|---|---|---|---|---|
| 0-5 | 1,020 | −329.1 | 85.7 | 77% | 1,555 | −231.8 | 31.3 | 71% |
| 6-10 | 318 | −658.9 | 345.9 | 84% | 400 | −235.3 | 28.2 | 65% |
| 11-19 | 402 | −1,259.4 | 842.0 | 90% | 283 | −241.2 | 32.3 | 65% |
| ≥ 20 | 600 | −3,465.0 | 2,890.2 | 92% | 102 | −232.8 | 62.3 | 83% |

## 6. Desalineación de la ventana de acciones del Dream Environment

`CorridorDreamEnvironment.step` (y el `DreamEnvironment` de la v1) avanzaba la ventana de acciones con la ventana anterior a insertar la acción elegida (`legacy`); `aligned` empareja cada estado con su acción, como `rollout_episode`. Ver `docs/v2/ADDENDUM_PLANIFICACION.md`, sección 11.

**Efecto en la fidelidad** (recortado; media de los modelos o controladores):

| | Pearson legacy | Pearson aligned | Sesgo legacy | Sesgo aligned |
|---|---|---|---|---|
| Acciones del dataset, lstm | 0.871 | 0.888 | 6.2 | 9.5 |
| Acciones del dataset, transformer | 0.897 | 0.899 | 25.0 | 19.3 |
| On-policy, lstm | 0.433 | 0.433 | 970.1 | 967.0 |
| On-policy, transformer | 0.667 | 0.677 | 32.2 | 30.6 |

Fuentes: `docs/results/v2/control/fidelity_validation.json`, `docs/results/v2/dream_alignment/fidelity_validation_aligned.json`, `docs/results/v2/control/fidelity_onpolicy_validation.json`, `docs/results/v2/dream_alignment/fidelity_onpolicy_validation_aligned.json`. Sesgo por retención con `aligned` (LSTM / Transformer): 0-5: 81.8 / 29.6; 6-10: 340.0 / 25.7; 11-19: 840.4 / 32.2; ≥ 20: 2,889.0 / 59.8.

**Efecto en el reentrenamiento de los 20 PPO del sueño** (misma configuración, `aligned`; evaluados con los de la Fase 3 en 24000–24023; catastróficos con < −3,600). Fuente: `docs/results/v2/dream_alignment/validation_v21_analysis.json`.

| Brazo | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados |
|---|---|---|---|---|---|---|---|---|---|
| phase3_lstm | −14,785.3 | −2,969.5 | −3,442.5 | −1,845.2 | −6,100.7 | −3,397.0 | 109/240 | 66/240 | 1/10 |
| aligned_lstm | −10,758.6 | −4,065.5 | −3,479.7 | −2,558.3 | −1,147.0 | −3,573.6 | 126/240 | 55/240 | 2/10 |
| phase3_transformer | −4,025.4 | −1,466.5 | −1,236.7 | −471.1 | −1,905.4 | −412.3 | 23/240 | 4/240 | 0/10 |
| aligned_transformer | −3,523.3 | −1,514.0 | −1,067.6 | −593.7 | −1,226.0 | −636.0 | 21/240 | 10/240 | 0/10 |

| Arquitectura | Corregido − Fase 3 | Welch p | IC 95% | ¿Cambio importante (criterio pre-registrado)? |
|---|---|---|---|---|
| lstm | 4,026.8 | 0.56 | [−10,700.7, 18,754.2] | no ({"blocked_episodes_halved": false, "catastrophic_halved": false, "return_ci_positive": false}) |
| transformer | 502.1 | 0.85 | [−5,218.7, 6,222.9] | no ({"blocked_episodes_halved": false, "catastrophic_halved": false, "return_ci_positive": false}) |

## 7. Validación del planificador (24000–24023, réplicas 0–2, 24 escenarios)

Catastróficos con < −3,700. * = H elegido. Fuente: `docs/results/v2/planning/validation_analysis.json`.

| Brazo | H | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | ms por decisión |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `plan_solo_lstm` | 3 * | −3,336.6 | −2,310.0 | −476.5 | −1,229.4 | −588.1 | −1,042.6 | 12/72 | 1/72 | 0/3 | 10 |
| `plan_solo_lstm` | 5 | −7,960.7 | −4,752.5 | −2,442.0 | −3,012.5 | −936.1 | −1,570.2 | 50/72 | 6/72 | 0/3 | 19 |
| `plan_solo_lstm` | 7 | −10,560.4 | −6,938.0 | −5,144.1 | −2,880.8 | −1,085.6 | −1,449.9 | 66/72 | 4/72 | 0/3 | 31 |
| `plan_ppo_lstm` | 3 * | −1,503.8 | −1,303.5 | −348.1 | −173.8 | −506.2 | −475.7 | 0/72 | 0/72 | 0/3 | 12 |
| `plan_ppo_lstm` | 5 | −1,829.8 | −1,557.5 | −429.4 | −218.4 | −529.1 | −652.9 | 1/72 | 0/72 | 0/3 | 24 |
| `plan_ppo_lstm` | 7 | −1,791.0 | −1,604.5 | −419.2 | −245.1 | −552.0 | −574.7 | 1/72 | 0/72 | 0/3 | 38 |
| `plan_solo_transformer` | 3 * | −1,360.7 | −1,199.0 | −330.6 | −318.0 | −556.4 | −155.7 | 1/72 | 0/72 | 0/3 | 11 |
| `plan_solo_transformer` | 5 | −2,901.3 | −2,349.0 | −540.5 | −1,021.3 | −833.5 | −506.0 | 17/72 | 1/72 | 0/3 | 19 |
| `plan_solo_transformer` | 7 | −4,649.1 | −3,214.0 | −1,133.7 | −1,126.1 | −1,367.5 | −1,021.7 | 28/72 | 0/72 | 0/3 | 27 |
| `plan_ppo_transformer` | 3 * | −1,123.9 | −1,069.5 | −318.4 | −155.4 | −509.0 | −141.1 | 0/72 | 0/72 | 0/3 | 13 |
| `plan_ppo_transformer` | 5 | −1,221.0 | −1,183.0 | −331.9 | −228.1 | −513.8 | −147.3 | 0/72 | 0/72 | 0/3 | 23 |
| `plan_ppo_transformer` | 7 | −1,256.6 | −1,176.5 | −342.9 | −254.4 | −489.2 | −170.1 | 0/72 | 0/72 | 0/3 | 34 |
| `sueno_lstm_10_semillas` | — | −14,785.3 | −2,969.5 | −3,442.5 | −1,845.2 | −6,100.7 | −3,397.0 | 109/240 | 66/240 | 1/10 | — |
| `sueno_lstm_semillas_0_1_2` | — | −14,335.6 | −8,054.5 | −5,313.6 | −1,497.8 | −1,263.4 | −6,260.7 | 45/72 | 29/72 | 1/3 | — |
| `sueno_transformer_10_semillas` | — | −4,025.4 | −1,466.5 | −1,236.7 | −471.1 | −1,905.4 | −412.3 | 20/240 | 4/240 | 0/10 | — |
| `sueno_transformer_semillas_0_1_2` | — | −9,373.8 | −1,620.5 | −3,184.0 | −986.6 | −4,728.7 | −474.5 | 12/72 | 4/72 | 0/3 | — |
| `fijo_2_3` | — | −1,622.4 | −1,612.0 | −392.8 | −163.2 | −893.5 | −172.9 | 0/24 | 0/24 | — | — |
| `min_verde_y_cambiar` | — | −1,247.2 | −1,206.5 | −409.8 | −230.4 | −322.6 | −284.5 | 0/24 | 0/24 | — | — |
| `cola_mas_larga` | — | −1,112.7 | −989.0 | −375.3 | −83.8 | −561.0 | −92.6 | 0/24 | 0/24 | — | — |
| `max_presion` | — | −6,329.5 | −6,013.0 | −1,304.5 | −2,028.0 | −1,189.2 | −1,807.9 | 23/24 | 2/24 | — | — |
| `espera_mas_larga` | — | −1,033.6 | −954.0 | −313.8 | −86.2 | −563.6 | −70.1 | 0/24 | 0/24 | — | — |

**Valores fijados por las reglas pre-registradas** (`selected`):

- H por brazo: `plan_solo_lstm` = 3, `plan_ppo_lstm` = 3, `plan_solo_transformer` = 3, `plan_ppo_transformer` = 3 (regla: menor costo; empate → H más chico).
- Mejor brazo por arquitectura: lstm: `plan_ppo_lstm` (−1,503.8), transformer: `plan_ppo_transformer` (−1,123.9).
- Umbral de catastróficos: −3,700 (1.5 × peor `fijo_2_3` = 1.5 × −2,402.0, redondeado hacia abajo a la centena).
- Mejores reglas: total: `espera_mas_larga` (−1,033.6), B0: `cola_mas_larga` (−83.8), C0: `min_verde_y_cambiar` (−322.6).

## 8. Prueba final con el planificador (25000–25047, 48 escenarios)

10 réplicas o semillas por método aprendido; 48 escenarios; catastróficos con < −3,700. Fuente: `docs/results/v2/planning/test_analysis.json`.

| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | ms por decisión |
|---|---|---|---|---|---|---|---|---|---|---|
| `directo_10k` | −5,736.8 | −2,990.5 | −1,756.5 | −1,269.6 | −1,783.2 | −927.5 | 179/480 | 7/480 | 0/10 | — |
| `directo_30k` | −1,752.3 | −1,729.0 | −508.9 | −290.1 | −688.4 | −265.0 | 1/480 | 0/480 | 0/10 | — |
| `plan_ppo_lstm` | −1,625.0 | −1,438.0 | −399.3 | −360.6 | −554.7 | −310.4 | 14/480 | 0/480 | 0/10 | 12.4 |
| `plan_ppo_transformer` | −1,165.4 | −1,157.5 | −327.0 | −168.1 | −517.7 | −152.6 | 0/480 | 0/480 | 0/10 | 13.5 |
| `plan_solo_lstm` | −6,084.4 | −3,232.5 | −933.7 | −2,905.1 | −672.7 | −1,572.8 | 206/480 | 57/480 | 1/10 | 10.2 |
| `plan_solo_transformer` | −1,457.0 | −1,333.5 | −351.2 | −327.4 | −559.1 | −219.3 | 5/480 | 3/480 | 0/10 | 11.3 |
| `fijo_2_3` | −1,630.5 | −1,559.5 | −377.1 | −219.7 | −857.0 | −176.8 | 0/48 | 0/48 | — | — |
| `min_verde_y_cambiar` | −1,313.8 | −1,317.0 | −460.1 | −252.3 | −316.5 | −285.0 | 0/48 | 0/48 | — | — |
| `cola_mas_larga` | −1,218.7 | −1,168.5 | −356.3 | −100.2 | −634.7 | −127.5 | 0/48 | 2/48 | — | — |
| `max_presion` | −7,053.0 | −6,175.5 | −1,616.6 | −2,145.7 | −1,186.0 | −2,104.6 | 46/48 | 11/48 | — | — |
| `espera_mas_larga` | −1,191.8 | −1,145.5 | −336.9 | −98.3 | −645.0 | −111.6 | 0/48 | 2/48 | — | — |
| `sueno_lstm` | −17,256.5 | −3,012.0 | −4,128.7 | −2,776.9 | −5,787.9 | −4,562.9 | 214/480 | 126/480 | 1/10 | — |
| `sueno_transformer` | −3,473.8 | −1,592.0 | −911.0 | −509.2 | −1,622.3 | −431.3 | 36/480 | 7/480 | 0/10 | — |

**Las 12 comparaciones planificadas** (α' = 0.05/12 = 0.004167, IC al 99.583%; principal: Welch sobre las 10 medias, o t de una muestra contra una regla; t pareada y Wilcoxon por escenario, secundarios y pseudorreplicación):

| # | Comparación | Diferencia | p principal | IC | ¿Sig.? | t pareada p | Wilcoxon p | Gana |
|---|---|---|---|---|---|---|---|---|
| Q1_lstm | `plan_ppo_lstm` − `sueno_lstm` (total) | 15,631.5 | 0.057 | [−11,635.3, 42,898.2] | no | 1.1e−15 | 7.1e−15 | 48/48 |
| Q2_lstm | `plan_ppo_lstm` − `directo_30k` (total) | 127.4 | 0.34 | [−329.8, 584.5] | no | 0.006 | 1.2e−6 | 42/48 |
| Q3_lstm | `plan_ppo_lstm` − `directo_10k` (total) | 4,111.8 | 0.023 | [−1,598.7, 9,822.3] | no | 1.3e−10 | 7.1e−15 | 48/48 |
| Q4_lstm | `plan_ppo_lstm` − `espera_mas_larga` (total) | −433.2 | 0.0057 | [−891.1, 24.7] | no | 2.6e−8 | 1.4e−7 | 7/48 |
| Q5_lstm | `plan_ppo_lstm` − `cola_mas_larga` (B0) | −260.4 | 0.012 | [−574.3, 53.5] | no | 1.1e−7 | 3.2e−12 | 3/48 |
| Q6_lstm | `plan_ppo_lstm` − `min_verde_y_cambiar` (C0) | −238.2 | 2.9e−8 | [−289.9, −186.5] | sí | 1.6e−17 | 2.2e−9 | 1/48 |
| Q1_transformer | `plan_ppo_transformer` − `sueno_transformer` (total) | 2,308.4 | 0.2 | [−4,102.1, 8,718.9] | no | 8.6e−5 | 7.1e−15 | 48/48 |
| Q2_transformer | `plan_ppo_transformer` − `directo_30k` (total) | 587.0 | 5.3e−8 | [423.3, 750.6] | sí | 2.1e−30 | 7.1e−15 | 48/48 |
| Q3_transformer | `plan_ppo_transformer` − `directo_10k` (total) | 4,571.4 | 0.014 | [−1,143.4, 10,286.2] | no | 8.4e−12 | 7.1e−15 | 48/48 |
| Q4_transformer | `plan_ppo_transformer` − `espera_mas_larga` (total) | 26.4 | 0.1 | [−29.0, 81.8] | no | 0.57 | 0.35 | 16/48 |
| Q5_transformer | `plan_ppo_transformer` − `cola_mas_larga` (B0) | −67.9 | 4.3e−4 | [−115.8, −20.1] | sí | 1.8e−7 | 2.4e−7 | 4/48 |
| Q6_transformer | `plan_ppo_transformer` − `min_verde_y_cambiar` (C0) | −201.2 | 8.9e−11 | [−224.0, −178.4] | sí | 5.3e−14 | 1.5e−12 | 1/48 |

**Medias por réplica o semilla, y la que más aporta a la varianza** (fracción de la suma de desvíos al cuadrado):

| Método | Medias (s0…s9) | Semilla dominante | Fracción |
|---|---|---|---|
| `directo_10k` | −4,255, −5,358, −2,818, −3,678, −4,628, −5,638, −19,011, −3,673, −4,827, −3,481 | direct_10k_s6 | 87% |
| `directo_30k` | −1,830, −1,550, −1,870, −1,575, −1,766, −1,922, −1,615, −1,772, −1,910, −1,713 | direct_30k_s1 | 25% |
| `plan_ppo_lstm` | −1,387, −2,060, −1,466, −1,547, −1,401, −1,553, −1,676, −1,331, −1,318, −2,512 | lstm_s9 | 60% |
| `plan_ppo_transformer` | −1,175, −1,201, −1,192, −1,072, −1,188, −1,151, −1,141, −1,136, −1,243, −1,156 | transformer_s3 | 45% |
| `plan_solo_lstm` | −2,301, −3,049, −6,629, −7,087, −4,565, −8,376, −3,840, −15,737, −4,646, −4,613 | lstm_s7 | 69% |
| `plan_solo_transformer` | −1,497, −1,381, −1,863, −1,379, −1,862, −1,109, −1,609, −1,312, −1,314, −1,243 | transformer_s2 | 29% |
| `sueno_lstm` | −27,314, −23,883, −4,952, −4,859, −1,915, −76,630, −15,979, −5,825, −6,032, −5,175 | dream_lstm_s5 | 76% |
| `sueno_transformer` | −1,522, −18,480, −1,411, −1,445, −1,340, −1,975, −3,818, −1,399, −1,676, −1,672 | dream_transformer_s1 | 88% |

Planificador con H = 3 en los 4 brazos; `plan_ppo` continúa con los PPO del sueño de la Fase 3 (`ADDENDUM_PLANIFICACION.md`, secciones 13.1 y 15).

## 9. OOD (23000–23029, 30 escenarios)

10 réplicas o semillas por método aprendido; 30 escenarios; catastróficos con < −3,700. Fuente: `docs/results/v2/ood_analysis.json`.

| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | ms por decisión |
|---|---|---|---|---|---|---|---|---|---|---|
| `directo_10k` | −6,309.2 | −2,917.0 | −2,120.0 | −1,432.1 | −1,846.5 | −910.7 | 106/300 | 9/300 | 0/10 | — |
| `directo_30k` | −1,703.1 | −1,663.0 | −498.5 | −277.6 | −667.4 | −259.5 | 1/300 | 0/300 | 0/10 | — |
| `plan_ppo_lstm` | −1,658.9 | −1,367.0 | −406.7 | −359.9 | −533.8 | −358.4 | 7/300 | 0/300 | 0/10 | 12.5 |
| `plan_ppo_transformer` | −1,149.9 | −1,101.5 | −339.9 | −154.7 | −498.6 | −156.7 | 0/300 | 0/300 | 0/10 | 13.6 |
| `fijo_2_3` | −1,685.9 | −1,510.5 | −360.3 | −200.4 | −937.4 | −187.7 | 0/30 | 0/30 | — | — |
| `min_verde_y_cambiar` | −1,347.8 | −1,308.0 | −488.5 | −242.5 | −309.8 | −307.0 | 0/30 | 0/30 | — | — |
| `cola_mas_larga` | −1,238.5 | −1,105.0 | −498.3 | −86.6 | −552.9 | −100.7 | 0/30 | 0/30 | — | — |
| `max_presion` | −7,836.2 | −6,823.5 | −1,565.7 | −3,258.5 | −1,145.6 | −1,866.4 | 28/30 | 9/30 | — | — |
| `espera_mas_larga` | −1,169.6 | −1,035.0 | −464.4 | −85.1 | −527.7 | −92.4 | 0/30 | 0/30 | — | — |
| `sueno_lstm` | −16,335.0 | −3,063.0 | −2,866.9 | −2,306.4 | −6,545.9 | −4,615.8 | 132/300 | 82/300 | 1/10 | — |
| `sueno_transformer` | −3,104.6 | −1,529.5 | −910.8 | −414.3 | −1,170.8 | −608.7 | 26/300 | 2/300 | 0/10 | — |

**Las 12 comparaciones planificadas** (α' = 0.05/12 = 0.004167, IC al 99.583%; principal: Welch sobre las 10 medias, o t de una muestra contra una regla; t pareada y Wilcoxon por escenario, secundarios y pseudorreplicación):

| # | Comparación | Diferencia | p principal | IC | ¿Sig.? | t pareada p | Wilcoxon p | Gana |
|---|---|---|---|---|---|---|---|---|
| O1_lstm | `plan_ppo_lstm` − `sueno_lstm` (total) | 14,676.2 | 0.062 | [−11,554.3, 40,906.7] | no | 8.5e−13 | 1.9e−9 | 30/30 |
| O2_lstm | `plan_ppo_lstm` − `directo_30k` (total) | 44.2 | 0.74 | [−423.1, 511.6] | no | 0.63 | 0.031 | 24/30 |
| O3_lstm | `plan_ppo_lstm` − `directo_10k` (total) | 4,650.4 | 0.032 | [−2,329.4, 11,630.1] | no | 4.0e−9 | 1.9e−9 | 30/30 |
| O4_lstm | `plan_ppo_lstm` − `espera_mas_larga` (total) | −489.3 | 0.0031 | [−955.9, −22.7] | sí | 1.1e−4 | 1.1e−5 | 2/30 |
| O5_lstm | `plan_ppo_lstm` − `cola_mas_larga` (B0) | −273.3 | 0.0021 | [−517.9, −28.6] | sí | 2.4e−4 | 3.1e−7 | 1/30 |
| O6_lstm | `plan_ppo_lstm` − `min_verde_y_cambiar` (C0) | −224.0 | 2.2e−7 | [−285.6, −162.4] | sí | 3.4e−13 | 3.7e−9 | 1/30 |
| O1_transformer | `plan_ppo_transformer` − `sueno_transformer` (total) | 1,954.7 | 0.12 | [−2,433.9, 6,343.3] | no | 7.6e−4 | 1.9e−9 | 30/30 |
| O2_transformer | `plan_ppo_transformer` − `directo_30k` (total) | 553.2 | 3.3e−7 | [365.7, 740.8] | sí | 1.2e−21 | 1.9e−9 | 30/30 |
| O3_transformer | `plan_ppo_transformer` − `directo_10k` (total) | 5,159.4 | 0.02 | [−1,824.1, 12,142.8] | no | 3.9e−10 | 1.9e−9 | 30/30 |
| O4_transformer | `plan_ppo_transformer` − `espera_mas_larga` (total) | 19.7 | 0.3 | [−49.1, 88.5] | no | 0.78 | 0.73 | 11/30 |
| O5_transformer | `plan_ppo_transformer` − `cola_mas_larga` (B0) | −68.0 | 4.2e−4 | [−115.8, −20.2] | sí | 1.0e−10 | 3.2e−6 | 2/30 |
| O6_transformer | `plan_ppo_transformer` − `min_verde_y_cambiar` (C0) | −188.8 | 8.8e−10 | [−216.5, −161.2] | sí | 3.0e−12 | 9.3e−9 | 1/30 |

**Medias por réplica o semilla, y la que más aporta a la varianza** (fracción de la suma de desvíos al cuadrado):

| Método | Medias (s0…s9) | Semilla dominante | Fracción |
|---|---|---|---|
| `directo_10k` | −3,809, −9,910, −2,454, −3,361, −3,206, −4,088, −21,748, −5,401, −4,750, −4,365 | direct_10k_s6 | 79% |
| `directo_30k` | −1,742, −1,526, −1,757, −1,621, −1,769, −1,775, −1,513, −1,661, −2,052, −1,617 | direct_30k_s8 | 55% |
| `plan_ppo_lstm` | −1,338, −2,069, −2,387, −1,426, −1,863, −1,532, −1,430, −1,248, −1,319, −1,977 | lstm_s2 | 39% |
| `plan_ppo_transformer` | −1,123, −1,186, −1,199, −1,087, −1,165, −1,139, −1,056, −1,124, −1,254, −1,166 | transformer_s8 | 37% |
| `sueno_lstm` | −25,458, −19,479, −4,952, −6,526, −1,839, −73,229, −20,509, −1,916, −2,490, −6,953 | dream_lstm_s5 | 76% |
| `sueno_transformer` | −1,496, −13,066, −1,394, −1,456, −1,290, −1,591, −3,683, −1,337, −4,078, −1,655 | dream_transformer_s1 | 83% |

**Control de reproducibilidad** (`reproducibility`): en los 20 escenarios OOD donde el dataset usó `fijo_2_3` o `cola_mas_larga`, el retorno evaluado frente al del manifiesto (`docs/results/v2/dataset/manifest.json`): diferencia máxima 0.0, discrepancias 0.

**Lectura pre-registrada** (`ADDENDUM_OOD.md`, sección 3) aplicada a O2_transformer: **confirma**.

OOD = semillas reservadas con la misma red, demanda y generador de escenarios (no una demanda distinta); sus 30 escenarios solo se habían simulado al recolectar el dataset (`ADDENDUM_OOD.md`, sección 4).

## 10. Interacciones reales y tiempos medidos

**Interacciones reales por réplica o semilla** (`interactions` de `docs/results/v2/planning/test_analysis.json`; pasos reales del RL directo y de la selección contados en `models/checkpoints/v2/control/*/run_info.json` → `real_steps_total`):

| Método | Sin compartir el dataset | Compartiéndolo entre 10 semillas | Composición |
|---|---|---|---|
| Planificador (H y brazo validados con las réplicas 0–2) | 21,240 | 6,552 | dataset 9,600 + selección del PPO 3,000 + validación de H y brazo 8,640 (sin compartir) o 2,592 (compartido) |
| PPO del sueño (Fase 3) | 12,600 | 3,960 | dataset 9,600 (o 960) + selección en SUMO real 3,000 |
| RL directo 10k | 13,240 | 13,240 | 10,240 de entrenamiento + 3,000 de evaluación periódica |
| RL directo 30k | 39,208 | 39,208 | 30,208 de entrenamiento + 9,000 de evaluación periódica |

Razones frente al planificador: directo 30k / planificador = 1.85x (sin compartir) y 5.98x (compartido); directo 10k / planificador = 0.62x y 2.02x.

**Tiempos de entrenamiento por corrida** (`run_info.json` → `wall_seconds`, con 4 procesos en paralelo, un hilo cada uno; `seconds_per_training_step` incluye las actualizaciones de PPO):

| Corridas | n | Minutos por corrida: media (mín–máx) | ms por paso de entrenamiento | Pasos reales |
|---|---|---|---|---|
| PPO del sueño LSTM (Fase 3) | 10 | 8.0 (7.1–8.5) | 4.4 | 3,000 |
| PPO del sueño Transformer (Fase 3) | 10 | 7.9 (7.1–8.8) | 4.5 | 3,000 |
| RL directo 10k | 10 | 16.7 (15.7–17.4) | 75.5 | 13,240 |
| RL directo 30k | 10 | 51.4 (47.2–57.5) | 78.8 | 39,208 |
| PPO del sueño corregido LSTM | 10 | 7.9 (7.6–8.3) | 4.4 | 3,000 |
| PPO del sueño corregido Transformer | 10 | 7.7 (7.5–8.0) | 4.4 | 3,000 |

**Planificador, ms por decisión (H = 3)** en la prueba final: `plan_ppo_lstm` 12.4, `plan_ppo_transformer` 13.5, `plan_solo_lstm` 10.2, `plan_solo_transformer` 11.3; en OOD: `plan_ppo_lstm` 12.5, `plan_ppo_transformer` 13.6 (`arms.<método>.ms_per_decision`).

**Duraciones totales de evaluación** (registradas en los addendums, no en un JSON): validación del planificador 29 min (`ADDENDUM_PLANIFICACION.md` 16); prueba final 1 h 44 min para 4,080 episodios (18); OOD 50 min 40 s para 1,950 episodios (`ADDENDUM_OOD.md` 7); 20 PPO del sueño 40 min (`ADDENDUM_CONTROL.md` 11.3); 20 PPO corregidos 39 min 52 s (`ADDENDUM_SUENO_CORREGIDO.md` 7). Todo con 4 procesos.

