# Pre-registro: el sueño corregido (v2.1-0)

**Escrito el 5 de octubre de 2026, antes de entrenar ningún controlador con el sueño corregido y
antes de evaluar nada en 24000–24023.** Es un experimento **descriptivo**: no decide la arquitectura
oficial ni cambia ningún resultado publicado de la Fase 3 (`ADDENDUM_CONTROL.md`).

## 1. Pregunta

`CorridorDreamEnvironment` tenía un desfase en la ventana de acciones (`ADDENDUM_PLANIFICACION.md`,
sección 11). Los 20 PPO del sueño de la Fase 3 se entrenaron con él (`window_alignment="legacy"`).
¿Cambia su comportamiento en SUMO real si se entrenan en el sueño corregido
(`window_alignment="aligned"`), sobre todo el bloqueo y los catastróficos de la LSTM?

**Solo cambia un factor: la alineación de la ventana.**

## 2. Qué se entrena

- **20 PPO del sueño:** LSTM y Transformer, semillas 0–9, con
  `CorridorDreamEnvironment(window_alignment="aligned")`.
- **Todo lo demás, idéntico a la Fase 3:**
  - `ControllerConfig()` y la misma función `ppo_config`;
  - 50,000 pasos pedidos (50,176 hechos);
  - `dream_max_steps` = 7 y el recorte [−345.43, 0.0];
  - el modelo del mundo de la misma semilla y arquitectura (`arch_comparison/<arq>_raw_s<i>`);
  - las ventanas semilla del train;
  - la semilla de PPO *i*.
- **Selección del checkpoint en SUMO real con 21000–21004**, las mismas de la Fase 3, no
  24000–24004, para que la selección no sea un segundo factor.
- **Salidas:** `models/checkpoints/v2/control_aligned/dream_<arq>_s<i>/`. Los 40 controladores de
  `models/checkpoints/v2/control/` no se tocan: el script de entrenamiento se niega a escribir en
  esa carpeta sin `--overwrite-official`, y los md5 se comparan antes y después.
- `training/train_controller_v2.py --window-alignment aligned`; el valor queda en `run_info.json`.
- **4 procesos**, uno por corrida, reanudable. Solo con el equipo enchufado y más de 2 GB
  disponibles.

## 3. Qué se evalúa

- **Los 40 controladores del sueño** (los 20 antiguos de la Fase 3 y los 20 corregidos), **todos
  en `validation_v21` (24000–24023)**, una sola evaluación por controlador, con acuerdo con las
  referencias en los mismos estados (`--reference-agreement`).
- No se usan 25000–25047, 23000–23029 ni ningún otro split.

## 4. Métricas (fijadas ahora)

Por brazo (arquitectura × antiguo/corregido):

- **Retorno:** media y mediana del total, y media por intersección (A0, B0, C0, D0).
- **Catastróficos:** episodios con retorno < **−3,600** (el umbral de la Fase 3).
- **Bloqueo**, con las reglas de la Fase 3:
  - controlador "bloqueado": media de menos de 3 cambios de fase por episodio en algún semáforo;
  - episodio "bloqueado": menos de 3 cambios en algún semáforo.
- **Cambios de fase** medios por semáforo.
- **Acuerdo con cada referencia** por semáforo, en los mismos estados.
- **Comparación por arquitectura, corregido − antiguo:** Welch sobre las 10 medias por semilla, con
  IC al 95%. Una sola comparación por arquitectura y **sin corrección**: es descriptiva.

## 5. Cómo se leerá (escrito antes de ver el resultado)

Para cada arquitectura, el cambio se considera **"importante"** si, en validation_v21, se cumple
**al menos uno** de estos criterios:

1. **Episodios bloqueados:** los corregidos tienen **menos de la mitad** que los antiguos, siempre
   que los antiguos tengan al menos 10 (con menos, la mitad es ruido).
2. **Catastróficos:** los corregidos tienen **menos de la mitad** que los antiguos, con el mismo
   mínimo de 10.
3. **Retorno:** el IC al 95% de Welch de (corregido − antiguo) **excluye 0 y es positivo**.

Si no se cumple ninguno, el cambio **no** es importante. Eso incluye que los corregidos empeoren;
en ese caso se reporta el empeoramiento, pero no cuenta como cambio importante para la regla de
`plan_ppo` (`ADDENDUM_PLANIFICACION.md`, sección 13).

**Lectura prevista:**

- **Importante en la LSTM:** el desfase contribuía al bloqueo, y los PPO de la Fase 3 no
  representan lo que el sueño correcto produce.
- **No importante:** el bloqueo no viene del desfase. Sería consistente con el diagnóstico 12.1:
  el punto ciego persiste con el sueño alineado.
- **En ningún caso** se reescriben los resultados de la Fase 3, que quedan como "entrenados con el
  sueño con defecto".

## 6. Interacciones reales

Cada PPO corregido consume 3,000 pasos reales de selección, como en la Fase 3, más el dataset
compartido. La evaluación en validation_v21 (24 × 60 = 1,440 pasos por controlador) es
diagnóstica y no forma parte de la construcción de ningún controlador.

## 7. Resultado (diagnóstico; escrito después de la evaluación única en 24000–24023, el 6 de octubre de 2026)

**Entrenamiento.**

- Los 20 PPO corregidos terminaron sin fallos, con 4 procesos, en **39 min 52 s**
  (08:56:55–09:36:47).
- Cada uno: `window_alignment = "aligned"`, 50,176 pasos imaginados y 3,000 pasos reales de
  selección (21000–21004).
- Los 40 controladores y los resultados de la Fase 3 tienen el mismo md5 antes y después (774
  archivos, `md5_aligned_before` / `md5_aligned_after_training`).

**Evaluación.**

- Los 40 controladores, una sola vez cada uno, en `validation_v21` (24000–24023), con acuerdo con
  las referencias. Fueron 4 procesos en 22 min (09:37–09:59).
- Resultados en `docs/results/v2/dream_alignment/validation_v21_{phase3,aligned}_{lstm,transformer}.*`;
  análisis (escrito antes de evaluar) en `validation_v21_analysis.json`.

| Brazo | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados |
|---|---|---|---|---|---|---|---|---|---|
| LSTM, Fase 3 | −14,785.3 | −2,969.5 | −3,442.5 | −1,845.2 | −6,100.7 | −3,397.0 | 109/240 | 66/240 | 1/10 |
| **LSTM, corregido** | **−10,758.6** | **−4,065.5** | −3,479.7 | −2,558.3 | −1,147.0 | −3,573.6 | **126/240** | **55/240** | **2/10** |
| Transformer, Fase 3 | −4,025.4 | −1,466.5 | −1,236.7 | −471.1 | −1,905.4 | −412.3 | 23/240 | 4/240 | 0/10 |
| **Transformer, corregido** | **−3,523.3** | **−1,514.0** | −1,067.6 | −593.7 | −1,226.0 | −636.0 | **21/240** | **10/240** | **0/10** |

**Medias por semilla (s0–s9):**

- LSTM, Fase 3: −18,291, −22,376, −2,340, −6,432, −1,987, −66,892, −14,973, −1,773, −5,103 y
  −7,686.
- LSTM, corregido: −27,812, −14,269, −8,705, −11,573, −5,127, −12,272, −20,324, −1,731, −2,355 y
  −3,418.
- Transformer, Fase 3: −1,476, −25,320, −1,325, −1,422, −1,317, −1,544, −3,222, −1,419, −1,651 y
  −1,558.
- Transformer, corregido: −1,361, −11,503, −1,521, −1,440, −1,223, −3,289, −1,926, −9,471, −2,229
  y −1,271.

**Controladores bloqueados:** LSTM Fase 3, s1; LSTM corregido, s0 y s1. Ninguno del Transformer.

**Cambios de fase medios por episodio** (A0 / B0 / C0 / D0):

| | Fase 3 | Corregido |
|---|---|---|
| LSTM | 19.3 / 17.2 / 21.4 / 15.0 | 21.2 / 16.1 / 23.7 / 12.9 |
| Transformer | 20.6 / 18.2 / 23.5 / 18.0 | 19.8 / 18.5 / 23.3 / 16.5 |

**Corregido − Fase 3** (Welch sobre las 10 medias por semilla, IC al 95%, sin corrección):

| | Diferencia | p | IC al 95% |
|---|---|---|---|
| LSTM | +4,026.8 | 0.56 | [−10,700.7, +18,754.2] |
| Transformer | +502.1 | 0.85 | [−5,218.7, +6,222.9] |

**Criterios de "cambio importante" (sección 5):**

| | Episodios bloqueados, menos de la mitad | Catastróficos, menos de la mitad | IC del retorno positivo | ¿Importante? |
|---|---|---|---|---|
| LSTM | no (66 → 55) | no (109 → 126) | no | **no** |
| Transformer | no (4 → 10; la Fase 3 tiene menos de 10) | no (23 → 21) | no | **no** |

**Acuerdo con las referencias en los mismos estados** (A0 / B0 / C0 / D0; 0.5 es azar):

- Casi sin cambio.
- La LSTM corregida se parece algo más a `min_verde_y_cambiar` en A0 y C0 (0.63 y 0.74, frente a
  0.57 y 0.66) y a `max_presion` en D0 (0.70 frente a 0.64).
- El Transformer queda igual (por ejemplo, `max_presion` 0.66 / 0.66 / 0.57 / 0.73 frente a
  0.67 / 0.67 / 0.56 / 0.71).

**Lectura (la de la sección 5, escrita antes):**

- **En ninguna arquitectura el cambio es importante.** Con la LSTM, la media sube (+4,027), pero
  la mediana baja (−2,970 → −4,066), los catastróficos suben (109 → 126) y hay un controlador
  bloqueado más.
- El desfase de la ventana **no explica** el bloqueo ni los catastróficos de la LSTM. Es
  consistente con el diagnóstico 12.1 de `ADDENDUM_PLANIFICACION.md`: el punto ciego persiste con
  el sueño alineado.
- La varianza entre semillas sigue dominando:
  - Fase 3: una semilla de la LSTM en −66,892 y una del Transformer en −25,320.
  - Corregido: dos semillas del Transformer en −11,503 y −9,471.
- Los resultados de la Fase 3 no cambian; quedan como "entrenados con el sueño con el defecto".

## 8. Definición de `plan_ppo` (regla de `ADDENDUM_PLANIFICACION.md`, sección 13, punto 2)

Ninguna arquitectura tuvo un cambio importante, así que:

- **`plan_ppo_lstm` usa los PPO de la Fase 3:** `models/checkpoints/v2/control/dream_lstm_s<i>/best_model.zip`.
- **`plan_ppo_transformer` usa los PPO de la Fase 3:** `models/checkpoints/v2/control/dream_transformer_s<i>/best_model.zip`.

Así está escrito en el código: `plan:ARCH,ppo` carga `models/checkpoints/v2/control/`. Los PPO
corregidos quedan en `control_aligned/` como evidencia de este diagnóstico, sin otro uso.
