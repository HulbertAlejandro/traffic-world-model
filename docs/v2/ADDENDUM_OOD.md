# Pre-registro: confirmación final en OOD (v2)

**Escrito el 6 de octubre de 2026, antes de evaluar ningún controlador en 23000–23029.** No se
ejecuta todavía. Lo que se fija aquí no se cambia después de ver resultados, y no se agrega
ninguna comparación después.

## 1. Qué se evalúa en 23000–23029 (30 escenarios), una sola vez

| Política | Qué es | Controladores | Episodios |
|---|---|---|---|
| `plan_ppo_transformer` | planificador, H = 3, réplicas s0–s9, continuación con el PPO del sueño de la Fase 3 (`ADDENDUM_PLANIFICACION.md`, secciones 13.1 y 15) | 10 | 300 |
| `plan_ppo_lstm` | ídem con la LSTM | 10 | 300 |
| `sueno_transformer` | los 10 PPO del sueño de la Fase 3 (`models/checkpoints/v2/control/dream_transformer_s<i>`) | 10 | 300 |
| `sueno_lstm` | ídem con la LSTM | 10 | 300 |
| `directo_30k` | los 10 RL directos de 30k de la Fase 3 | 10 | 300 |
| `directo_10k` | los 10 RL directos de 10k de la Fase 3 | 10 | 300 |
| Las 5 reglas | `fijo_2_3`, `min_verde_y_cambiar`, `cola_mas_larga`, `max_presion`, `espera_mas_larga` | 5 | 150 |

- **Nada más.** `plan_solo_*` y los PPO del sueño corregido no se evalúan en OOD.
- Todo con acuerdo con las referencias en los mismos estados, como en el test nuevo.
- Total: **1,950 episodios**.
- **Valores que no se recalculan en OOD:**
  - el umbral de catastróficos: **−3,700** (sección 15 de `ADDENDUM_PLANIFICACION.md`);
  - las mejores reglas: `espera_mas_larga` en el total, `cola_mas_larga` en B0 y
    `min_verde_y_cambiar` en C0, fijadas en validation_v21.

## 2. Comparaciones planificadas

Son las 12 del test nuevo (`ADDENDUM_PLANIFICACION.md`, sección 17). Todas tienen sentido con lo
que se evalúa y no se agrega ninguna:

| # | Comparación | Métrica |
|---|---|---|
| O1_lstm | `plan_ppo_lstm` − `sueno_lstm` | total |
| O2_lstm | `plan_ppo_lstm` − `directo_30k` | total |
| O3_lstm | `plan_ppo_lstm` − `directo_10k` | total |
| O4_lstm | `plan_ppo_lstm` − `espera_mas_larga` | total |
| O5_lstm | `plan_ppo_lstm` − `cola_mas_larga` | B0 |
| O6_lstm | `plan_ppo_lstm` − `min_verde_y_cambiar` | C0 |
| O1_transformer | `plan_ppo_transformer` − `sueno_transformer` | total |
| O2_transformer | `plan_ppo_transformer` − `directo_30k` | total |
| O3_transformer | `plan_ppo_transformer` − `directo_10k` | total |
| O4_transformer | `plan_ppo_transformer` − `espera_mas_larga` | total |
| O5_transformer | `plan_ppo_transformer` − `cola_mas_larga` | B0 |
| O6_transformer | `plan_ppo_transformer` − `min_verde_y_cambiar` | C0 |

- **12 comparaciones → Bonferroni α' = 0.05/12 = 0.004167; IC al 99.583%.**
- **Test principal:** Welch sobre las 10 medias por réplica o semilla; contra una regla, t de una
  muestra.
- t pareada y Wilcoxon por escenario (30): **solo secundarios, marcados como pseudorreplicación**.
- Sin significancia en el test principal, se reporta "brecha no significativa".
- Las demás métricas por política, descriptivas, son las mismas del test nuevo:
  - media y mediana, y por intersección;
  - catastróficos (< −3,700);
  - bloqueados (controlador y episodio, reglas de la Fase 3);
  - cambios de fase por semáforo y ms por decisión;
  - acuerdo con las referencias;
  - medias por semilla y la semilla que más aporta a la varianza;
  - las dos contabilidades de interacciones.
- Se reportan todas las políticas, sin desempates.

## 3. Lectura previa (escrita antes de ver nada en OOD)

En el test nuevo (sección 18) se obtuvo:

| Comparación | Test nuevo |
|---|---|
| Q2_transformer: `plan_ppo_transformer` − `directo_30k` | **+587, significativo** |
| Q4_transformer: frente a `espera_mas_larga` | +26, empate no significativo |
| Q5_transformer: frente a `cola_mas_larga` en B0 | en contra, significativo |
| Q6 (las dos arquitecturas): frente a `min_verde_y_cambiar` en C0 | en contra, significativo |
| Q1: frente al sueño | positivo, no significativo |

| Resultado en OOD | Lectura |
|---|---|
| **O2_transformer positivo y significativo** | **Confirma** el hallazgo principal: el planificador con Transformer supera al RL directo de 30k con menos interacciones reales. |
| O2_transformer positivo pero no significativo | **Confirmación parcial.** El signo se mantiene; se reporta como brecha no significativa, sin declarar el hallazgo confirmado ni refutado. |
| O2_transformer negativo, o significativo en contra | **Contradice** el hallazgo del test nuevo, que se reportaría como no robusto a escenarios nuevos. |
| O4_transformer no significativo, cualquier signo | Coincide con el empate con la mejor regla. |
| O4_transformer significativo a favor | Iría más allá de lo visto: el planificador superaría a la mejor regla. |
| O4_transformer significativo en contra | Contradiría el empate. |
| O5 y O6 en contra y significativos | Confirman que el planificador pierde en B0 y C0 frente a la mejor regla de cada intersección. |
| O5 u O6 a favor | Contradiría el test nuevo. |
| O1 positivo | Coincide con el test nuevo; su significancia depende de si en OOD aparece de nuevo una semilla del sueño desbocada. |

Ninguna de estas lecturas cambia lo publicado en `ADDENDUM_CONTROL.md` ni en
`ADDENDUM_PLANIFICACION.md`: OOD se reporta al lado.

## 4. Qué es OOD en la v2 (para no sobrevender su alcance)

- **No es una demanda distinta.** Los escenarios 23000–23029 usan la **misma red, la misma demanda
  (candidata it5), el mismo generador de escenarios** (offset del pulso sorteado con la semilla), el
  mismo horizonte de 300 s y el mismo protocolo que todos los demás splits
  (`ADDENDUM_DATASET.md`, sección 2). "Fuera de distribución" significa **semillas reservadas desde
  el principio**, no un cambio de condiciones.
- En la práctica es **un segundo conjunto de test**, más chico (30 escenarios frente a 48), sobre
  la misma distribución. Mide si los resultados se sostienen en escenarios nuevos del mismo tipo,
  **no** si los métodos generalizan a otra demanda, otra red u otro horizonte. Eso queda fuera del
  alcance de la v2.
- **No es estrictamente virgen para todo.** Los 30 escenarios se simularon al recolectar el
  dataset, con las 3 políticas de recolección (10 episodios cada una: `aleatoria`, `fijo_2_3` y
  `cola_mas_larga`), y sus retornos están en `docs/results/v2/dataset/manifest.json`. El cierre de
  la Fase 1 inspeccionó un episodio de `aleatoria` (23006). Ningún controlador aprendido, ningún
  planificador y ninguna otra regla se evaluaron nunca ahí, y nada de OOD se usó para una decisión.
  Los episodios de `fijo_2_3` y `cola_mas_larga` en OOD reproducirán, en 10 escenarios cada uno,
  retornos que ya están en el manifiesto; eso sirve de control de reproducibilidad.
- 30 escenarios y 10 semillas: la potencia del test principal depende sobre todo de la varianza
  entre semillas, no del número de escenarios.

## 5. Cómo se ejecutará y cuánto tarda

- **Antes de lanzar:**
  - un lanzador `scripts/v2/run_ood.py`, igual que `run_planning_test.py` pero con
    `--split ood --confirm-held-out` y solo las políticas de la sección 1;
  - un análisis `scripts/v2/analyze_ood.py`, igual que `analyze_planning_test.py` con las 12
    comparaciones de la sección 2.
  - Los dos se escriben, se prueban con datos sintéticos y se commitean antes de simular nada en
    OOD.
- **Ejecución:**
  - 4 procesos, cada política aprendida partida en dos mitades de 5 semillas: 13 trabajos de
    ≈ 150 episodios.
  - Reanudable: salta los trabajos con su JSON. Se niega a arrancar sin corriente o con 2 GB o
    menos.
  - Salida: `docs/results/v2/ood/`.
- **Antes de simular,** una nota "evaluación OOD iniciada" con la hora, commiteada.
- **Estimación:** el test nuevo tardó 1 h 44 min para 4,080 episodios con 4 procesos (≈ 1.53 s
  por episodio). Para 1,950 episodios: **≈ 50 min, entre 45 y 60 min**. El planificador cuesta
  ≈ 13 ms por decisión con H = 3, ya incluido en esa tasa.
- **Integridad:** md5 antes y después de los 40 controladores de la Fase 3 y de sus resultados;
  pytest encadenado con `&&` antes de cada commit.

## 6. Evaluación en OOD iniciada: 2026-10-05 14:36 (hora local)

Hasta este momento no se ha evaluado ningún controlador, planificador ni regla en 23000–23029 en
esta fase. El lanzador (`scripts/v2/run_ood.py`) y el análisis (`scripts/v2/analyze_ood.py`)
están commiteados (`335e97a`), y los md5 de antes en `md5_ood_before.json` (1,037 archivos). Lo
siguiente es la evaluación única con `--confirm-held-out`.

## 7. Resultado de la confirmación en OOD (escrito después de la evaluación única)

**Ejecución.**

- 13 trabajos con 4 procesos, sin fallos ni interrupciones, en **50 min 40 s** (14:38:15–15:28:55
  del 6 de octubre). El equipo estuvo enchufado todo el tiempo.
- Una sola evaluación por controlador en 23000–23029: 1,950 episodios.
- Resultados por episodio en `docs/results/v2/ood/`; el análisis (`scripts/v2/analyze_ood.py`,
  escrito y probado antes de simular) en `docs/results/v2/ood_analysis.json`.

**Reproducibilidad.** En los 20 escenarios OOD donde el dataset usó `fijo_2_3` o `cola_mas_larga`,
el retorno evaluado coincide con el del manifiesto: **diferencia máxima 0.0, ninguna
discrepancia**.

**Todas las políticas** (30 escenarios; en las aprendidas, media de las 10 medias por réplica o
semilla; catastróficos con < −3,700):

| Política | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos | Episodios bloqueados | Controladores bloqueados | Cambios A0/B0/C0/D0 |
|---|---|---|---|---|---|---|---|---|---|---|
| `plan_ppo_lstm` | −1,658.9 | −1,367.0 | −406.7 | −359.9 | −533.8 | −358.4 | 7/300 | 0/300 | 0/10 | 23.5 / 19.4 / 26.7 / 20.4 |
| **`plan_ppo_transformer`** | **−1,149.9** | −1,101.5 | −339.9 | −154.7 | −498.6 | −156.7 | **0/300** | 0/300 | 0/10 | 23.5 / 18.6 / 25.6 / 19.2 |
| `sueno_lstm` | −16,335.0 | −3,063.0 | −2,866.9 | −2,306.4 | −6,545.9 | −4,615.8 | 132/300 | 82/300 | 1/10 (s1) | 19.7 / 16.7 / 21.6 / 14.8 |
| `sueno_transformer` | −3,104.6 | −1,529.5 | −910.8 | −414.3 | −1,170.8 | −608.7 | 26/300 | 2/300 | 0/10 | 21.0 / 17.6 / 23.7 / 18.2 |
| `directo_10k` | −6,309.2 | −2,917.0 | −2,120.0 | −1,432.1 | −1,846.5 | −910.7 | 106/300 | 9/300 | 0/10 | 20.4 / 15.9 / 20.7 / 17.0 |
| `directo_30k` | −1,703.1 | −1,663.0 | −498.5 | −277.6 | −667.4 | −259.5 | 1/300 | 0/300 | 0/10 | 21.2 / 16.7 / 23.5 / 18.6 |
| `fijo_2_3` | −1,685.9 | −1,510.5 | −360.3 | −200.4 | −937.4 | −187.7 | 0/30 | 0/30 | — | 23 / 23 / 23 / 23 |
| `min_verde_y_cambiar` | −1,347.8 | −1,308.0 | −488.5 | −242.5 | **−309.8** | −307.0 | 0/30 | 0/30 | — | 29 / 29 / 29 / 29 |
| `cola_mas_larga` | −1,238.5 | −1,105.0 | −498.3 | **−86.6** | −552.9 | −100.7 | 0/30 | 0/30 | — | 19.4 / 6.7 / 20.8 / 8.2 |
| `max_presion` | −7,836.2 | −6,823.5 | −1,565.7 | −3,258.5 | −1,145.6 | −1,866.4 | 28/30 | 9/30 | — | 13.6 / 3.4 / 16.8 / 4.6 |
| `espera_mas_larga` | −1,169.6 | **−1,035.0** | −464.4 | −85.1 | −527.7 | −92.4 | 0/30 | 0/30 | — | 19.6 / 6.8 / 21.4 / 8.3 |

**Medias por réplica o semilla (s0–s9):**

| Política | Medias |
|---|---|
| `plan_ppo_lstm` | −1,338, −2,069, −2,387, −1,426, −1,863, −1,532, −1,430, −1,248, −1,319, −1,977 |
| `plan_ppo_transformer` | −1,123, −1,186, −1,199, −1,087, −1,165, −1,139, −1,056, −1,124, −1,254, −1,166 |
| `sueno_lstm` | −25,458, −19,479, −4,952, −6,526, −1,839, **−73,229**, −20,509, −1,916, −2,490, −6,953 |
| `sueno_transformer` | −1,496, **−13,066**, −1,394, −1,456, −1,290, −1,591, −3,683, −1,337, −4,078, −1,655 |
| `directo_10k` | −3,809, −9,910, −2,454, −3,361, −3,206, −4,088, **−21,748**, −5,401, −4,750, −4,365 |
| `directo_30k` | −1,742, −1,526, −1,757, −1,621, −1,769, −1,775, −1,513, −1,661, −2,052, −1,617 |

**Qué semilla domina la varianza:**

| Política | Semilla | Fracción |
|---|---|---|
| `sueno_lstm` | s5 | 76% |
| `sueno_transformer` | s1 | 83% |
| `directo_10k` | s6 | 79% |
| `directo_30k` | s8 | 55% |
| `plan_ppo_lstm` | s2 | 39% |
| `plan_ppo_transformer` | s8 | 37%, con un rango de solo −1,056 a −1,254 |

Las semillas desbocadas son las mismas del test nuevo (sueño LSTM s5, sueño Transformer s1,
directo 10k s6).

**Bloqueo:** ningún planificador queda bloqueado. Su mínimo es de 10.0 cambios por episodio en
algún semáforo. Sigue bloqueado el PPO del sueño LSTM s1, con 1.2 cambios por episodio.

### Las 12 comparaciones planificadas

α' = 0.05/12 = 0.004167, IC al 99.583%. Test principal: Welch sobre las 10 medias (t de una
muestra contra una regla). La t pareada y el Wilcoxon por escenario son secundarios
(pseudorreplicación).

| # | Comparación | Diferencia | p principal | IC al 99.583% | ¿Significativo? | t pareada | Wilcoxon | Gana | En el test nuevo |
|---|---|---|---|---|---|---|---|---|---|
| O1_lstm | `plan_ppo_lstm` − `sueno_lstm` | +14,676.2 | 0.062 | [−11,554.3, +40,906.7] | no | 8.5e-13 | 1.9e-9 | 30/30 | +15,632, no |
| O2_lstm | `plan_ppo_lstm` − `directo_30k` | +44.2 | 0.74 | [−423.1, +511.6] | no | 0.63 | 0.031 | 24/30 | +127, no |
| O3_lstm | `plan_ppo_lstm` − `directo_10k` | +4,650.4 | 0.032 | [−2,329.4, +11,630.1] | no | 4.0e-9 | 1.9e-9 | 30/30 | +4,112, no |
| O4_lstm | `plan_ppo_lstm` − `espera_mas_larga` (total) | **−489.3** | **0.0031** | [−955.9, −22.7] | **sí, en contra** | 1.1e-4 | 1.1e-5 | 2/30 | −433, no (p = 0.0057) |
| O5_lstm | `plan_ppo_lstm` − `cola_mas_larga` (B0) | **−273.3** | **0.0021** | [−517.9, −28.6] | **sí, en contra** | 2.4e-4 | 3.2e-7 | 1/30 | −260, no (p = 0.012) |
| O6_lstm | `plan_ppo_lstm` − `min_verde_y_cambiar` (C0) | **−224.0** | **2.3e-7** | [−285.6, −162.4] | **sí, en contra** | 3.4e-13 | 3.7e-9 | 1/30 | −238, sí |
| O1_transformer | `plan_ppo_transformer` − `sueno_transformer` | +1,954.7 | 0.12 | [−2,433.9, +6,343.3] | no | 7.6e-4 | 1.9e-9 | 30/30 | +2,308, no |
| **O2_transformer** | **`plan_ppo_transformer` − `directo_30k`** | **+553.2** | **3.3e-7** | [**+365.7, +740.8**] | **sí, a favor** | 1.2e-21 | 1.9e-9 | 30/30 | +587, sí |
| O3_transformer | `plan_ppo_transformer` − `directo_10k` | +5,159.4 | 0.020 | [−1,824.1, +12,142.8] | no | 3.9e-10 | 1.9e-9 | 30/30 | +4,571, no |
| O4_transformer | `plan_ppo_transformer` − `espera_mas_larga` (total) | +19.7 | 0.30 | [−49.1, +88.5] | no | 0.78 | 0.73 | 11/30 | +26, no |
| O5_transformer | `plan_ppo_transformer` − `cola_mas_larga` (B0) | **−68.0** | **4.2e-4** | [−115.8, −20.2] | **sí, en contra** | 1.0e-10 | 3.2e-6 | 2/30 | −68, sí |
| O6_transformer | `plan_ppo_transformer` − `min_verde_y_cambiar` (C0) | **−188.8** | **8.8e-10** | [−216.5, −161.2] | **sí, en contra** | 3.0e-12 | 9.3e-9 | 1/30 | −201, sí |

### Lectura previa (sección 3), aplicada tal cual

| Resultado | Lectura |
|---|---|
| **O2_transformer: +553, significativo** | **CONFIRMA** el hallazgo principal del test nuevo: el planificador con Transformer supera al RL directo de 30k con 1.8x a 6.0x menos interacciones reales. El tamaño es casi igual (+553 frente a +587). |
| O4_transformer: +20, no significativo | Coincide con el empate con la mejor regla en el total. |
| O5 y O6: en contra y significativas en las dos arquitecturas | **Confirman** que el planificador pierde en B0 y C0 frente a la mejor regla de cada intersección. |
| O1: positivo, no significativo en las dos | Coincide con el test nuevo. La razón es la misma: las semillas desbocadas del sueño (LSTM s5, Transformer s1). |

**Lo que cambia respecto del test nuevo, dicho tal cual.** Con la LSTM, dos comparaciones que en
el test nuevo eran brechas no significativas **ahora son significativas en contra**:

- O4_lstm, en el total frente a `espera_mas_larga`: −489, p = 0.0031.
- O5_lstm, en B0 frente a `cola_mas_larga`: −273, p = 0.0021.

El signo es el mismo y el tamaño parecido; en el test nuevo los p eran 0.0057 y 0.012, cerca de
α'. **El planificador con LSTM queda por debajo de la mejor regla, y en OOD de forma
significativa.** No contradice ninguna lectura previa, que solo fijaba el total del Transformer.

**En resumen:** con dos conjuntos reservados (25000–25047 y 23000–23029), el planificador con
Transformer:

- supera al RL directo de 30k (significativo en los dos);
- empata con la mejor regla en el total (no significativo en los dos);
- pierde en B0 y C0 frente a la mejor regla de cada intersección (significativo en los dos).

Ningún método aprendido de la v2 supera a la mejor regla sin aprendizaje en el total.

**Integridad:** md5 de antes (`md5_ood_before.json`) y de después (`md5_ood_after.json`), sin
cambios en los controladores ni en los resultados anteriores.
