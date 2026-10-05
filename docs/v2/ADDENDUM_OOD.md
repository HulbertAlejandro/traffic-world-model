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
