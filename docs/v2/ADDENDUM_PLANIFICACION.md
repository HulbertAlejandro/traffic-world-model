# Pre-registro: planificación con el modelo del mundo (v2.1-A)

**Escrito el 5 de octubre de 2026, antes de simular nada de esta fase** (ni las reglas ni los
planificadores, en ninguna semilla). Lo que se fija aquí no cambia después de ver resultados; un
cambio se agrega como enmienda fechada que dice si se escribió antes o después de ver qué datos.

## 0. Motivo

En la Fase 3 (`ADDENDUM_CONTROL.md`, secciones 11–12) ningún controlador aprendido superó a la
mejor regla sin aprendizaje. El PPO del sueño con LSTM se bloquea, y el diagnóstico on-policy
(12.1) mostró que el controlador explota los puntos ciegos del modelo: el sesgo de la recompensa
imaginada crece con las fases largas. Aquí se prueba la idea del trabajo de 7 intersecciones:
**planificar en cada decisión con el modelo del mundo, siempre desde el estado real**, para que el
modelo no tenga que imaginar más allá de unos pocos pasos ni a partir de estados inventados por la
política.

**Se reutilizan los 20 modelos del mundo** de `models/checkpoints/v2/arch_comparison/` (LSTM
oculta 137 y Transformer d_model 80, semillas 0–9) **sin reentrenar nada**, y, para un brazo, los
20 PPO del sueño de la Fase 3.

## 1. El planificador

En **cada paso de control real** t (60 por episodio):

1. **Ventana real.** Los últimos 16 estados reales `s_{t−15..t}`, normalizados con
   `datasets/v2/processed/scaler.pkl` (el mismo puente que el PPO, `ScaledCorridorEnvironment`), y
   las acciones realmente aplicadas `a_{t−15..t−1}`.
2. **Primera acción.** Se evalúan las **16 acciones conjuntas** posibles (4 bits mantener/cambiar,
   A0..D0) como acción de `s_t`, en lote. Cada una se codifica con `encode_actions`, como en el
   sueño.
3. **Continuación.** Después de la primera acción, el modelo avanza **H − 1 pasos más** con una
   política de continuación fija (sección 2), aplicada sobre el estado imaginado.
4. **Puntaje.** El retorno imaginado acumulado de los H pasos: la suma, sin descuento, de la
   recompensa predicha **recortada a [−345.43, 0.0]**, la misma que ve PPO en
   `CorridorDreamEnvironment`.
5. **Elección.** Se aplica en SUMO la acción con mayor puntaje. **Desempate:** entre las acciones
   empatadas en el máximo (igualdad exacta), la que pide **menos cambios**, de modo que "mantener
   la fase actual" en los 4 semáforos gana si está empatada. Si sigue el empate, la primera en el
   orden fijo `itertools.product((0, 1), repeat=4)`. Los empates no son raros: con el recorte en
   0.0, varias acciones pueden puntuar igual con la red casi vacía.

**[Corregido por la enmienda de la sección 11.]** El ventaneo de cada paso imaginado es el del
**Experimento 1** (`rollout_episode`): cada estado queda emparejado con la acción aplicada en él;
la ventana avanza un paso, el estado predicho entra al final y la acción de la continuación ocupa
la nueva última posición. **No** es el ventaneo del `CorridorDreamEnvironment` usado en la Fase 3,
que tiene un desfase (sección 11). Un test comprueba que, con las acciones registradas, los retornos
imaginados coinciden con los de `rollout_episode`.

El planificador **no tiene aprendizaje propio**: cada modelo del mundo da una réplica.

## 2. Brazos

| Brazo | Continuación tras la primera acción | Modelos |
|---|---|---|
| `plan_solo_lstm` | "mantener" en los 4 semáforos | LSTM s0–s9 |
| `plan_solo_transformer` | "mantener" en los 4 semáforos | Transformer s0–s9 |
| `plan_ppo_lstm` | el PPO del sueño de la misma semilla y arquitectura (`models/checkpoints/v2/control/dream_lstm_s<i>/best_model.zip`), determinista, sobre el estado imaginado. **Son los PPO de la Fase 3, entrenados en el sueño con el defecto de ventaneo** (sección 11) | LSTM s0–s9 |
| `plan_ppo_transformer` | ídem con `dream_transformer_s<i>` | Transformer s0–s9 |

Son 4 brazos × 10 réplicas = **40 planificadores**. La réplica *i* usa el modelo del mundo de
semilla *i* (y en `plan_ppo`, el PPO de semilla *i*, entrenado en ese mismo modelo).

## 3. Horizonte H

- Se elige **una sola vez, en validación (24000–24023)**, entre **{3, 5, 7}**, por brazo.
- **Regla:** el H de **menor COSTO**, es decir, de **mayor retorno medio en validación** (el
  menos negativo; media de las 10 réplicas de su media por episodio). Si hay empate exacto, el H
  más chico, que es más barato. Confirmado por el autor en la enmienda de la sección 11.
- H queda fijo para el test. No se ajusta nada más, y nada con el test.

## 4. Arranque del episodio (menos de 16 pasos de historia)

En el paso t < 15 hay t + 1 estados reales. **La ventana se completa por delante con copias del
estado inicial `s_0`, emparejadas con la acción "mantener" (0, 0, 0, 0)**, hasta tener 16
posiciones. Después vienen los estados reales con sus acciones aplicadas, y la candidata en la
última posición.

Ejemplos:

- t = 0: 15 copias de `s_0` con "mantener", y `s_0` con la candidata.
- t = 3: 12 copias de `s_0` con "mantener", `s_0..s_2` con sus acciones reales, y `s_3` con la
  candidata.

Es una regla de **inferencia** decidida por el autor para esta fase. No cambia la regla del
proyecto de descartar, no rellenar, los episodios cortos al **entrenar**: no se entrena nada. El
modelo nunca vio ventanas así, y lo que haga el planificador en los primeros 15 pasos (75 s) es
parte de lo que se mide.

## 5. Escenarios y referencias

| Rango | Uso en esta fase |
|---|---|
| **24000–24023** (validación v2.1, `ADDENDUM_CONTROL.md` 13) | Las 5 reglas y los 40 planificadores con H ∈ {3, 5, 7}. Elige H, el mejor brazo por arquitectura, el umbral de catastróficos y las mejores reglas. |
| **25000–25047** (test v2.1) | **Una sola vez, en la etapa 2.** |
| 20000–23029, 30000–30999 | No se usan. |

**En la etapa 2 se evaluarán en 25000–25047** los 4 brazos del planificador (con su H), las 5
reglas, los 20 PPO del sueño y los 20 RL directos de la Fase 3. No se reutilizan los episodios del
test viejo.

**Umbral de catastróficos:** la misma regla de la Fase 3, 1.5 × el peor retorno de `fijo_2_3` en
las 24 semillas de validación, redondeado hacia abajo a la centena.

**Mejor regla:** la de mayor retorno medio en validación, por separado para el total, B0 y C0.

## 6. Comparaciones planificadas (sobre el test 25000–25047)

**Mejor brazo por arquitectura**, fijado en validación: entre `plan_solo` y `plan_ppo` (cada uno con
su H), el de mayor retorno medio en validación. Si hay empate exacto, `plan_solo` (no depende de un
controlador aprendido). No depende del test.

Para cada arquitectura a ∈ {LSTM, Transformer}, con P*_a el mejor brazo de a:

| # | Comparación | Métrica |
|---|---|---|
| Q1_a | P*_a − `sueno_a` (el PPO del sueño de la misma arquitectura) | total |
| Q2_a | P*_a − `directo_30k` | total |
| Q3_a | P*_a − `directo_10k` | total |
| Q4_a | P*_a − mejor regla del total | total |
| Q5_a | P*_a − mejor regla de B0 | retorno de B0 |
| Q6_a | P*_a − mejor regla de C0 | retorno de C0 |

- **12 comparaciones → Bonferroni α' = 0.05 / 12 = 0.004167; IC al 99.583%.**
- **Test principal:** Welch sobre las medias por réplica o semilla (10 frente a 10); contra una
  regla determinista, t de una muestra sobre las 10 diferencias.
- t pareada y Wilcoxon por escenario (48): **solo secundarios**, marcados como
  **pseudorreplicación respecto del método**.
- Q1 empareja por semilla (el planificador *i* y el PPO *i* comparten modelo del mundo), pero se
  usa Welch como en el resto, para no cambiar de test entre comparaciones.
- Una diferencia cuyo test principal no llega a α' se reporta como **brecha no significativa**.
- **Se reportan TODOS los brazos.** El otro brazo de cada arquitectura se evalúa en test igual, de
  forma descriptiva y sin corrección. No hay regla de desempate que esconda un brazo malo.

## 7. Hipótesis

- **H1p.** El planificador **reduce los episodios catastróficos** frente al PPO del sueño sin
  planificación de la misma arquitectura. Se reporta el conteo, de forma descriptiva.
- **H2p.** El planificador **mejora el retorno** frente a ese PPO del sueño (Q1).
- Si el planificador no supera a la mejor regla (Q4–Q6), se reporta así, sin matices que lo
  escondan.

## 8. Interacciones reales (contabilidad conservadora, fijada ahora)

Todo lo que se usa para construir el planificador elegido de cada arquitectura, por réplica:

| Concepto | Pasos reales |
|---|---|
| Dataset del modelo del mundo | 9,600, o 960 compartido entre las 10 réplicas |
| Selección del PPO de la Fase 3 (los dos brazos se consideraron) | 3,000 |
| Validación de H y del brazo: 2 brazos × 3 H × 24 episodios × 60 pasos | **8,640** |
| **Total** | **21,240 sin compartir el dataset; 12,600 compartiéndolo** |

Las 5 reglas en validación no se cuentan: son referencias.

## 9. Cómputo

- **4 procesos.** Antes de lanzar se comprueba que haya más de 2 GB de memoria disponible y que
  el equipo esté enchufado y sin suspensión. El autor autoriza cerrar Steam, Edge y Copilot, nada
  más.
- Cada proceso usa un hilo de PyTorch.
- Cada brazo y H es un trabajo de 10 réplicas × 24 episodios, con su propio JSON; un análisis los
  combina sin simular.

## 10. Etapas

- **Etapa 1 (esta):** pre-registro, código, tests y validación (Paso 4).
- **Etapa 2:** la nota "evaluación en test iniciada" con la hora, la evaluación única en
  25000–25047 y las comparaciones de la sección 6.
- **OOD (23000–23029)** sigue reservado para la comparación final.

## 11. Enmienda del 5 de octubre de 2026: escrita ANTES de simular ningún planificador, DESPUÉS de descubrir un defecto en el Dream Environment

Ningún planificador, ni ninguna regla de esta fase, se ha simulado todavía en ninguna semilla.

**El defecto.** `CorridorDreamEnvironment.step` (y el `DreamEnvironment` de la v1, con el mismo
código) avanza la ventana de acciones con la ventana **anterior** a insertar la acción elegida
(`self._window_actions[1:]`), en vez de con la ventana ya actualizada.

- La acción aplicada en un estado nunca queda emparejada con ese estado en la historia; queda la
  del paso anterior.
- Con las acciones registradas, la diferencia aparece desde el **tercer** paso imaginado; con
  acciones distintas de las registradas, desde el segundo.
- Se detectó porque el test (b) del planificador no coincidía con `control_fidelity.py`.
- Comprobado con la LSTM s0, en el episodio de validación 21000, con las acciones registradas:

  | Inicio | `rollout_episode` (referencia) | Planificador | Sueño con el defecto | Real |
  |---|---|---|---|---|
  | 0 | −546.22 | −546.22 | −628.44 | −600 |
  | 10 | −1,444.01 | −1,444.01 | −1,318.26 | −1,661 |
  | 20 | −537.58 | −537.58 | −552.31 | −560 |

**Lo que cambia en este pre-registro:**

1. **El planificador usa la alineación del Experimento 1:** la acción de cada estado queda pegada
   a ese estado. No usa la del Dream Environment. Se corrigió el texto de la sección 1.
2. **El test (b) compara contra `rollout_episode`**, capturando sus predicciones con
   `scripts/v2/control_fidelity.py::rollout_episode_returns`, no contra `control_fidelity.py`.
3. **`plan_ppo` usa los PPO de la Fase 3, que se entrenaron en el sueño con el defecto.** Se
   mantienen tal cual, porque son los controladores que existen, y queda escrito.
4. **Regla de H:** el H de **menor costo** (mayor retorno, el menos negativo), como ya decía la
   sección 3. Se corrige la redacción.

Nada más cambia: brazos, H ∈ {3, 5, 7}, relleno inicial, desempate, semillas, comparaciones,
Bonferroni y contabilidad.

## 12. Diagnóstico: fidelidad con el sueño alineado (5 de octubre de 2026; descriptivo, no entrena ni evalúa planificadores)

Se repiten los dos diagnósticos de fidelidad de la Fase 3 con
`CorridorDreamEnvironment(window_alignment="aligned")`, la alineación del Experimento 1 (sección
11), al lado de los publicados, que usan `"legacy"`:

- Resultados: `docs/results/v2/dream_alignment/fidelity_validation_aligned.json` y
  `fidelity_onpolicy_validation_aligned.json`.
- Scripts: `scripts/v2/control_fidelity.py` y `control_fidelity_onpolicy.py` con
  `--window-alignment aligned`. Se niegan a reescribir los JSON publicados.

No se usan 25000–25047 ni 23000–23029.

**(i) Con las acciones del dataset, validación 21000–21023** (como `ADDENDUM_CONTROL.md` 10.2;
media de los 10 modelos, recompensa recortada, IC al 95% por bootstrap de episodios):

| | Pearson, legacy (publicado) | Pearson, aligned | Sesgo, legacy | Sesgo, aligned |
|---|---|---|---|---|
| LSTM | 0.871 [0.827, 0.900] | **0.888** [0.843, 0.919] | +6.2 [−13.1, +26.4] | **+9.5** [−4.2, +24.1] |
| Transformer | 0.897 [0.861, 0.926] | **0.899** [0.864, 0.929] | +25.0 [+5.4, +44.6] | **+19.3** [+4.3, +36.4] |

Sin recortar, con "aligned": LSTM 0.893 y +9.4; Transformer 0.901 y +17.4.

**(ii) On-policy, con las trayectorias reales de los 20 controladores de la Fase 3 en 21000–21005**
(como 12.1):

- Las trayectorias son las mismas: los mismos retornos reales en las 4,680 ventanas.
- La reproducción de la selección da una diferencia máxima de 0.0008.
- Solo cambia lo imaginado.

| | Pearson, legacy (publicado) | Pearson, aligned | Sesgo, legacy | Sesgo, aligned |
|---|---|---|---|---|
| LSTM | 0.433 [0.364, 0.563] | **0.433** [0.364, 0.559] | +970.1 [+564.4, +1,334.5] | **+967.0** [+560.0, +1,333.7] |
| Transformer | 0.667 [0.579, 0.709] | **0.677** [0.590, 0.722] | +32.2 [+18.0, +44.9] | **+30.6** [+15.5, +44.4] |

**Sesgo según la fase mantenida más larga** (recortado; entre paréntesis, la fracción de ventanas
con imaginado > real, "aligned"):

| Retención | LSTM: ventanas | LSTM legacy | LSTM aligned | Transformer: ventanas | Transformer legacy | Transformer aligned |
|---|---|---|---|---|---|---|
| 0–5 | 1,020 | +85.7 | +81.8 (76%) | 1,555 | +31.3 | +29.6 (70%) |
| 6–10 | 318 | +345.9 | +340.0 (81%) | 400 | +28.2 | +25.7 (63%) |
| 11–19 | 402 | +842.0 | +840.4 (90%) | 283 | +32.3 | +32.2 (65%) |
| ≥ 20 | 600 | +2,890.2 | +2,889.0 (92%) | 102 | +62.3 | +59.8 (83%) |

Con 20 o más pasos, por semáforo ("aligned"):

| | LSTM | Transformer |
|---|---|---|
| A0 | 42 ventanas, +8,882 | 0 ventanas |
| B0 | 146 ventanas, +1,830 | 54 ventanas, +76 |
| C0 | 13 ventanas, +27,002 | 3 ventanas, +462 |
| D0 | 468 ventanas, +2,504 | 55 ventanas, +36 |

**Lectura (descriptiva):**

- **Sobre las trayectorias ya registradas, corregir la alineación cambia poco la fidelidad
  medida.** Con las acciones del dataset, la LSTM sube de 0.871 a 0.888. On-policy, los números
  casi no se mueven.
- **El patrón del "punto ciego" persiste con el sueño alineado:** en la LSTM, el sesgo crece de
  +82 a +2,889 con la retención. Así que, **en estas trayectorias, no es un artefacto del
  desfase**. Sigue confundido con la congestión, como en 12.1.
- **Lo que esto NO mide:** si un PPO entrenado en el sueño alineado aprendería otra política. Los
  20 PPO de la Fase 3 se entrenaron con el desfase, y su comportamiento (por ejemplo, el bloqueo
  de la LSTM) podría cambiar al reentrenar. Eso no se ha medido.
- Por qué el efecto es chico aquí: el desfase solo cambia la acción emparejada con estados
  anteriores dentro de la ventana, desde el tercer paso imaginado. Cuando las acciones
  consecutivas coinciden (por ejemplo, al mantener una fase), no hay diferencia.

**Estimaciones (Paso 6, no ejecutadas):**

**(a) Reentrenar los 20 PPO del sueño con `"aligned"`** (misma configuración, selección en
24000–24004):

- La etapa 2 tardó **40 min** con 4 procesos (cada corrida, ~8 min: 50,176 pasos imaginados a
  ~3 ms más 50 episodios reales de selección).
- La alineación no cambia el costo del paso imaginado, así que serían **≈ 40–45 min** con el
  equipo enchufado y a plena potencia.
- No incluye evaluar en ningún test.

**(b) Validación del planificador** (Paso 4 del prompt anterior: 5 reglas y 40 planificadores ×
H ∈ {3, 5, 7} × 24 episodios = 2,880 episodios de planificador más 120 de reglas).

Costo por decisión, medido con un microbenchmark sobre 60 estados de validación del dataset, 1
hilo, sin SUMO. El equipo venía de funcionar con batería y estaba ralentizado; el valor de la LSTM
con `ppo` y H = 3, más lento que con H = 5, indica ruido:

| | H = 3 | H = 5 | H = 7 |
|---|---|---|---|
| LSTM, solo | 190 ms | 309 ms | 399 ms |
| LSTM, ppo | 349 ms | 300 ms | 457 ms |
| Transformer, solo | 42 ms | 76 ms | 109 ms |
| Transformer, ppo | 48 ms | 79 ms | 118 ms |

Por episodio (60 decisiones + ~2.5 s de SUMO):

| | Por episodio | Por brazo (240 episodios por cada H) |
|---|---|---|
| LSTM | ≈ 14–30 s | solo ≈ 4.0 h, ppo ≈ 5.0 h |
| Transformer | ≈ 5–10 s | solo ≈ 1.4 h, ppo ≈ 1.5 h |
| Reglas | ≈ 2 s | ≈ 4 min |

**Total ≈ 11.9 h en serie, ≈ 4 h con 4 procesos** (en la etapa 3, 4 procesos rindieron unas
3 veces más que uno). Si el costo de la LSTM con el equipo a plena potencia resulta la mitad,
serían ≈ 2.5 h. La LSTM domina el costo.

## 13. Enmienda del 5 de octubre de 2026: escrita ANTES de simular ningún planificador y DESPUÉS de ver los diagnósticos de fidelidad (sección 12)

Ningún planificador se ha simulado todavía. Esta enmienda también se escribió **antes de entrenar
y evaluar los PPO del sueño corregido** (`ADDENDUM_SUENO_CORREGIDO.md`).

**1. H se elige con 3 réplicas, no con 10.**

- En validation_v21 (24000–24023) se evalúan, para cada brazo, solo los modelos del mundo de
  semillas **0, 1 y 2**, con H ∈ {3, 5, 7}.
- Misma regla de la sección 3: el H de **menor costo** (mayor retorno medio, media de las 3
  réplicas de su media por episodio); ante empate exacto, el H más chico.
- H queda fijo. **En el test nuevo (25000–25047) se usan las 10 semillas.**
- El mejor brazo por arquitectura (sección 6) se elige con esas mismas 3 réplicas y su H, con la
  misma regla.
- Motivo: el costo. La validación con 10 réplicas se estimó en ≈ 4 h con 4 procesos (sección 12).
  Con 3 son ≈ 1.2 h.

**2. Definición de `plan_ppo` según la Parte A, por arquitectura.** El criterio de "cambio
importante" es el de la sección 5 de `ADDENDUM_SUENO_CORREGIDO.md`:

- Si los PPO corregidos de esa arquitectura **cambian de forma importante**, `plan_ppo_<arq>` usa
  como continuación los **PPO corregidos**
  (`models/checkpoints/v2/control_aligned/dream_<arq>_s<i>/best_model.zip`).
- **Si no**, usa los **PPO de la Fase 3** (`models/checkpoints/v2/control/dream_<arq>_s<i>/`),
  como fija la sección 11.
- La decisión se toma con el resultado de validation_v21 de la Parte A, se escribe en
  `ADDENDUM_SUENO_CORREGIDO.md`, y no se cambia después.

**3. Contabilidad de interacciones (sección 8), ajustada a 3 réplicas.** La validación de H y del
brazo cuesta 2 brazos × 3 H × 24 episodios × 60 pasos × 3 réplicas = **25,920 pasos reales por
arquitectura**, que sirven a las 10 réplicas del test:

| | Por réplica |
|---|---|
| Sin compartir (lo que costaría la validación a una réplica sola: 2 × 3 × 24 × 60) | 8,640 pasos de validación + 9,600 del dataset + 3,000 de la selección del PPO = **21,240** |
| Compartiéndolos entre las 10 réplicas | 2,592 + 960 + 3,000 = **6,552** |

Si `plan_ppo` usa los PPO corregidos, su selección también cuesta 3,000 pasos por semilla, así que
las cifras no cambian.

Nada más cambia: brazos, H ∈ {3, 5, 7}, alineación del Experimento 1, relleno inicial, desempate,
semillas, comparaciones y Bonferroni (12, α' = 0.05/12).

### 13.1 Resultado de la regla de `plan_ppo` (6 de octubre de 2026)

Con la evaluación de `ADDENDUM_SUENO_CORREGIDO.md`, sección 7, ninguna arquitectura tuvo un cambio
importante. **`plan_ppo` usa los PPO de la Fase 3 en las dos arquitecturas**
(`models/checkpoints/v2/control/dream_<arq>_s<i>/`). Ningún planificador se ha simulado todavía.

## 14. Plan de ejecución del Paso 4, validación (escrito el 6 de octubre de 2026, antes de simular)

**Qué se evalúa en validation_v21 (24000–24023), una vez:**

- Las 5 reglas: 120 episodios.
- Los planificadores de las réplicas 0, 1 y 2 de cada brazo (`plan_solo` y `plan_ppo`, con LSTM y
  con Transformer), con H ∈ {3, 5, 7}: 4 brazos × 3 H × 3 réplicas × 24 = **864 episodios**.
  `plan_ppo` continúa con los PPO de la Fase 3 (13.1).
- Todo con acuerdo con las referencias en los mismos estados.

No se evalúan las 10 réplicas ni nada en 25000–25047 o 23000–23029.

**Cómo:**

- `python scripts/v2/run_planning_validation.py --workers 4`: 13 trabajos, cada uno una llamada a
  `evaluate_control_v2.py` en su propio proceso con un hilo de PyTorch.
- Los más lentos (LSTM, H largo) van primero.
- Reanudable: salta los trabajos cuyo JSON ya existe.
- Se niega a arrancar sin corriente o con 2 GB o menos.
- Salida: `docs/results/v2/planning/validation/`.

**Después,** `python scripts/v2/analyze_planning_validation.py`, escrito antes de simular, aplica
sin cambios:

- **H por brazo:** menor costo; ante empate exacto, el H más chico.
- **Mejor brazo por arquitectura:** con las réplicas 0–2 y su H; ante empate, `plan_solo`.
- **Umbral de catastróficos:** 1.5 × el peor `fijo_2_3` en 24000–24023, redondeado hacia abajo a
  la centena.
- **Mejor regla:** en total, B0 y C0.

Esos valores se escriben aquí (sección 15) antes que cualquier otra cosa.

**Estimación** (microbenchmark de la sección 12; SUMO ≈ 2.5 s por episodio; el acuerdo con las
referencias agrega algo):

| Trabajos | Por episodio | Total en serie |
|---|---|---|
| 6 de LSTM (72 episodios cada uno) | ≈ 15–30 s | ≈ 2.5 h |
| 6 de Transformer | ≈ 5–10 s | ≈ 50 min |
| Reglas | | ≈ 5 min |

**≈ 3.3 h en serie; ≈ 1–1.5 h con 4 procesos.**

**Integridad:**

- md5 antes y después de los controladores y resultados de la Fase 3
  (`md5_official.py --planning`, que incluye `models/checkpoints/v2/control/` y
  `docs/results/v2/control/`).
- Equipo enchufado; 3.0 GB disponibles tras cerrar Edge, con la autorización del autor.

## 15. Valores fijados por las reglas en validación (6 de octubre de 2026; escrito antes que cualquier otra cosa)

Salen de `docs/results/v2/planning/validation_analysis.json`
(`scripts/v2/analyze_planning_validation.py`, escrito antes de simular), aplicando sin cambios las
reglas de las secciones 3, 5, 6 y 13 a la validación de la sección 14 (24000–24023; réplicas 0–2):

| Qué | Valor | Detalle |
|---|---|---|
| Umbral de catastróficos | **−3,700** | Peor `fijo_2_3`: −2,402.0; 1.5 × −2,402.0 = −3,603 → −3,700 |
| Mejor regla, total (Q4) | **`espera_mas_larga`** | −1,033.6 |
| Mejor regla, B0 (Q5) | **`cola_mas_larga`** | −83.8 |
| Mejor regla, C0 (Q6) | **`min_verde_y_cambiar`** | −322.6 |
| H de `plan_solo_lstm` | **3** | |
| H de `plan_ppo_lstm` | **3** | |
| H de `plan_solo_transformer` | **3** | |
| H de `plan_ppo_transformer` | **3** | |
| Mejor brazo LSTM (P*_lstm) | **`plan_ppo_lstm`**, H = 3 | −1,503.8, frente a `plan_solo_lstm` H = 3: −3,336.6 |
| Mejor brazo Transformer (P*_transformer) | **`plan_ppo_transformer`**, H = 3 | −1,123.9, frente a `plan_solo_transformer` H = 3: −1,360.7 |

Ningún empate exacto intervino. `plan_ppo` usa los PPO de la Fase 3 (13.1). Estos valores quedan
fijos para la etapa 2 (test 25000–25047, 10 réplicas).

## 16. Resultados de la validación del Paso 4 (escrito después de fijar la sección 15)

**Ejecución y fuentes.**

- 13 trabajos con 4 procesos, sin fallos, en **29 min** (10:15–10:44 del 6 de octubre). La
  estimación de la sección 14 era de 1–1.5 h: el microbenchmark de la sección 12 se había medido
  con el equipo ralentizado.
- Resultados por episodio en `docs/results/v2/planning/validation/`; el análisis, en
  `docs/results/v2/planning/validation_analysis.json`.
- Los PPO del sueño de la Fase 3 son los mismos episodios de `ADDENDUM_SUENO_CORREGIDO.md`,
  sección 7, contados aquí con el umbral de esta fase (−3,700). Por eso sus catastróficos difieren
  un poco de los de allí, que usan −3,600.

**Importante: son las réplicas 0–2 en el mismo conjunto en que se eligieron H y el brazo.** Los
números de los brazos elegidos (\*) tienen un sesgo optimista de selección. La comparación válida
es la de la etapa 2, en test, con 10 réplicas.

| Brazo | Media | Mediana | A0 | B0 | C0 | D0 | Catastróficos (< −3,700) | Episodios bloqueados | Controladores bloqueados | Cambios A0/B0/C0/D0 | ms por decisión | s por episodio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `plan_solo_lstm` H3 \* | −3,336.6 | −2,310.0 | −476.5 | −1,229.4 | −588.1 | −1,042.6 | 12/72 | 1/72 | 0/3 | 22.3 / 8.8 / 27.1 / 6.4 | 10 | 5.9 |
| `plan_solo_lstm` H5 | −7,960.7 | −4,752.5 | −2,442.0 | −3,012.5 | −936.1 | −1,570.2 | 50/72 | 6/72 | 0/3 | 18.2 / 6.6 / 25.2 / 6.7 | 19 | 6.5 |
| `plan_solo_lstm` H7 | −10,560.4 | −6,938.0 | −5,144.1 | −2,880.8 | −1,085.6 | −1,449.9 | 66/72 | 4/72 | 0/3 | 15.6 / 7.1 / 23.3 / 7.9 | 31 | 7.3 |
| **`plan_ppo_lstm` H3 \*** | **−1,503.8** | **−1,303.5** | −348.1 | −173.8 | −506.2 | −475.7 | **0/72** | **0/72** | 0/3 | 24.1 / 22.5 / 26.5 / 14.9 | 12 | 6.1 |
| `plan_ppo_lstm` H5 | −1,829.8 | −1,557.5 | −429.4 | −218.4 | −529.1 | −652.9 | 1/72 | 0/72 | 0/3 | 25.2 / 23.3 / 27.0 / 13.5 | 24 | 6.8 |
| `plan_ppo_lstm` H7 | −1,791.0 | −1,604.5 | −419.2 | −245.1 | −552.0 | −574.7 | 1/72 | 0/72 | 0/3 | 24.2 / 23.5 / 26.3 / 14.2 | 38 | 7.6 |
| `plan_solo_transformer` H3 \* | −1,360.7 | −1,199.0 | −330.6 | −318.0 | −556.4 | −155.7 | 1/72 | 0/72 | 0/3 | 23.7 / 10.1 / 24.9 / 10.8 | 11 | 5.8 |
| `plan_solo_transformer` H5 | −2,901.3 | −2,349.0 | −540.5 | −1,021.3 | −833.5 | −506.0 | 17/72 | 1/72 | 0/3 | 21.9 / 7.8 / 22.2 / 9.8 | 19 | 6.5 |
| `plan_solo_transformer` H7 | −4,649.1 | −3,214.0 | −1,133.7 | −1,126.1 | −1,367.5 | −1,021.7 | 28/72 | 0/72 | 0/3 | 19.6 / 8.7 / 20.0 / 9.7 | 27 | 6.9 |
| **`plan_ppo_transformer` H3 \*** | **−1,123.9** | **−1,069.5** | −318.4 | −155.4 | −509.0 | −141.1 | **0/72** | **0/72** | 0/3 | 24.6 / 21.1 / 25.9 / 18.9 | 13 | 5.9 |
| `plan_ppo_transformer` H5 | −1,221.0 | −1,183.0 | −331.9 | −228.1 | −513.8 | −147.3 | 0/72 | 0/72 | 0/3 | 26.1 / 24.5 / 26.0 / 19.8 | 23 | 6.8 |
| `plan_ppo_transformer` H7 | −1,256.6 | −1,176.5 | −342.9 | −254.4 | −489.2 | −170.1 | 0/72 | 0/72 | 0/3 | 25.0 / 24.2 / 25.8 / 20.9 | 34 | 7.3 |
| Sueño LSTM, Fase 3 (10 semillas) | −14,785.3 | −2,969.5 | −3,442.5 | −1,845.2 | −6,100.7 | −3,397.0 | 109/240 | 66/240 | 1/10 | 19.3 / 17.2 / 21.4 / 15.0 | — | 5.5 |
| Sueño LSTM, Fase 3 (semillas 0–2) | −14,335.6 | −8,054.5 | −5,313.6 | −1,497.8 | −1,263.4 | −6,260.7 | 45/72 | 29/72 | 1/3 | 20.7 / 20.7 / 24.8 / 7.4 | — | 5.5 |
| Sueño Transformer, Fase 3 (10 semillas) | −4,025.4 | −1,466.5 | −1,236.7 | −471.1 | −1,905.4 | −412.3 | 20/240 | 4/240 | 0/10 | 20.6 / 18.2 / 23.5 / 18.0 | — | 5.4 |
| Sueño Transformer, Fase 3 (semillas 0–2) | −9,373.8 | −1,620.5 | −3,184.0 | −986.6 | −4,728.7 | −474.5 | 12/72 | 4/72 | 0/3 | 20.5 / 19.9 / 21.9 / 14.8 | — | 5.5 |
| `fijo_2_3` | −1,622.4 | −1,612.0 | −392.8 | −163.2 | −893.5 | −172.9 | 0/24 | 0/24 | — | 23 / 23 / 23 / 23 | — | 4.2 |
| `min_verde_y_cambiar` | −1,247.2 | −1,206.5 | −409.8 | −230.4 | **−322.6** | −284.5 | 0/24 | 0/24 | — | 29 / 29 / 29 / 29 | — | 2.4 |
| `cola_mas_larga` | −1,112.7 | −989.0 | −375.3 | **−83.8** | −561.0 | −92.6 | 0/24 | 0/24 | — | 18.1 / 8.0 / 20.2 / 7.3 | — | 2.4 |
| `max_presion` | −6,329.5 | −6,013.0 | −1,304.5 | −2,028.0 | −1,189.2 | −1,807.9 | 23/24 | 2/24 | — | 12.3 / 4.7 / 16.8 / 4.6 | — | 2.6 |
| `espera_mas_larga` | **−1,033.6** | **−954.0** | −313.8 | −86.2 | −563.6 | −70.1 | 0/24 | 0/24 | — | 18.5 / 8.2 / 20.6 / 7.4 | — | 2.5 |

\* = el H elegido para ese brazo. Los "s por episodio" incluyen SUMO, la planificación y las
consultas de acuerdo con las referencias, con 4 procesos a la vez.

**Medias por réplica (s0, s1, s2):**

| Brazo | H3 | H5 | H7 |
|---|---|---|---|
| `plan_ppo_lstm` | −1,303, −2,031, −1,177 | −1,410, −2,440, −1,640 | −1,521, −2,375, −1,477 |
| `plan_ppo_transformer` | −1,138, −1,103, −1,131 | −1,208, −1,250, −1,206 | −1,180, −1,319, −1,270 |
| `plan_solo_lstm` | −2,299, −2,851, −4,860 | — | — |
| `plan_solo_transformer` | −1,499, −1,203, −1,380 | — | — |

**Bloqueo** (las reglas de la Fase 3):

- **Ningún planificador queda bloqueado como controlador**: ninguna réplica tiene una media de
  menos de 3 cambios por episodio en ningún semáforo.
- **Los valores más bajos:**
  - `plan_solo_lstm` H7 s2: 4.4 cambios por episodio en un semáforo.
  - `plan_ppo_lstm` s1 con H5 y H7: 5.8 y 5.6.
  - `plan_solo_lstm` H3: 5.6–6.4.
- **A nivel de episodio:** `plan_solo_lstm` 1, 6 y 4 de 72 (H3, H5 y H7) y `plan_solo_transformer`
  H5 1/72. Los `plan_ppo`, 0/72 en todos los H.
- **`plan_solo` cambia poco en B0 y D0** (6–11 cambios por episodio), parecido a
  `cola_mas_larga` y `espera_mas_larga` (≈ 8). Con H largo la continuación "mantener" lo empuja a
  mantener más, y el costo crece.

**Acuerdo con las referencias en los mismos estados, H = 3** (A0 / B0 / C0 / D0; 0.5 es azar):

| Referencia | `plan_solo_lstm` | `plan_ppo_lstm` | `plan_solo_transformer` | `plan_ppo_transformer` |
|---|---|---|---|---|
| `fijo_2_3` | 0.53 / 0.57 / 0.48 / 0.57 | 0.53 / 0.52 / 0.50 / 0.56 | 0.48 / 0.56 / 0.49 / 0.56 | 0.51 / 0.52 / 0.49 / 0.53 |
| `min_verde_y_cambiar` | 0.71 / 0.27 / **0.86** / 0.21 | 0.69 / 0.68 / **0.78** / 0.45 | **0.74** / 0.31 / 0.73 / 0.34 | **0.73** / 0.60 / **0.73** / 0.59 |
| `cola_mas_larga` | 0.55 / 0.59 / 0.54 / 0.58 | 0.63 / 0.54 / 0.61 / 0.56 | 0.57 / 0.66 / 0.64 / 0.70 | 0.60 / 0.58 / 0.65 / 0.57 |
| `max_presion` | 0.64 / **0.80** / 0.52 / **0.84** | 0.62 / 0.63 / 0.56 / **0.73** | 0.58 / **0.79** / 0.61 / **0.79** | 0.56 / 0.57 / 0.57 / 0.66 |
| `espera_mas_larga` | 0.55 / 0.59 / 0.55 / 0.57 | 0.63 / 0.54 / 0.61 / 0.56 | 0.57 / 0.66 / 0.64 / 0.70 | 0.60 / 0.58 / 0.65 / 0.57 |

**Lectura descriptiva** (validación y 3 réplicas; no es un resultado de test):

1. **Planificar desde el estado real con H = 3 elimina el bloqueo y los catastróficos del PPO del
   sueño**:

   | Arquitectura | PPO del sueño, Fase 3, s0–s2 | `plan_ppo`, H = 3 |
   |---|---|---|
   | LSTM | −14,336; 45/72 catastróficos; 29/72 episodios bloqueados | −1,504; 0/72; 0/72 |
   | Transformer | −9,374; 12/72 catastróficos | −1,124; 0/72 |

2. **Horizonte más largo, peor, en los 4 brazos.** Es consistente con los diagnósticos de
   fidelidad: el error del modelo crece con los pasos imaginados. Con `plan_solo` es mucho más
   marcado, porque la continuación "mantener" lleva el modelo a fases largas.
3. **Frente a las reglas:**
   - `plan_ppo_transformer` H3 (−1,124) queda cerca de `cola_mas_larga` (−1,113) y de
     `espera_mas_larga` (−1,034), sin superarlas.
   - `plan_ppo_lstm` H3 (−1,504) queda entre `fijo_2_3` (−1,622) y `min_verde_y_cambiar`
     (−1,247).
   - En C0, el mejor (−506 a −509) no alcanza a `min_verde_y_cambiar` (−323), aunque se le parece
     en las decisiones (0.73–0.78).
4. **Costo:** de 10 a 38 ms por decisión con un hilo por proceso (≈ 6 s por episodio con SUMO y
   el acuerdo). El costo crece con H y es similar entre arquitecturas.

**Nada de esto decide nada.** Las decisiones de la etapa 2 (sección 15) ya están fijadas, y la
comparación con test se hará con las 10 réplicas.

## 17. Etapa 2: plan de la evaluación en el test nuevo (escrito el 6 de octubre de 2026, antes de simular en 25000–25047)

Nada de lo fijado en las secciones 6, 13, 14 y 15 cambia: ni reglas, ni H, ni brazos.

**Qué se evalúa en 25000–25047 (48 escenarios), una sola vez por controlador:**

| Política | Controladores | Episodios |
|---|---|---|
| Los 4 brazos del planificador (`plan_solo_lstm`, `plan_ppo_lstm`, `plan_solo_transformer`, `plan_ppo_transformer`), con **H = 3** y las **10 réplicas** (s0–s9); `plan_ppo` continúa con los PPO de la Fase 3 (13.1) | 40 | 1,920 |
| Los 20 PPO del sueño de la Fase 3 (`sueno_lstm`, `sueno_transformer`, `models/checkpoints/v2/control/dream_<arq>_s<i>`) | 20 | 960 |
| Los 20 RL directos de la Fase 3 (`directo_10k`, `directo_30k`) | 20 | 960 |
| Las 5 reglas | 5 | 240 |

- **Los episodios del test viejo (22000–22023) no se reutilizan.** No se usan 20000–23029 ni OOD.
- Todo con acuerdo con las referencias en los mismos estados.

**Comparaciones planificadas** (sección 6), con los mejores brazos fijados en la sección 15:
P*_lstm = `plan_ppo_lstm` H3 y P*_transformer = `plan_ppo_transformer` H3.

| # | Comparación | Métrica |
|---|---|---|
| Q1_lstm | `plan_ppo_lstm` − `sueno_lstm` | total |
| Q2_lstm | `plan_ppo_lstm` − `directo_30k` | total |
| Q3_lstm | `plan_ppo_lstm` − `directo_10k` | total |
| Q4_lstm | `plan_ppo_lstm` − `espera_mas_larga` | total |
| Q5_lstm | `plan_ppo_lstm` − `cola_mas_larga` | B0 |
| Q6_lstm | `plan_ppo_lstm` − `min_verde_y_cambiar` | C0 |
| Q1_transformer | `plan_ppo_transformer` − `sueno_transformer` | total |
| Q2_transformer | `plan_ppo_transformer` − `directo_30k` | total |
| Q3_transformer | `plan_ppo_transformer` − `directo_10k` | total |
| Q4_transformer | `plan_ppo_transformer` − `espera_mas_larga` | total |
| Q5_transformer | `plan_ppo_transformer` − `cola_mas_larga` | B0 |
| Q6_transformer | `plan_ppo_transformer` − `min_verde_y_cambiar` | C0 |

- **12 comparaciones → Bonferroni α' = 0.05/12 = 0.004167; IC al 99.583%.**
- **Test principal:** Welch sobre las 10 medias por réplica o semilla; contra una regla, t de una
  muestra.
- t pareada y Wilcoxon por escenario (48): **solo secundarios, marcados como pseudorreplicación**.
- Sin significancia en el test principal, se reporta "brecha no significativa".
- **Se reportan TODOS los brazos**, sin regla de desempate que elija uno. `plan_solo_lstm` y
  `plan_solo_transformer` se comparan con los mismos objetivos, de forma descriptiva y sin
  corrección.
- **Métricas por brazo:**
  - retorno medio y mediana, y por intersección;
  - catastróficos (< −3,700, sección 15);
  - bloqueados, con las reglas de la Fase 3 (controlador y episodio);
  - cambios de fase por semáforo y ms por decisión;
  - acuerdo con las referencias;
  - medias por semilla y la semilla que más aporta a la varianza.
- **Interacciones reales por réplica o semilla**, las dos contabilidades:

  | | Sin compartir | Compartiendo |
  |---|---|---|
  | Planificador (sección 13) | 21,240 | 6,552 |
  | Sueño | 12,600 | 3,960 |
  | Directo 10k | 13,240 | 13,240 |
  | Directo 30k | 39,208 | 39,208 |

**Cómo:**

- `python scripts/v2/run_planning_test.py --workers 4`: 17 trabajos de unos 240 episodios. Cada
  política aprendida se parte en dos mitades de 5 semillas.
- Cada trabajo es `evaluate_control_v2.py --split test_v21 --confirm-held-out --reference-agreement`
  en su propio proceso, con un hilo.
- Reanudable. Se niega a arrancar sin corriente o con 2 GB o menos.
- Salida: `docs/results/v2/planning/test/`.
- Análisis: `scripts/v2/analyze_planning_test.py`, escrito y probado con datos sintéticos antes de
  simular.

**Estimación:** la validación de la sección 16 dio unos 1.8 s por episodio con 4 procesos. Para
4,080 episodios, **≈ 2–2.5 h**.

**Integridad:** md5 antes y después de los 40 controladores y los resultados de la Fase 3.

### 17.1 Etapa 2 del planificador: evaluación en test iniciada, 2026-10-05 12:23 (hora local)

Hasta este momento no se ha simulado nada en 25000–25047. Lo siguiente es la evaluación única de
la sección 17 (`run_planning_test.py`, con `--confirm-held-out`).
