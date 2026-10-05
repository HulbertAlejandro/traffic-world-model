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
