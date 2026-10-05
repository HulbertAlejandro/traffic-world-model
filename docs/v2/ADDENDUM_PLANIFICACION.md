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
