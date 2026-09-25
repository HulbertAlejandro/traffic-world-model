# TODO.md

## Ya resuelto (referencia, no acción)

- [x] Experimento 0 completado: el Autoencoder se mantiene en el sistema final,
      confirmado con `reward_mse` consistentemente mejor en los 10 horizontes
      evaluados (ver PROJECT_STATUS.md para el detalle numérico).
- [x] Fix de normalización de recompensa en el LSTM.
- [x] Fix de lectura de fase del semáforo (bug crítico, ver historial de commits).
- [x] Dataset regenerado con 40 episodios.
- [x] Sobreajuste residual del LSTM investigado y cerrado (`weight_decay` + early
      stopping aplicados; conclusión: no era el cuello de botella real, ver
      PROJECT_STATUS.md).
- [x] Dream Environment (primera versión): `predict_next_step` extraído y reutilizado,
      `DreamEnvironment` compatible con `gymnasium.Env`, sembrado con episodios reales,
      `max_dream_steps=10`, 27/27 tests en verde (ver PROJECT_STATUS.md para el
      detalle de las tres decisiones de diseño que esto resolvió).
- [x] Sanity check manual del Dream Environment con el checkpoint real: encontró
      extrapolación OOD del LSTM ante rachas largas de acción constante. Mitigado con
      `max_dream_steps` 7 (antes 10) y recorte de recompensa al rango empírico
      `[-165.05, 1.00]` (percentiles 1/99 reales), con seguimiento en
      `info["consecutive_action_streak"]` e `info["reward_clipped"]`. 29/29 tests en
      verde, commiteado y subido. Ver PROJECT_STATUS.md para el detalle completo,
      incluida la limitación honesta que queda: el recorte acota el problema, no lo
      corrige de raíz.
- [x] Controlador PPO implementado y entrenado dentro del Dream Environment
      (`configs/controller.py`, `training/train_controller.py`,
      `scripts/evaluate_controller.py`, commit `86228ae`, 33/33 tests en verde).
      Entrenado 50,176 timesteps con curva de validación ruidosa y no monótona.
      Supera a las 4 políticas de referencia en la autoevaluación dentro del propio
      Dream Environment, pero con una salvedad seria — ver PROJECT_STATUS.md: su tasa
      de `reward_clipped` (13.3%) es más alta que la de la política "alternando"
      (4.8%) pese a evitar rachas largas.
- [x] Investigada la hipótesis de explotación del recorte de recompensa, con la
      magnitud del recorte (no solo su frecuencia): `info["raw_predicted_reward"]`
      expuesto, `scripts/analyze_controller_actions.py` creado, commit `8044262`,
      34/34 tests en verde. Resultado: la magnitud del recorte en PPO es pequeña
      (cercana a "alternando", muy por debajo de las políticas constantes) —
      **debilita pero no descarta** la sospecha de explotación (no distingue de un
      patrón temporal más sutil). Ver PROJECT_STATUS.md para el detalle completo.
- [x] Controlador PPO evaluado contra SUMO real: bloque completo, **con una corrección
      posterior que cambió la conclusión**. Puente `EncodedTrafficEnvironment` (commit
      `915bd79`); diagnóstico de episodios catastróficos de v1 por rachas de acción y
      fix `dream_max_steps=20` → v2 (commit `8d84d52`). Después se encontró que el
      puente no normalizaba con `scaler.pkl` (fix en commit `990c6e5`), y al re-evaluar
      todo con el puente corregido: v1 (`dream_max_steps=7`) no tiene ningún episodio
      catastrófico y supera a v2 y a tiempo fijo en las 30 comparaciones. El
      diagnóstico de rachas y el fix eran artefactos del bug; `dream_max_steps` se
      revirtió a 7. **Resultado oficial: PPO v1, -287.76 ± 23.87 (`seed_base=3000`) /
      -293.35 ± 41.28 (`seed_base=5000`)**, frente a tiempo fijo -570.27 / -605.27; nunca
      mejor en throughput. Los números anteriores (-290.05/-316.76 de v2 con bug y
      -421.69/-419.65 de v2 corregido) ya no son el resultado del método. Ver
      PROJECT_STATUS.md, sección "Controlador PPO contra SUMO real: historia completa y
      resultado oficial (v1)", incluida la lección metodológica.
- [x] Baseline de RL directo (Sección 18): `ReseedingWrapper`, `train_controller_direct.py`
      y `evaluate_direct_vs_dream.py` (commits `6c648d4`, `1a2874c`), 39/39 tests en
      verde. **Hallazgo**: ni v1 ni el PPO directo aprendieron control dependiente del
      estado. Ambos convergen a la regla "pedir siempre la fase contraria", la mejor
      política encontrada en este escenario (100% y 99.2% de acuerdo con ella en los pasos
      donde la acción afecta al tráfico). **Sin ahorro demostrable de interacciones
      reales** en este escenario: 2,400 del World Model (dataset) frente a ≤2,600 del RL
      directo. Ver PROJECT_STATUS.md, sección "Baseline de RL directo y hallazgo final".
- [x] Evaluación final en SUMO real: tiempo fijo vs. RL directo vs. World Model (más la
      regla "fase contraria" como referencia). Tabla final en la misma sección de
      PROJECT_STATUS.md.
- [x] Escenario de demanda asimétrica (500/150 veh/h; 700/150 descartado por saturar la
      vía principal) con el pipeline completo re-ejecutado desde cero (commits `febef8e`,
      `6c753d2`, `f58347e`; `REWARD_CLIP_MIN` recalculado a -266.13). **Resultado**: la
      regla trivial pasa a ser la peor política y los dos PPO ya no la siguen; ambos
      superan en promedio a tiempo fijo, pero con episodios catastróficos (7/30 y 5/30)
      cuya causa no se identificó por completo. Ver PROJECT_STATUS.md, sección "Escenario
      asimétrico, primera ronda".
- [x] Escenario asimétrico, segunda ronda (commits `2a63328` a `9b6a389`, 45/45 tests):
      semilla de SUMO por episodio en `collect_dataset.py` y dataset de 80 episodios;
      Experimento 0 repetido (Autoencoder gana 9/10); selección del checkpoint del sueño
      en SUMO real (el reward imaginado no predecía el real, Pearson +0.08); `eval_seed`
      en `DreamEnvironment`; `VecNormalize` de recompensa en ambos PPO y de observaciones
      en el directo; verificación con 3 semillas por método. **Resultado: PPO del sueño
      -326.79, PPO directo -453.74, tiempo fijo -411.27 (90/90/30 episodios)**; ventaja
      clara a nivel de episodio, no significativa a nivel de semilla. Ver PROJECT_STATUS.md,
      sección "Escenario asimétrico, segunda ronda".
- [x] Más presupuesto de entrenamiento real para el RL directo: verificado con 30,000
      pasos (3x) en las 3 semillas, fuera del repositorio y sin tocar los checkpoints
      oficiales. **Resultado:** el RL directo alcanza un desempeño comparable al World
      Model (-335.24 frente a -326.79; p = 0.71 a nivel de semilla), la desviación de sus
      medias por semilla baja de 96.2 a 30.6, la semilla 2 deja de colapsar y los
      episodios catastróficos bajan de 23/90 a 11/90, pero consume ~8.5 veces más
      interacciones reales (39,000 frente a ~4,600 por semilla). La ventaja del World
      Model es de eficiencia en interacciones. Ver PROJECT_STATUS.md, sección
      "Verificación: RL directo con 3x presupuesto".

## Siguiente paso recomendado

- [x] **(a)** Documentación final (`docs/DOCUMENTACION_PROYECTO.md`, `docs/PROPUESTA.md`),
      subida en el commit `9ca20ac`.
- [ ] **(b)** Seguir investigando alguna de las limitaciones abiertas (PROJECT_STATUS.md,
      sección "Escenario asimétrico, segunda ronda", punto 9): episodios catastróficos,
      función de valor del PPO directo, o más semillas por método (el presupuesto del
      RL directo ya se verificó).
- [x] **Experimento 3:** Transformer y TSMixer como alternativas a la LSTM (commits
      `90d7641` a `f49cf09`, 62/62 tests). **Resultado: se mantiene la LSTM**, que gana en
      `reward_mse` en los 10 horizontes frente a ambas (reducción mediana de 38.1% frente
      al Transformer y de 56.1% frente a TSMixer; umbral para reemplazarla: 7.3%).
      Convergencia de TSMixer verificada con 300 épocas sin early stopping: sigue perdiendo
      10/10. Hipótesis de la propuesta, ahora evaluadas: **H4 confirmada** (el Transformer
      se comporta distinto, en contra suya en la recompensa); **H6 rechazada** (TSMixer es
      el peor de los tres en todos los horizontes). Hallazgo: el Transformer tiene mejor
      `latent_mse` y mejor pérdida de validación, pero peor `reward_mse`, el mismo
      desacople que el reward imaginado del PPO (Pearson +0.08). Ver PROJECT_STATUS.md,
      sección "Experimento 3".
- [ ] Reflejar el Experimento 3 en `docs/PROPUESTA.md` (H4, H6, Secciones 20, 24 y 30).
      Lo revisa el autor por separado.

## Pendiente, no bloqueante

- [x] Actualizar `DOCUMENTACION_PROYECTO.md` con LSTM, bug de fase, Experimento 0,
      cierre del sobreajuste, y Dream Environment (resuelto en el commit `9ca20ac`).
- [ ] Investigar y resolver el tema del colaborador/app desconocido en GitHub.
- [x] Corregir la documentación de ProjectActionSpace y cualquier referencia a
      'mantener/cambiar' -- la acción es en realidad el índice de fase verde destino
      (confirmado en sumo_rl.TrafficSignal.set_next_phase). No es bloqueante porque el
      pipeline completo usa la convención de forma consistente, pero la documentación es
      engañosa para cualquiera que lea el código después. Resuelto en `c253d88`
      (docstrings y comentarios, sin cambio de comportamiento) y en `30c3fae` (Sección 12
      de `docs/PROPUESTA.md`).
- [ ] Corregir `ProjectRewardFunction.phase_change`: hoy penaliza pedir la fase 1
      (`action == 1`), no cambiar efectivamente de fase. Misma raíz que el punto
      anterior; ver PROJECT_STATUS.md, sección del baseline de RL directo, punto 7.
- [ ] Entender por qué ambos PPO de la primera ronda asimétrica nunca sostenían el verde de
      la fase 1 más de 8 s (la duración mínima posible de una fase); volver a medirlo con los
      checkpoints actuales.

## Después (orden según la propuesta del proyecto)

- [ ] Más semillas por método (siguen siendo 3; el presupuesto ya se verificó, ver
      "Ya resuelto").
- [ ] Curva de desempeño frente a interacciones reales del RL directo (hoy solo dos
      puntos: 13,000 y 39,000 por semilla).
- [ ] Medir el momento de los cambios de fase respecto a las colas de cada brazo, comparando
      episodios catastróficos con buenos (quedan 5/90 en el sueño y 23/90 en el directo).
- [ ] Demanda variable en el tiempo (la asimetría ya está implementada; la variación
      temporal no).
