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
- [x] Controlador PPO evaluado contra SUMO real y validado — bloque completo: puente
      `EncodedTrafficEnvironment` (commit `915bd79`), diagnóstico de 4 episodios
      catastróficos (3 de 4 por rachas de acción de 6-14 pasos no vistas en
      entrenamiento; el cuarto, seed=3005, un pico puntual de `waiting_total`
      documentado como limitación conocida sin resolver), fix
      `ControllerConfig.dream_max_steps=20` (commit `8d84d52`), y verificación en dos
      corridas con semillas distintas (3000 y 5000) confirmando que el arreglo
      generaliza. 35/35 tests en verde. **Resultado final**: PPO v2 supera a tiempo
      fijo en reward/espera/cola de forma consistente, pero nunca en throughput — ver
      PROJECT_STATUS.md para el detalle numérico completo y la interpretación honesta.

## Siguiente paso recomendado — Baseline de RL directo (PPO sin Dream Environment)

Con el controlador PPO ya validado contra SUMO real, falta la pieza que la propuesta
pide explícitamente (Sección 18) para responder la pregunta de investigación completa:

- [ ] Entrenar un PPO directamente contra `TrafficEnvironment` (SUMO real), sin pasar
      por el Dream Environment, como baseline de "RL directo".
- [ ] Comparar ese baseline contra el PPO v2 (entrenado en el sueño) usando el mismo
      protocolo de evaluación ya validado (scripts/evaluate_controller_sumo.py,
      semillas de EVALUACIÓN 3000 y 5000) -- la semilla de entrenamiento del baseline
      de RL directo es independiente y no necesita coincidir con nada ya usado.
- [ ] Con ese resultado, responder la pregunta de investigación del proyecto: ¿el
      World Model realmente redujo las interacciones necesarias con SUMO frente a
      entrenar RL directo, sin perder desempeño de control?

## Pendiente, no bloqueante

- [ ] Actualizar `DOCUMENTACION_PROYECTO.md` con LSTM, bug de fase, Experimento 0,
      cierre del sobreajuste, y Dream Environment.
- [ ] Investigar y resolver el tema del colaborador/app desconocido en GitHub.

## Después (orden según la propuesta del proyecto)

- [ ] Evaluación final en SUMO real: tiempo fijo vs. RL directo vs. World Model.
- [ ] Extensiones opcionales (misma prioridad): sustituir el LSTM por Transformer, y
      por separado, por TSMixer — reutilizando el mismo protocolo de comparación que
      ya se usó en el Experimento 0 (mismos hiperparámetros, mismo criterio de
      `reward_mse`, mismo split de test).