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

## Siguiente paso recomendado — Controlador PPO dentro del Dream Environment

Con `DreamEnvironment` implementado y probado, el orden de la propuesta indica que
sigue el controlador:

- [ ] Entrenar un controlador PPO (Stable-Baselines3) usando `DreamEnvironment` como
      entorno de entrenamiento, sin tocar SUMO.
- [ ] Definir el criterio de evaluación del controlador entrenado en el Dream
      Environment: ¿se evalúa primero dentro del propio Dream Environment, o se pasa
      directo a SUMO real?
- [ ] Decidir si `max_dream_steps=10` es suficiente horizonte de entrenamiento para PPO,
      o si conviene revisarlo una vez haya resultados preliminares.

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