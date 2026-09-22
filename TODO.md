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
      verde. **Pendiente de commit** — aprobado por el autor, ver PROJECT_STATUS.md
      para el detalle completo, incluida la limitación honesta que queda: el recorte
      acota el problema, no lo corrige de raíz.

## Siguiente paso recomendado — Confirmar commit, luego Controlador PPO dentro del Dream Environment

- [ ] Commitear y subir las mitigaciones de extrapolación OOD del Dream Environment
      (cambios ya aprobados, ver arriba).

Con `DreamEnvironment` implementado y probado, el orden de la propuesta indica que
sigue el controlador:

- [ ] Entrenar un controlador PPO (Stable-Baselines3) usando `DreamEnvironment` como
      entorno de entrenamiento, sin tocar SUMO.
- [ ] Definir el criterio de evaluación del controlador entrenado en el Dream
      Environment: ¿se evalúa primero dentro del propio Dream Environment, o se pasa
      directo a SUMO real?
- [ ] Decidir si `max_dream_steps=7` es suficiente horizonte de entrenamiento para PPO,
      o si conviene revisarlo una vez haya resultados preliminares.
- [ ] Monitorear `info["reward_clipped"]` e `info["consecutive_action_streak"]` durante
      el entrenamiento del PPO — si el agente pasa mucho tiempo en zonas de racha larga
      (que el recorte solo acota, no corrige), es señal de que la limitación conocida
      del Dream Environment está afectando el entrenamiento real, no solo el sanity
      check manual.

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