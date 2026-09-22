# TODO.md

## Ya resuelto (referencia, no acción)

- [x] Bug crítico: `CustomStateBuilder` leía la fase del semáforo desde una fuente de
      TraCI que `sumo_rl` nunca actualiza. Corregido: ahora lee `TrafficSignal.green_phase`
      / `time_since_last_phase_change` a través de `sumo_rl_env`. Verificado con
      ejecución real.
- [x] Dataset regenerado con 40 episodios (antes 20) tras el fix.
- [x] Pipeline completo re-entrenado (Autoencoder + LSTM) sobre datos limpios.
      Resultado: el World Model supera al baseline persistente en los 10 horizontes,
      con márgenes de 5x-17x según la métrica — resultado confiable, ver
      PROJECT_STATUS.md para el detalle numérico completo.
- [x] Normalización de la recompensa en la pérdida del LSTM (fix anterior, ya validado
      dos veces: antes y después del fix de fase).

## Siguiente paso recomendado — Experimento 0

Con el dataset ya confiable, este es el orden correcto según la propuesta del proyecto:

- [ ] Entrenar un LSTM equivalente pero **sobre el estado crudo normalizado** (sin pasar
      por el Autoencoder), usando exactamente el mismo dataset, mismo `sequence_length`,
      mismos hiperparámetros de entrenamiento.
- [ ] Comparar su error de predicción (mismo protocolo de `evaluate_world_model.py`:
      baseline persistente, 10 horizontes) contra el LSTM actual (sobre `z`).
- [ ] Decidir con esa evidencia si el Autoencoder se queda en el sistema final o se
      descarta, según el criterio ya establecido: se mantiene solo si mejora medible y
      consistentemente la predicción frente al estado crudo.

## Pendiente, no bloqueante

- [ ] Sobreajuste residual del LSTM (brecha train/val ~3x, `val_loss` estancado desde
      la época ~70). Candidatos: early stopping, o recolectar aún más episodios si hay
      tiempo. No urgente — el resultado actual ya es válido para avanzar.
- [ ] Actualizar `DOCUMENTACION_PROYECTO.md` con todo el bloque LSTM + el bug de fase +
      los resultados finales (sigue reflejando un estado muy anterior).
- [ ] Investigar y resolver el tema del colaborador/app desconocido en GitHub.

## Después (orden según la propuesta del proyecto)

- [ ] Dream Environment: usar el LSTM (con acciones candidatas) para imaginar
      trayectorias sin tocar SUMO.
- [ ] Controlador PPO (Stable-Baselines3) entrenado dentro del Dream Environment.
- [ ] Evaluación final en SUMO real: tiempo fijo vs. RL directo vs. World Model.
- [ ] Extensiones opcionales (misma prioridad): sustituir el LSTM por Transformer, y
      por separado, por TSMixer.