# TODO.md

## Ya resuelto (referencia, no acción)

- [x] Experimento 0 completado: el Autoencoder se mantiene en el sistema final,
      confirmado con `reward_mse` consistentemente mejor en los 10 horizontes
      evaluados (ver PROJECT_STATUS.md para el detalle numérico).
- [x] Fix de normalización de recompensa en el LSTM.
- [x] Fix de lectura de fase del semáforo (bug crítico, ver historial de commits).
- [x] Dataset regenerado con 40 episodios.

## Siguiente paso recomendado — Dream Environment

Con el Autoencoder confirmado y el LSTM (sobre `z`) validado como el modelo del
sistema final, el orden de la propuesta indica que sigue el Dream Environment:

- [ ] Diseñar el mecanismo de "imaginación": dado un `z_t` real (del Encoder) y una
      lista de acciones candidatas, usar el LSTM para proyectar varios pasos hacia
      adelante sin tocar SUMO, sumando las recompensas predichas por cada trayectoria
      imaginada.
- [ ] Definir el horizonte de imaginación a usar (¿el mismo `sequence_length=16`, o uno
      más corto para decisiones en tiempo real?).
- [ ] Decidir la interfaz exacta: ¿el Dream Environment expone algo similar a
      `TrafficEnvironment` (con `reset`/`step`) pero operando en el espacio latente, o
      es una función más simple de "evaluar una secuencia de acciones candidata"?

## Pendiente, no bloqueante

- [ ] Sobreajuste residual del LSTM (brecha train/val ~3x desde el fix de fase).
- [ ] Actualizar `DOCUMENTACION_PROYECTO.md` con LSTM, bug de fase, y Experimento 0.
- [ ] Investigar y resolver el tema del colaborador/app desconocido en GitHub.

## Después (orden según la propuesta del proyecto)

- [ ] Controlador PPO (Stable-Baselines3) entrenado dentro del Dream Environment.
- [ ] Evaluación final en SUMO real: tiempo fijo vs. RL directo vs. World Model.
- [ ] Extensiones opcionales (misma prioridad): sustituir el LSTM por Transformer, y
      por separado, por TSMixer — reutilizando el mismo protocolo de comparación que
      ya se usó en el Experimento 0 (mismos hiperparámetros, mismo criterio de
      `reward_mse`, mismo split de test).