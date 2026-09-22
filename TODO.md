# TODO.md

## Inmediato (bloqueante) — el hallazgo más importante en este momento

- [ ] Aplicar la normalización de la recompensa en `training/train_world_model.py`:
      calcular `reward_mean`/`reward_std` solo con el split de entrenamiento, guardarlos
      en un `reward_scaler.json` junto al checkpoint, y usar el objetivo normalizado en
      la pérdida (`mse(pred_latent, target_latent) + mse(pred_reward, target_reward_normalizado)`).
- [ ] Actualizar `evaluation/world_model_evaluation.py` para desnormalizar la predicción
      de recompensa del modelo antes de compararla contra la recompensa real, usando el
      `reward_scaler.json` guardado.
- [ ] Re-entrenar el LSTM (`python training/train_world_model.py`) con el fix aplicado.
- [ ] Re-correr la evaluación (`python scripts/evaluate_world_model.py`) y comparar:
      ¿el modelo ya supera al baseline persistente en el horizonte 1, tanto en latente
      como en recompensa? Si no, seguir investigando antes de avanzar.

## Antes de sacar conclusiones fuertes del Experimento 1

- [ ] Confirmar cuántos episodios existen en `datasets/raw/` actualmente. Si siguen
      siendo ~20 (el default), generar más (30-40+) con `python scripts/collect_dataset.py`
      para que el conjunto de prueba no dependa de solo 3 episodios.
- [ ] Verificación de la fase del semáforo: contar valores únicos en `states[:, 20:24]`
      del dataset real. Si el semáforo casi no cambió de fase durante la recolección con
      acciones aleatorias, investigar por qué antes de re-entrenar con más datos.
- [ ] Con el fix de normalización aplicado y más datos, volver a correr el pipeline
      completo end-to-end y revisar de nuevo `results/world_model_compounding_error.png`.

## Documentación

- [ ] Actualizar `DOCUMENTACION_PROYECTO.md` para incluir el bloque completo del LSTM
      y su evaluación (está desactualizado, se generó antes de estos commits).
- [ ] Investigar y resolver el tema del colaborador/app desconocido en GitHub.

## Después (orden según la propuesta del proyecto), una vez el LSTM supere al baseline

- [ ] Experimento 0: comparar LSTM entrenado sobre el estado crudo normalizado vs.
      sobre `z` del Autoencoder — decidir si el Autoencoder se queda en el sistema
      final o se descarta. (Nota: esto quedó pospuesto por el problema de escala de la
      recompensa, que hay que resolver primero para que la comparación sea justa.)
- [ ] Dream Environment: usar el LSTM (con acciones candidatas) para imaginar
      trayectorias sin tocar SUMO.
- [ ] Controlador PPO (Stable-Baselines3) entrenado dentro del Dream Environment.
- [ ] Evaluación final en SUMO real, comparando: control de tiempo fijo vs. RL directo
      vs. World Model.
- [ ] Extensiones opcionales (misma prioridad entre sí): sustituir el LSTM por
      Transformer, y por separado, por TSMixer, reutilizando el mismo pipeline y
      métricas.
