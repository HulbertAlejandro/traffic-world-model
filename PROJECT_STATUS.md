# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: commit `98770d9` (`Add world model evaluation pipeline and LSTM
validation`), confirmado por `git clone` completo del repositorio.

## ✅ Completo y verificado con evidencia de ejecución real (no solo tests sintéticos)

- **Entorno**: `TrafficEnvironment` + `CustomStateBuilder` + `TrafficState` (26 dims).
  Throughput correcto (arrivals, no vehículos presentes), sin índices mágicos,
  `observation_space`/`action_space` como `gymnasium.spaces` reales.
- **Pipeline de dataset**: recolección → split por episodio (sin fuga, verificado) →
  normalización (scaler ajustado solo con train, verificado) → `TransitionDataset`.
- **Autoencoder**: entrenado 100 épocas contra datos reales. Train loss final 0.0298,
  validation 0.0827 — ambas curvas bajando juntas, sin señal de overfitting.
  ⚠️ Pendiente de confirmar si las columnas de fase (one-hot) tienen variación real en
  el dataset recolectado (ver sección Pendiente).
- **`encode_latent_dataset.py`**: bug de `load_autoencoder` (faltaba `input_dim`) **ya
  corregido** en el commit `98770d9`. Corrió exitosamente sobre train/validation/test.
- **`LatentSequenceDataset`**: ventanas por episodio, episodios cortos descartados,
  con tests de regresión de límites de episodio.
- **`LatentDynamicsLSTM`**: entrenado 100 épocas contra datos reales latentes. Predice
  `(ẑ_{t+1}, r̂_{t+1})`, `action_dim` desde config, `Protocol TemporalModel` para
  intercambiabilidad futura con Transformer/TSMixer.
- **`evaluation/world_model_evaluation.py`**: implementado y corrido — compara el LSTM
  contra un baseline persistente a 1 y varios pasos (horizonte 1 a 10), calcula
  compounding error, genera `results/world_model_evaluation.json` y
  `results/world_model_compounding_error.png`. Diseño revisado y correcto (indexación
  del baseline de recompensa evita un falso "error 0"; usa acciones reales, no
  imaginadas).
- **Tests**: 22 pruebas, todas pasando (`22 passed in 3.78s`, confirmado con salida
  real de `pytest -q` del usuario).

## 🔴 Problema real encontrado — el LSTM no aprendió bien la dinámica latente

**Evidencia (salida real del entrenamiento y evaluación):**
- Entrenamiento del LSTM: train_loss cae de ~1250 a ~133 en 100 épocas; validation_loss
  solo baja de ~6440 a ~4182, de forma ruidosa (no monotónica) — brecha train/validation
  enorme, señal de sobreajuste.
- Evaluación (horizonte 1): `latent_mse (modelo) = 1.107` vs. `latent_mse (baseline
  persistente) = 0.994` — **el modelo pierde contra el baseline trivial "nada cambia"**.
  Lo mismo para recompensa: `1307.87` (modelo) vs. `928.05` (baseline).

**Diagnóstico (con evidencia numérica, ver conversación completa para el detalle):**
La pérdida de entrenamiento combina `mse(latente) + 1.0 * mse(recompensa)` sin
normalizar. El error cuadrático de recompensa es ~1000 veces mayor en magnitud bruta
que el error latente (recompensa sin normalizar tiene escala de decenas/cientos;
latente ya está normalizado a varianza ~1). Con peso 1.0, el término de recompensa
domina casi por completo el gradiente de entrenamiento — el LSTM apenas recibe señal
para aprender la dinámica de `z`, y sobreajusta al intentar ajustar exactamente valores
de recompensa de escala grande con pocos datos.

**Fix propuesto, NO aplicado todavía:**
Normalizar la recompensa (media/desviación calculadas solo con el split de
entrenamiento, mismo principio que ya se usa para el estado) antes de calcular la
pérdida en `training/train_world_model.py`, guardando un `reward_scaler.json` junto al
checkpoint. `evaluation/world_model_evaluation.py` debe desnormalizar la predicción del
modelo antes de compararla contra la recompensa real, para seguir reportando el error
en unidades interpretables.

## 🟡 Sin confirmar / pendiente de verificar

1. ¿El fix de normalización de recompensa ya se aplicó? — no, pendiente en este handoff.
2. Verificación de la fase del semáforo: contar valores únicos en `states[:, 20:24]` del
   dataset real recolectado, para confirmar o descartar que el semáforo casi no cambió
   de fase con acciones aleatorias — sigue sin verificarse.
3. El conjunto de prueba tiene solo **3 episodios** (con el `num_episodes=20` por
   defecto y ratios 70/15/15) — los 105 puntos por horizonte en la evaluación vienen de
   ventanas deslizantes altamente correlacionadas dentro de esos 3 episodios, no de
   observaciones independientes. Recomendado: recolectar más episodios (30-40+) antes
   de sacar conclusiones fuertes.
4. Cuenta o app desconocida apareció como colaborador en GitHub — sin resolver, no
   verificable desde `git log` (solo un autor: `HulbertAlejandro`).

## ⚪ No implementado todavía

- Dream Environment.
- Controlador PPO (Stable-Baselines3).
- Transformer y TSMixer como alternativas al LSTM.
- Evaluación final y comparación de experimentos (fijo vs. RL directo vs. World Model).
- `DOCUMENTACION_PROYECTO.md` (generado antes del bloque LSTM) — desactualizado.

## Qué se estaba haciendo justo antes de este handoff

Se corrió el pipeline completo real por primera vez (`train_autoencoder.py` →
`encode_latent_dataset.py` → `train_world_model.py` → `evaluate_world_model.py` →
`pytest -q`, 22 tests pasando). Los resultados de la evaluación mostraron que el LSTM no
supera al baseline persistente a un paso. Se diagnosticó la causa raíz (desbalance de
escala entre la pérdida latente y la de recompensa) con evidencia numérica de la propia
salida de entrenamiento/evaluación. El fix (normalizar la recompensa) fue propuesto pero
**todavía no se ha escrito ni aplicado en el repositorio**.
