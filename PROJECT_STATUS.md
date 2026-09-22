# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: Experimento 0 completo (LSTM sobre `z` del Autoencoder vs. sobre
estado crudo normalizado), con 4 commits aplicados y subidos, `pytest` en 22/22 en
cada paso.

## ✅ Experimento 0 — completado con decisión clara: el Autoencoder SE MANTIENE

**Diseño**: mismo `LatentDynamicsLSTM`, misma semilla, mismos hiperparámetros de
entrenamiento (100 épocas, `hidden_dim=128`, `batch_size=32`, `lr=1e-3`), mismo split
de test (6 episodios) — la única diferencia entre los dos experimentos es la
representación de entrada: `z` de 8 dimensiones (Autoencoder) vs. estado crudo
normalizado de 26 dimensiones.

**Criterio de decisión, acordado antes de implementar**: `reward_mse` (desnormalizado
a unidades reales) es la única métrica directamente comparable entre ambos
experimentos, porque la recompensa es la misma magnitud física sin importar qué
representación alimenta al LSTM. El error de estado/latente (`state_mse`/`latent_mse`)
vive en espacios distintos (8 vs. 26 dims, escalas distintas) y se reporta solo como
contexto, nunca como el número decisivo.

**Resultado — el Autoencoder gana en los 10/10 horizontes evaluados en reward_mse**:
```
h=1:  41.3 (z) vs. 59.4 (crudo)   → z ~1.44x mejor
h=10: 172.3 (z) vs. 335.6 (crudo) → z ~1.95x mejor
```
La ventaja del Autoencoder **crece** con el horizonte — el estado crudo acumula error
autorregresivo más rápido que el `z` comprimido, con la misma capacidad de red. Nota
metodológica importante: `state_mse` crudo salió numéricamente *menor* que `latent_mse`
de `z` (0.153 vs. 0.193) — si esa hubiera sido la métrica de decisión, la conclusión
habría sido la contraria. Confirma que decidir de antemano usar `reward_mse` (no
`state_mse`) como criterio fue la elección metodológicamente correcta.

**Decisión oficial del proyecto**: el Autoencoder se mantiene en el sistema final,
según el criterio ya establecido en la Sección 24 de la propuesta ("El Autoencoder/VAE
debe mantenerse solo si demuestra utilidad para la representación").

**Archivos nuevos de este bloque**: `scripts/prepare_raw_sequence_dataset.py`,
`training/train_world_model_raw.py`, `scripts/evaluate_world_model_raw.py`,
`scripts/compare_experiment_0.py`. Checkpoints del experimento crudo aislados en
`models/checkpoints/raw_state/` (evita colisión de nombre con `reward_scaler.json` del
experimento `z`). `training/train_world_model.py` y todo el pipeline del Autoencoder/`z`
quedaron intactos, sin tocar.

## ✅ Todo lo anterior sigue vigente sin cambios

Entorno, pipeline de dataset, fix de fase del semáforo, fix de normalización de
recompensa, Autoencoder y LSTM sobre `z` — sin cambios desde el handoff anterior.

## 🟡 Pendiente

1. Sobreajuste residual del LSTM (mencionado en el handoff anterior) — sigue sin
   atenderse, no bloqueante.
2. Cuenta o app desconocida en GitHub — sigue sin resolver.
3. `DOCUMENTACION_PROYECTO.md` sigue desactualizado (no incluye LSTM, bug de fase, ni
   Experimento 0).

## ⚪ No implementado todavía

- Dream Environment, controlador PPO, Transformer, TSMixer, evaluación final
  comparativa.

## Qué se estaba haciendo justo antes de este handoff

Se ejecutó el Experimento 0 completo con Claude Code en modo automático (4 commits,
verificados con `pytest` después de cada uno, push autorizado tras revisar el resultado
final). Conclusión oficial: el Autoencoder se mantiene en el sistema, con evidencia
consistente en los 10 horizontes evaluados usando `reward_mse` como criterio decisivo.
Siguiente paso natural según la propuesta: Dream Environment.