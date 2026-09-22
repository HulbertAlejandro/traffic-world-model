# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: Dream Environment implementado (`c2bbd8c`, `16ec05d`), 27/27
tests en verde, confirmado contra el repo real vía `git clone`.

## ✅ Dream Environment — implementado (primera versión)

**Commits**: `c2bbd8c` (refactor: extrae `predict_next_step` de `rollout_episode` en
`evaluation/world_model_evaluation.py`, sin cambio de comportamiento, verificado número
por número contra la corrida anterior) y `16ec05d` (feat: `environments/dream_environment.py`).

**Qué es**: `DreamEnvironment`, una clase compatible con `gymnasium.Env` (`reset`/`step`)
que imagina trayectorias usando el `LatentDynamicsLSTM` ya entrenado, sin tocar SUMO en
ningún momento. Responde las tres preguntas de diseño que quedaban abiertas en
`TODO.md`:

- **Mecanismo de imaginación**: cada episodio imaginado se **siembra** con una ventana
  real de `sequence_length` pasos tomada de un episodio grabado (`*_latent.npz`) — el
  histórico `(z, acción)` de esa ventana es dato real; solo la acción elegida en cada
  `step()` en adelante es hipotética. `step(action)` reemplaza únicamente la última
  acción de la ventana actual y llama a `predict_next_step` (la misma función que usa
  `rollout_episode` en la evaluación), así que la lógica de ventaneo y desnormalización
  de recompensa tiene una única fuente de verdad para evaluación e imaginación.
- **Horizonte de imaginación**: `max_dream_steps=10` por defecto — coincide con el
  rango de horizontes efectivamente validado en el Experimento 1
  (`evaluate_world_model.py` mide hasta horizonte 10); más allá de eso no hay evidencia
  de qué tan confiables son las predicciones del modelo.
- **Interfaz**: se optó por `reset`/`step` al estilo `TrafficEnvironment` (no una
  función simple de "evaluar una secuencia de acciones candidata"), para poder pasar
  `DreamEnvironment` directamente a un controlador tipo PPO de Stable-Baselines3 más
  adelante, sin una capa de adaptación intermedia.

**Verificación**: 5 tests nuevos (`tests/test_dream_environment.py`) cubriendo reset
válido, step válido, truncamiento en `max_dream_steps`, rechazo de episodios más cortos
que `sequence_length`, y rechazo de acción inválida — **27/27 tests en verde** (22
previos + 5 nuevos), sin regresiones.

**Qué NO hace todavía**: no hay ningún controlador (PPO) usándolo aún — eso es el
siguiente paso. `DreamEnvironment` en sí mismo es solo el mecanismo de imaginación.

## ✅ Sobreajuste residual del LSTM — investigado y cerrado (conclusión: no era el cuello de botella real)

**Cambios aplicados** en `training/train_world_model.py`: `weight_decay=1e-4` en el
optimizador Adam, y early stopping con paciencia de 15 épocas sin mejora en
`validation_loss`.

**Resultado**: mejora marginal en la brecha train/validation (~3.0x → ~2.6x; el early
stopping no llegó a activarse, terminó en 14/15 épocas sin mejora al llegar a la época
100). **Sin cambio significativo en las métricas de evaluación sobre test**
(`reward_mse` horizonte 1: 41.3 → 41.9; horizonte 10: 172.3 → 167.0 — diferencias
dentro del ruido de reentrenar).

**Por qué el cambio fue tan pequeño, explicación confirmada**: `evaluate_world_model.py`
ya usaba `world_model_best.pt` (el checkpoint de menor `validation_loss`, no el de la
última época) desde antes de este fix. Las épocas tardías donde `train_loss` seguía
bajando mientras `validation_loss` se estancaba **nunca afectaron el resultado
reportado** — ya estaban descartadas por la selección de mejor checkpoint. El
"sobreajuste" visible en la curva de pérdida era más cosmético que sustantivo.

**Decisión**: se aprueba y mantiene el cambio (deja infraestructura de regularización
reutilizable para cuando se entrenen Transformer/TSMixer, que podrían sobreajustar más
en serio), pero se documenta honestamente que no resolvió un problema grave porque el
problema nunca fue tan grave como parecía en la curva de pérdida. Palancas adicionales
(weight_decay más agresivo, dropout real con num_layers=2, más episodios,
sequence_length distinto) quedan disponibles pero NO se aplican ahora — no hay evidencia
de que el sobreajuste esté limitando ningún resultado actual del proyecto.

## ✅ Todo lo anterior sigue vigente sin cambios

Entorno, pipeline de dataset, fix de fase del semáforo, fix de normalización de
recompensa, Autoencoder, LSTM sobre `z`, Experimento 0 (Autoencoder confirmado, se
mantiene), y el cierre del sobreajuste residual del LSTM — sin cambios desde el
handoff anterior.

## 🟡 Pendiente

1. Cuenta o app desconocida en GitHub — sigue sin resolver.
2. `DOCUMENTACION_PROYECTO.md` sigue desactualizado (no incluye LSTM, bug de fase,
   Experimento 0, cierre del sobreajuste, ni Dream Environment).

## ⚪ No implementado todavía

- Controlador PPO (Stable-Baselines3) entrenado dentro del `DreamEnvironment`.
- Transformer, TSMixer, evaluación final comparativa.

## Qué se estaba haciendo justo antes de este handoff

Se implementó `DreamEnvironment`: primero un refactor sin cambio de comportamiento
(`predict_next_step` extraído de `rollout_episode`, verificado número por número),
luego la clase en sí, compatible con `gymnasium.Env`, sembrada con episodios reales e
imaginando en espacio latente. 27/27 tests en verde, ambos commits verificados en
GitHub. Siguiente paso natural: controlador PPO entrenado dentro de
`DreamEnvironment`.