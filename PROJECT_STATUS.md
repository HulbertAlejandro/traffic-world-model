# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: dataset regenerado desde cero (40 episodios) tras corregir un bug
crítico de lectura de fase del semáforo, con el pipeline completo re-entrenado y
evaluado. Todo validado con ejecución real.

## 🔴 → ✅ Bug crítico encontrado y RESUELTO: la fase del semáforo nunca se leía correctamente

**Qué pasaba:** `CustomStateBuilder` leía la fase actual y el tiempo de fase con
`traci.trafficlight.getPhase()` / `getNextSwitch()` / `getPhaseDuration()`. `sumo_rl`
controla el semáforo escribiendo el estado rojo/ámbar/verde directamente como cadena de
texto (`TrafficSignal.set_next_phase` → `setRedYellowGreenState`), **sin usar el
programa de fases nativo del `.net.xml`**. Como consecuencia, esas llamadas de TraCI
nunca reflejaban los cambios reales — quedaban congeladas en el estado inicial
(confirmado empíricamente: 1200 pasos de recolección real, 100% en la misma fase, pese
a que el semáforo sí cambiaba físicamente, verificado comparando `ts.green_phase` y
`traci.getRedYellowGreenState()`, que sí cambiaban en paralelo).

**Impacto real:** 6 de las 26 dimensiones del estado (4 de one-hot de fase + 2 de
tiempo de fase) eran ruido o constantes sin sentido en TODO el dataset recolectado
hasta ese momento. El Autoencoder y el LSTM entrenados antes de este fix quedaron
invalidados y fueron descartados junto con el dataset.

**Fix aplicado:** `CustomStateBuilder.build_state()` ahora recibe `sumo_rl_env` (la
instancia de `sumo_rl.SumoEnvironment`) y lee la fase real desde
`TrafficSignal.green_phase` y el tiempo transcurrido desde
`TrafficSignal.time_since_last_phase_change` — los atributos que `sumo_rl` sí
mantiene actualizados al aplicar acciones. `remaining_phase_time` se redefinió como
"tiempo que falta para que el cambio de fase sea legal" (`max(0, min_green - elapsed)`),
ya que bajo control por RL no existe un "próximo cambio programado" real.
`TrafficEnvironment.reset()`/`step()` actualizados para pasar `sumo_rl_env=self._env`.

**Verificado con evidencia real:** el one-hot de fase ahora sí se mueve entre pasos
(`[1,0,0,0]` → `[0,1,0,0]`, confirmado con ejecución directa). 22/22 tests siguen
pasando (el fallback sin `sumo_rl_env` protegió la compatibilidad hacia atrás).

## ✅ Dataset regenerado y pipeline completo re-entrenado (40 episodios, no 20)

Se subió `DEFAULT_NUM_EPISODES` de 20 a 40 en `scripts/collect_dataset.py`
aprovechando que había que regenerar todo de cualquier forma. Resultado del split:
train ≈ 28 episodios, validation ≈ 6, test ≈ 6 (antes: train ≈ 14, validation ≈ 3,
test ≈ 3) — mucho mejor base estadística.

**Resultados del re-entrenamiento (con datos limpios):**

- **Autoencoder**: train=0.0191, validation=0.0200 (época 100) — brecha train/val casi
  nula (antes, con el bug: 0.0298 vs 0.0827, brecha ~2.8x). Mejora consistente con que
  ahora hay señal real que aprender en las columnas de fase.
- **LSTM**: train=0.077, validation=0.231 (época 100) — brecha ~3x (antes: ~49x con 20
  episodios y datos rotos). Sobreajuste residual todavía presente: `val_loss` deja de
  mejorar consistentemente después de la época ~70, mientras `train_loss` sigue
  bajando. No bloqueante, pero pendiente de atender (ver sección Pendiente).
- **Evaluación (`evaluate_world_model.py`), test con 6 episodios, n=210 por horizonte**:
  el World Model supera al baseline persistente en los 10 horizontes evaluados, con
  márgenes mucho más amplios que antes del fix:
  - Horizonte 1: latente 0.193 (modelo) vs. 1.002 (baseline) → ~5.2x mejor.
  - Horizonte 1: recompensa 41.3 (modelo) vs. 716.6 (baseline) → ~17.4x mejor.
  - Horizonte 10: el modelo sigue siendo ~3.8x mejor en latente y ~7.6x mejor en
    recompensa que el baseline.
  - Compounding error: el error latente del modelo crece 2.61x del horizonte 1 al 10.

Este resultado es cualitativamente distinto al anterior (que también "ganaba" al
baseline, pero por márgenes ajustados y sobre datos con una variable crítica rota). Este
es el primer resultado del bloque LSTM que se considera confiable para reportar.

## 🟡 Pendiente

1. Sobreajuste residual del LSTM (brecha train/val ~3x, val_loss estancado desde la
   época ~70) — candidato natural: early stopping, o simplemente más episodios si el
   tiempo lo permite. No bloqueante para avanzar.
2. Cuenta o app desconocida en GitHub — sigue sin resolver.
3. `DOCUMENTACION_PROYECTO.md` sigue reflejando el estado anterior a todo este bloque —
   desactualizado.

## ⚪ No implementado todavía

- Experimento 0 (Autoencoder vs. estado crudo) — ahora que el dataset es confiable, es
  el siguiente paso natural según el orden de la propuesta.
- Dream Environment, controlador PPO, Transformer, TSMixer, evaluación final
  comparativa.

## Qué se estaba haciendo justo antes de este handoff

Se diagnosticó y corrigió un bug crítico (lectura de fase del semáforo desconectada de
la realidad simulada), se regeneró el dataset completo con más episodios (40, antes 20),
y se re-entrenó y evaluó el pipeline completo. El resultado confirma que el World Model
supera al baseline persistente con márgenes amplios y consistentes en todos los
horizontes evaluados — el primer resultado del proyecto considerado confiable para
avanzar o reportar.