# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: controlador PPO implementado y entrenado dentro del
`DreamEnvironment` (`86228ae`), 33/33 tests en verde, confirmado contra el repo real.
**Resultado NO validado todavía contra SUMO real** — ver la sección de abajo antes de
asumir que el controlador "funciona".

## ✅ Controlador PPO — implementado y entrenado dentro del Dream Environment

**Commit**: `86228ae` (`configs/controller.py`, `training/train_controller.py`,
`scripts/evaluate_controller.py`, `tests/test_controller.py`; 4 tests nuevos, 33/33 en
verde). Los checkpoints generados (`models/checkpoints/controller/*.zip`,
`evaluations.npz`) NO se commitearon, mismo criterio que los demás pesos entrenados del
proyecto.

**Qué es**: un `PPO` de Stable-Baselines3 (`MlpPolicy`) entrenado enteramente dentro de
`DreamEnvironment` — nunca toca SUMO durante el entrenamiento. `ControllerConfig`
define los hiperparámetros (punto de partida razonable, no afinado empíricamente:
`total_timesteps=50_000`, `learning_rate=3e-4`, `n_steps=256`, `gamma=0.99`), con
`EvalCallback` evaluando sobre `validation_latent.npz` cada ~1000 timesteps y guardando
el mejor checkpoint por recompensa de validación.

**Entrenamiento real ejecutado**: 50,176 timesteps, 196 iteraciones. Curva de
evaluación **ruidosa y no monótona** — oscila entre ~-46 y ~-77 durante todo el
entrenamiento, sin convergencia limpia (mejor punto observado: -46.55 en el timestep
30,000; valor final en el timestep 50,000: -62.90). No hay una mejora clara y estable
como función del número de pasos de entrenamiento.

**Evaluación en el split de test** (`scripts/evaluate_controller.py`, nunca visto por
el PPO ni por el LSTM; 30 episodios × hasta 7 pasos = 210 pasos por política):

```
Politica                  |    mean |    std |      min |      max | streak>=5 | clipped
PPO (entrenado)           |  -53.53 |  31.65 |  -147.04 |  -17.89  |  0/210 (0.0%)  | 28/210 (13.3%)
Accion aleatoria          | -144.80 | 100.49 |  -545.40 |  -32.01  |  5/210 (2.4%)  | 16/210 (7.6%)
Siempre mantener (0)      | -521.80 | 239.24 | -1124.76 | -141.49  | 90/210 (42.9%) | 25/210 (11.9%)
Siempre cambiar (1)       | -493.38 | 197.05 |  -969.11 | -173.03  | 90/210 (42.9%) | 32/210 (15.2%)
Alternando cada paso      | -129.59 |  59.60 |  -303.31 |  -43.02  |  0/210 (0.0%)  | 10/210 (4.8%)
```

PPO supera a las 4 políticas de referencia en `mean` (-53.53, la menos negativa) y
nunca entra en racha de acción ≥5.

**⚠️ Salvedad importante, NO resuelta — dicha sin suavizar**: la tasa de
`reward_clipped` de PPO (13.3%) es **más alta** que la de la política "alternando"
(4.8%), a pesar de que PPO evita rachas largas por completo. Es un posible síntoma de
que la política está explotando el recorte de recompensa para volver artificialmente
barato el error de extrapolación del modelo, en vez de aprender control de tráfico
genuino — no se puede distinguir con la evidencia disponible cuál de las dos
explicaciones es la correcta. **Este resultado NO debe interpretarse como desempeño
validado** hasta contrastarlo contra SUMO real (`TrafficEnvironment`) — paso siguiente
necesario antes de cualquier afirmación sobre la calidad de este controlador.

## ✅ Dream Environment — mitigaciones de extrapolación fuera de distribución (OOD)

**Contexto**: antes de construir el controlador PPO, se hizo una verificación manual
del `DreamEnvironment` con el checkpoint real entrenado (no pesos aleatorios), corriendo
4 políticas simples (aleatoria, siempre mantener, siempre cambiar, alternando) durante
20 episodios imaginados cada una y sumando la recompensa total por episodio.

**Hallazgo**: con `max_dream_steps=10` (el valor original), las políticas de acción
constante ("siempre mantener"/"siempre cambiar") produjeron recompensas imaginadas
totales ~4.7x-5.6x más negativas que el orden de magnitud esperado (`10 × reward_mean
por paso`), mientras que las políticas aleatoria y alternada caían justo en el rango
esperado. Diagnóstico confirmado con los datos reales: en `train_latent.npz`, de 881
rachas de acción idéntica consecutiva observadas en 28 episodios, solo **1** llega a
longitud 10 o más (una racha de 12). El LSTM prácticamente nunca vio secuencias de 10
acciones repetidas durante el entrenamiento — esas dos políticas alimentan al modelo
con entradas fuera de distribución (OOD), y el salto de magnitud es extrapolación
inestable, no una señal físicamente plausible de la dinámica del tráfico.

**Se descartó explícitamente** "corregir" la acción antes de alimentar al LSTM
replicando el filtrado de `min_green` de SUMO: el LSTM se entrenó con la acción
*solicitada* cruda (`ProjectActionSpace.sample()`), nunca con la acción realmente
aplicada tras el filtro de `min_green` — alimentarlo con la versión filtrada
introduciría un patrón distinto, igualmente no visto en entrenamiento, sin resolver
el problema real.

**Dos mitigaciones aplicadas, ambas basadas en evidencia de los datos reales, ninguna
inventada**:

1. `max_dream_steps` bajado de 10 a 7 — el histograma real de rachas
   (`{5: 27, 6: 11, 7: 5, 8: 2, 9: 3, 12: 1}`) muestra que 5-7 pasos está bien
   representado en los datos; 8+ es raro. Mejora el problema (constante-acción bajó de
   ~4.7x-5.6x a ~3.7x-3.9x del orden de magnitud esperado) pero **no lo resuelve**.
2. Recorte (`torch.clamp`) de la recompensa imaginada al rango
   `[REWARD_CLIP_MIN=-165.05, REWARD_CLIP_MAX=1.00]` — percentiles 1 y 99 de las
   recompensas reales en `train_latent.npz` (min real=-324.10, max real=2.90;
   percentiles preferidos sobre el min/max crudo para ignorar outliers extremos raros).
   Expuesto en `info["reward_clipped"]` para monitorear, una vez entrenado el PPO, qué
   tan seguido se activa.

**Verificación final, con ambas mitigaciones activas** (20 episodios × hasta 7 pasos
= 140 pasos por política):

```
Acción aleatoria     | mean= -180.75 | reward_clipped:  12/140 ( 8.6%)
Siempre mantener (0) | mean= -601.31 | reward_clipped:  23/140 (16.4%)
Siempre cambiar (1)  | mean= -533.04 | reward_clipped:  29/140 (20.7%)
Alternando cada paso | mean= -158.09 | reward_clipped:   4/140 ( 2.9%)
```

El recorte se activa 5-7 veces más seguido en las políticas de acción constante que en
aleatoria/alternada — confirma que el diagnóstico está bien dirigido. **Pero, dicho sin
suavizar**: las políticas de acción constante siguen ~3.4x-3.8x por encima del orden de
magnitud esperado (`7 × reward_mean ≈ -156.52`) incluso con el recorte activo — el
recorte acota la consecuencia de un solo paso malo, pero no corrige que el modelo
sistemáticamente predice peor (no solo en outliers) cuando la acción no cambia. Esto
queda documentado como limitación conocida, no como problema resuelto; se decidió
aprobar y avanzar con el controlador PPO de todas formas, monitoreando
`info["reward_clipped"]` y `info["consecutive_action_streak"]` una vez el PPO esté
entrenando.

**Tests**: `test_action_streak_tracked_in_info` (rastrea rachas de acción vía
`info["consecutive_action_streak"]`) y
`test_imagined_reward_is_clipped_to_empirical_range` (recorte, verificado
determinísticamente con `monkeypatch` sobre un rango estrecho) — **29/29 tests en
verde** (27 previos + 2 nuevos).

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
- **Horizonte de imaginación**: `max_dream_steps=10` por defecto en esta versión
  inicial — coincide con el rango de horizontes efectivamente validado en el
  Experimento 1 (`evaluate_world_model.py` mide hasta horizonte 10); más allá de eso
  no hay evidencia de qué tan confiables son las predicciones del modelo. **Revisado
  después**: ver la sección "Dream Environment — mitigaciones de extrapolación fuera
  de distribución (OOD)" más arriba — bajado a 7 tras un sanity check manual.
- **Interfaz**: se optó por `reset`/`step` al estilo `TrafficEnvironment` (no una
  función simple de "evaluar una secuencia de acciones candidata"), para poder pasar
  `DreamEnvironment` directamente a un controlador tipo PPO de Stable-Baselines3 más
  adelante, sin una capa de adaptación intermedia.

**Verificación**: 5 tests nuevos (`tests/test_dream_environment.py`) cubriendo reset
válido, step válido, truncamiento en `max_dream_steps`, rechazo de episodios más cortos
que `sequence_length`, y rechazo de acción inválida — **27/27 tests en verde** (22
previos + 5 nuevos), sin regresiones.

**Qué NO hacía todavía en este punto**: no había ningún controlador (PPO) usándolo aún.
**Actualización**: el controlador PPO ya se implementó y entrenó — ver la sección
"Controlador PPO" al inicio de este documento.

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

- Evaluación del controlador PPO contra SUMO real (`TrafficEnvironment`) — necesaria
  antes de validar la salvedad del `reward_clipped` de arriba.
- Transformer, TSMixer, evaluación final comparativa (tiempo fijo vs. RL directo vs.
  World Model).

## Qué se estaba haciendo justo antes de este handoff

Se implementó y entrenó el controlador PPO dentro del `DreamEnvironment` (33/33 tests
en verde, commit `86228ae` subido a GitHub). El entrenamiento corrió sin errores y el
PPO superó a las 4 políticas de referencia en la autoevaluación dentro del propio Dream
Environment, pero quedó una salvedad seria sin resolver: su tasa de `reward_clipped`
es más alta que la de la política "alternando" pese a evitar rachas largas, lo cual
podría indicar que está explotando el recorte de recompensa en vez de aprender control
real. Documentado explícitamente como resultado no validado. Siguiente paso natural:
evaluar este controlador contra SUMO real antes de sacar cualquier conclusión sobre su
calidad.