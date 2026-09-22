> Documento completo de la propuesta académica disponible en `docs/PROPUESTA.md`.
> Consúltalo para el detalle exacto de cualquier sección citada aquí (por ejemplo, Sección 15, 20, 30).

# CLAUDE.md — Contexto permanente del proyecto

## Objetivo académico

Trabajo de grado de Ingeniería de Sistemas (Universidad del Quindío). Implementar un
**World Model** (Ha & Schmidhuber, 2018) para control inteligente de semáforos en una
intersección simulada con SUMO, comparando este enfoque contra Reinforcement Learning
directo sobre el simulador.

Pregunta de investigación: ¿puede un modelo aprendido de la dinámica del tráfico reducir
las interacciones necesarias con SUMO, sin perder desempeño de control?

## Papers de referencia

- Ha, D., & Schmidhuber, J. (2018). *World Models*. — arquitectura base (V+M+C).
- Hafner, D., et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*.
  — referencia conceptual para el Dream Environment.
- Alegre, L. N. *SUMO-RL* (github.com/LucasAlegre/sumo-rl) — infraestructura de
  simulación reutilizada, no una técnica de modelado propia del proyecto.

Nota: **no se implementa** la Mixture Density Network (MDN-RNN) del paper original, ni
CMA-ES para el controlador — decisión de diseño explícita, documentada en la Sección 30
de la propuesta (auditoría tecnológica curricular).

## Arquitectura general (estado real, no aspiracional)

```
SUMO → TrafficEnvironment → CustomStateBuilder → TrafficState (vector de 26)
     → dataset (collect → split por episodio → normalize) → TransitionDataset
     → Autoencoder (Encoder/Decoder) → z                                        [COMPLETO, entrenado]
     → encode_latent_dataset.py → *_latent.npz                                   [COMPLETO]
     → LatentSequenceDataset (ventanas por episodio)                             [COMPLETO]
     → LatentDynamicsLSTM (recibe z_t + a_t, predice ẑ_{t+1} y r̂_{t+1})         [ENTRENADO, con un
                                                                                   problema real, ver abajo]
     → evaluation/world_model_evaluation.py (Experimento 1: modelo vs. baseline
       persistente, a 1 y varios pasos, con compounding error)                   [COMPLETO]
     → [PENDIENTE] Dream Environment → [PENDIENTE] PPO → [PENDIENTE] evaluación final
```

Extensiones opcionales previstas, misma prioridad, no implementadas todavía:
Transformer y TSMixer como sustitutos intercambiables del bloque LSTM (interfaz común
definida en `models/world_model/base.py::TemporalModel`).

## Decisiones de diseño ya tomadas

- El estado (`TrafficState`) tiene **26 dimensiones**: 4 carriles × 5 variables
  (vehicle_counts, queue_lengths, waiting_times, mean_speeds, occupancies) + 4 de
  one-hot de fase del semáforo + 2 (elapsed/remaining phase time).
- La fase del semáforo y la acción del controlador se codifican **one-hot**
  (`action_dim = 2`: 0 = mantener fase, 1 = cambiar de fase).
- El "throughput" en la recompensa se mide como vehículos que **llegaron a destino**
  (`traci.simulation.getArrivedNumber()`, delta entre pasos) — nunca vehículos
  presentes en el carril.
- Split de datos **por episodio completo**, nunca por transición suelta.
- Normalización de estados ajustada **solo con el split de entrenamiento**.
- `RepresentationConfig.input_dim` **no tiene valor por defecto** — siempre se obtiene
  del tamaño real del estado.
- `WorldModelConfig.latent_dim` deriva de `RepresentationConfig.latent_dim` — única
  fuente de verdad, con validación de consistencia si se pasan ambos.
- Episodios más cortos que `sequence_length` se **descartan**, nunca se rellenan.
- El `LatentDynamicsLSTM` predice **dos salidas**: `(ẑ_{t+1}, r̂_{t+1})` — la predicción
  de recompensa es parte del núcleo según la Sección 15 de la propuesta.
- Cada checkpoint entrenado (`.pt`) guarda junto a sí un `.json` con los hiperparámetros
  exactos usados.
- Semillas fijas (`torch`, `numpy`, `cuda`) en todo entrenamiento.
- La evaluación del World Model (Experimento 1) compara siempre contra un **baseline
  persistente** ("nada cambia respecto al último valor real observado"), usando las
  **acciones reales** del episodio, no acciones imaginadas — evaluar con acciones
  hipotéticas es responsabilidad del futuro Dream Environment, no de esta etapa.
- **Pendiente de aplicar** (ver PROJECT_STATUS.md): la pérdida de entrenamiento del LSTM
  debe normalizar la recompensa antes de combinarla con la pérdida latente — sin esto,
  el término de recompensa domina el entrenamiento por diferencia de escala.

## Decisiones que NO deben cambiarse sin consultar primero

- No agregar MDN-RNN ni CMA-ES.
- No usar CNN (el estado es un vector, no una imagen).
- No usar embeddings de tokens discretos para el estado.
- No cambiar el criterio de episodios cortos (descartar, no rellenar) sin discutirlo.
- No commitear datasets generados ni pesos de checkpoint (`.npz`, `.pt`) — sí se
  permiten los `.json` de hiperparámetros junto a cada checkpoint (son pequeños y
  documentan qué configuración produjo cada resultado).
- No evaluar el modelo temporal con acciones imaginadas todavía — eso es del Dream
  Environment, una etapa posterior y separada.

## Estructura del repositorio

```
configs/        EnvironmentConfig, RewardConfig, RepresentationConfig, TrainingConfig, WorldModelConfig
datasets/       transition_dataset.py, latent_sequence_dataset.py, metadata.json, raw/, processed/ (generados)
docs/           PROPUESTA.md (propuesta académica completa)
environments/   TrafficEnvironment, CustomStateBuilder, TrafficState, ProjectActionSpace,
                ProjectRewardFunction, contratos Protocol, single-intersection/ (red SUMO propia)
models/
  representation/  Encoder, Decoder, Autoencoder
  world_model/     base.py (Protocol TemporalModel), lstm.py (LatentDynamicsLSTM, con reward_head)
training/       train_autoencoder.py, train_world_model.py
evaluation/     autoencoder_evaluation.py, evaluate_autoencoder.py,
                world_model_evaluation.py, evaluate_world_model.py
scripts/        collect_dataset.py, split_dataset.py, merge_dataset.py, normalize_dataset.py,
                visualize_dataset.py, encode_latent_dataset.py, test_environment.py, test_sumo_rl.py,
                train_controller.py (placeholder), evaluate_world_model.py (script real, ya no placeholder)
tests/          test_traffic_environment.py, test_dataset_pipeline.py, test_autoencoder.py,
                test_world_model.py, test_latent_sequence_dataset.py, test_world_model_evaluation.py
pytest.ini, requirements.txt, .gitignore, README.md, LICENSE
```

No existen (a propósito): `utils/`, `notebooks/`.

## Convenciones de trabajo establecidas en este proyecto

- Auditoría antes de avanzar: cada bloque grande se cierra con revisión línea por línea
  contra el repositorio real (`git clone`, no tarball ni narración) antes de pasar al
  siguiente.
- Commits pequeños, una responsabilidad por commit, verificados con `pytest -v` antes
  de subir.
- Diseño explicado y aprobado antes de escribir código.
- Nunca asumir que "corrió sin error" significa "está bien" — se exige inspeccionar los
  números reales de la salida (curvas de pérdida, métricas de evaluación), no solo que
  el script termine.

## Pendiente de verificar (no confirmado, no inventar)

- Por qué apareció una cuenta/app como colaborador en el repositorio de GitHub que el
  autor del proyecto no reconoce — sin resolver, requiere revisar Settings →
  Collaborators y Settings → Integrations en GitHub directamente.
