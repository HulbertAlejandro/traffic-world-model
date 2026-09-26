# Resultados versionados de evaluación en SUMO real

Cada par `.json` / `.csv` es la salida completa de `scripts/evaluate_multiseed_statistical.py`: los
argumentos con los que se ejecutó, un resumen por política, las comparaciones estadísticas y **todos
los episodios** (política, semilla de entrenamiento, semilla de escenario, reward, espera y cola
medias, cambios de fase reales). Los `.csv` tienen una fila por episodio, para auditar cualquier
cifra publicada sin volver a ejecutar SUMO.

| Archivo | Políticas | Escenarios |
|---|---|---|
| `multiseed_official_scenarios` | PPO del sueño (3 semillas), RL directo 10k y 30k con el fix de C1 (semillas 0–2), tiempo fijo, regla "fase contraria" | Oficiales: `seed_base` 3000 y 5000, 15 episodios cada una |
| `multiseed_fresh_scenarios_7000` | Las mismas | Nuevos: 7000–7029, nunca usados durante el desarrollo |
| `direct_10k_fix_seed3_official_scenarios`, `direct_10k_fix_seed3_fresh_scenarios_7000` | RL directo 10k con el fix, semilla 3 (+ tiempo fijo como control) | Oficiales / nuevos |
| `direct_30k_fix_seed3_official_scenarios`, `direct_30k_fix_seed3_fresh_scenarios_7000` | RL directo 30k con el fix, semilla 3 (+ tiempo fijo como control) | Oficiales / nuevos |

## Rutas de checkpoints: al evaluar frente a hoy

Los archivos registran la ruta que tenía cada checkpoint **en el momento de evaluarse**. No se
reescriben: son el registro de lo que se ejecutó. Después se reorganizaron los checkpoints (commits
`150e10b` y `fa7bc51`), con md5 verificado de cada archivo movido; esta es la correspondencia:

| Ruta registrada en los resultados | Ruta actual | Qué es |
|---|---|---|
| `controller/best_model_seed0_worse.zip` | `controller/best_model_seed0_worse.zip` (sin cambio) | PPO del sueño, semilla 0 |
| `controller/best_model.zip` | `controller/best_model_seed1.zip` | PPO del sueño, semilla 1 (oficial hasta `150e10b`) |
| `controller/best_model_seed2.zip` | `controller/best_model.zip` | PPO del sueño, semilla 2 (**oficial** desde `150e10b`) |
| `controller_direct_fixed/seed0/best_model.zip` | `controller_direct/best_model.zip` | RL directo 10k, semilla 0 (**oficial**) |
| `controller_direct_fixed/seedN/best_model.zip` (N = 1, 2, 3) | `controller_direct/best_model_seedN.zip` | RL directo 10k, semillas 1–3 |
| `controller_direct_30k_fixed/seed1/best_model.zip` | `controller_direct_30k/best_model.zip` | RL directo 30k, semilla 1 (**oficial** de la verificación de 30k) |
| `controller_direct_30k_fixed/seedN/best_model.zip` (N = 0, 2, 3) | `controller_direct_30k/best_model_seedN.zip` | RL directo 30k, semillas 0, 2 y 3 |

Todas las rutas son relativas a `models/checkpoints/`. Las estadísticas de `VecNormalize` de cada
checkpoint siguen el mismo cambio de nombre (`<nombre>_vecnormalize.pkl`), igual que su `.json` de
hiperparámetros, sus `evaluations*.npz` y su modelo final `ppo_controller_direct_final*`.

Los checkpoints del RL directo entrenados **antes** del fix de C1 están archivados, con sus nombres
originales, en `controller_direct_prefix_bug/` y `controller_direct_30k_prefix_bug/`.

## Criterio del checkpoint "oficial"

En cada carpeta, el oficial es la semilla con mejor media en los 30 escenarios oficiales. Es una
etiqueta **descriptiva**: solo decide qué checkpoint cargan los scripts que evalúan uno solo
(`evaluate_final_comparison.py`, `ver_controlador.py`). Todo resultado de un método se reporta como la
media de **todas** sus semillas; las demás semillas quedan archivadas como evidencia de la
variabilidad entre entrenamientos, no como candidatas descartadas.
