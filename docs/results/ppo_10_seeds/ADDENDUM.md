# Addendum (escrito y commiteado antes de entrenar): PPO con 10 semillas por controlador

Es una **extensión del resultado central oficial** (Tabla 3 del artículo; PROJECT_STATUS.md,
"Auditoría técnica y correcciones", punto 2), no una exploración descartable.

**Pregunta.** Con 3 semillas del PPO del sueño y 4 de cada RL directo, ¿se sostienen las
conclusiones con 10 semillas por controlador? En particular, ¿se vuelve detectable la diferencia
frente al RL directo de 30k en los escenarios nuevos (pareado p = 0.54)?

## Semillas: fijadas aquí, antes de ver ningún resultado

| Condición | Semillas existentes (se reutilizan sin tocarlas) | Semillas nuevas | Total |
|---|---|---|---|
| PPO del sueño (World Model) | 0, 1, 2 (`controller/`) | **3, 4, 5, 6, 7, 8, 9** (7) | 10 |
| RL directo, 10,000 pasos | 0, 1, 2, 3 (`controller_direct/`) | **4, 5, 6, 7, 8, 9** (6) | 10 |
| RL directo, 30,000 pasos | 0, 1, 2, 3 (`controller_direct_30k/`) | **4, 5, 6, 7, 8, 9** (6) | 10 |

- Son 19 entrenamientos nuevos.
- Las semillas son las siguientes enteras de cada condición; no se elige ninguna después de ver
  resultados.
- Si un entrenamiento falla por una causa técnica (un proceso que muere, el bloqueo de DLLs), se
  repite **la misma semilla**. No se sustituye por otra, y cualquier repetición se documenta.
- **Ninguna semilla se descarta** por su resultado.

## Protocolo de entrenamiento: idéntico al de las semillas oficiales

**RL directo.** `training/train_controller_direct.py` sin modificar, con
`--seed N --total-timesteps {10000|30000} --output-dir <carpeta nueva>`. Es exactamente como se
entrenaron las semillas 1 a 3. El resto viene de `ControllerConfig` y del script:

- `normalize_obs=True` y `normalize_reward=True`, con `reward_clip` 10;
- `learning_rate` 3e-4, `n_steps` 256, `batch_size` 64, `n_epochs` 10, `gamma` 0.99;
- `EvalCallback` cada 1,000 pasos, con 5 episodios en las semillas fijas 20000–20004;
- `ReseedingWrapper.training_seeds()` para el entrenamiento.

**PPO del sueño.** `training/train_controller.py` no acepta semilla ni carpeta: toma la semilla de
`ControllerConfig` y escribe en `controller/`, la carpeta oficial. Para no modificar código oficial
ni arriesgar el checkpoint oficial:

- Un envoltorio, `docs/results/ppo_10_seeds/train_dream_seed.py`, importa el módulo, redirige su
  `CONTROLLER_DIR` a una carpeta nueva y llama a su `main()` con
  `ControllerConfig(seed=N)`.
- Todo lo demás es el código oficial, sin cambios: `DreamEnvironment` con el LSTM oficial
  (`world_model_best.pt`), `dream_max_steps` 7, 50,000 pasos imaginados, `normalize_reward=True`,
  `normalize_obs=False`, y selección en SUMO real cada 5,000 pasos con 5 episodios en las semillas
  20000–20004.
- El envoltorio se niega a escribir en una carpeta que ya tenga archivos.

**Carpetas nuevas** (las oficiales no se tocan):

- `models/checkpoints/controller_10seeds/seed{3..9}/`
- `models/checkpoints/controller_direct_10seeds/seed{4..9}/`
- `models/checkpoints/controller_direct_30k_10seeds/seed{4..9}/`

Los pesos (`.zip`, `.pkl`, `.npz`) no se versionan, como siempre. Los `.json` de hiperparámetros sí.

**Paralelismo.** Hasta 3 entrenamientos a la vez, como en las semillas existentes. No se cambia el
número de hilos de torch: el valor por defecto es el que usaron los oficiales.

## Controles obligatorios antes de entrenar semillas nuevas (si alguno falla, se detiene)

1. **Reproducción del entrenamiento del sueño.** Con el envoltorio se reentrena la **semilla 2** en
   una carpeta temporal. Su `evaluations.npz` debe coincidir exactamente con
   `controller/evaluations.npz`, y los pesos de su `best_model.zip` con los del checkpoint oficial
   (`controller/best_model.zip`, semilla 2). Esto confirma que el envoltorio es el protocolo oficial
   y que el SUMO reinstalado da lo mismo.
2. **Reproducción del entrenamiento directo.** Con el script oficial se reentrena la semilla 3 de
   10k en una carpeta temporal. Debe coincidir exactamente con `controller_direct/evaluations_seed3.npz`
   y con los pesos de `best_model_seed3.zip`.
   - Las de 30k comparten las primeras evaluaciones con las de 10k (documentado), así que este
     control también cubre su arranque.
3. **Reproducción de la evaluación.** En la evaluación final, cada checkpoint existente debe
   reproducir exactamente sus recompensas por episodio publicadas en `docs/results/`
   (`multiseed_*` y `direct_*_seed3_*`).
   - El emparejamiento se hace por las recompensas, no por el nombre del archivo: las etiquetas de
     las semillas del sueño se corrigieron en el commit `150e10b` y las rutas publicadas son las
     anteriores.
   - Tiempo fijo y la regla también deben reproducirse.
4. **Reproducción de los p publicados.** Con los episodios versionados de 3 + 1 semillas, el script de
   análisis debe reproducir los p publicados de sueño vs. 10k y sueño vs. 30k: semilla y pareado, en
   los dos conjuntos de escenarios.

Si el control 1 o el 2 no coinciden exactamente, se detiene antes de entrenar y se reporta.

## Evaluación (en SUMO real, protocolo de siempre)

`scripts/evaluate_multiseed_statistical.py`, sin modificar, con las 10 semillas de cada método,
tiempo fijo y la regla:

- **escenarios oficiales:** `--seed-bases 3000 5000 --episodes-per-base 15` (30 escenarios);
- **escenarios nuevos:** `--seed-bases 7000 --episodes-per-base 30` (7000–7029).

Salidas versionadas: `docs/results/ppo_10_seeds/eval_official_scenarios.{json,csv}` y
`eval_fresh_scenarios_7000.{json,csv}`.

## Métricas (las de la Tabla 3)

- **Recompensa media:** la media de todas las semillas, en los dos conjuntos de escenarios.
- **Episodios catastróficos:** recompensa < −600, sobre semillas × escenarios.
- **Interacciones reales por semilla**, con la misma contabilidad de siempre:
  - RL directo: 13,240 (10k) y 39,208 (30k), por construcción. Se verifica con la longitud de los
    `evaluations.npz` nuevos.
  - World Model: 3,000 de selección por semilla, más el dataset de 4,800 transiciones:
    - amortizado entre las semillas que lo comparten. Con 10 semillas son 480 por semilla, y el
      total queda en 3,480;
    - sin amortizar, 7,800.
  - Se reportan las dos cifras y la razón como rango. **Advertencia fijada aquí:** la cifra
    amortizada mejora solo por entrenar más semillas. La comparación conservadora es la de 7,800.

## Pruebas estadísticas (fijadas)

Para cada comparación, **sueño vs. directo 10k** y **sueño vs. directo 30k**, en cada conjunto de
escenarios (oficiales y nuevos):

- **Las mismas pruebas que produjeron los p publicados**, con las mismas funciones del script:
  - Welch sobre las medias por semilla (n = 10 vs. 10);
  - t pareada por escenario, sobre la media de las semillas de cada método en cada escenario.
- **Además, Wilcoxon de rangos con signo pareado por escenario** (bilateral), sobre las mismas
  diferencias por escenario.
  - El repositorio no lo usaba: los p publicados del pareado son de una t. Se agrega porque el texto
    del artículo cita Wilcoxon.
  - Exacto por programación dinámica si no hay empates ni ceros. Con empates o ceros, aproximación
    normal con corrección por empates, y se dice.
  - Se valida contra enumeración por fuerza bruta en n pequeños.
  - scipy no es dependencia del proyecto.
  - También se calcula con las 3 + 1 semillas publicadas, para el antes y después.
- **Antes y después:** p con 3 / 4 semillas (los publicados, reproducidos en el control 4) frente a
  p con 10 / 10.
- **Una conclusión "cambia"** si, en alguna comparación y conjunto, la t pareada o el Wilcoxon
  cruzan α = 0.05 respecto de lo publicado.
- No se corrige por comparaciones múltiples, igual que en lo publicado. Se reporta como está.
- Comparaciones con tiempo fijo: descriptivas, para completar la tabla.

## Qué NO cambia

- Los checkpoints oficiales siguen siendo los mismos: sueño semilla 2, directo 10k semilla 0 y
  directo 30k semilla 1. **No se reetiquetan** aunque una semilla nueva sea mejor, porque el
  resultado se reporta como media de todas las semillas.
- Ningún archivo existente de `models/checkpoints/` ni de `docs/results/` se modifica.
- Se verifica el md5 de todo `models/checkpoints/`, `datasets/`, `results/` y `docs/results/` (fuera
  de lo exploratorio) antes y después; solo se agregan archivos nuevos.
- No se toca ningún documento de main (CLAUDE.md, PROJECT_STATUS.md, artículo). Todo vive en la rama
  `extension/ppo-10-seeds-and-tables`.
