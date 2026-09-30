# Addendum (escrito y commiteado antes de simular): test ampliado a 48 episodios (curiosidad del autor)

Pregunta: si el test tuviera más episodios, ¿el resultado se estabiliza como predice la estadística
(el ancho del IC por episodios cae con √n), o sigue igual de sensible que con 12 (sección 15)?

- **Solo se generan episodios nuevos de test.** No se tocan train ni validación, y no se reentrena
  ningún modelo.
- Es más ligero que los addenda de entrenamiento, porque no hay hiperparámetros que elegir. Todo lo
  que se mide queda fijado aquí.

## Protocolo de recolección: el oficial, verificado en el código

| Pieza | Oficial | Aquí |
|---|---|---|
| Función | `scripts/collect_dataset.py::collect_dataset` | la misma, **importada** (no copiada) |
| Entorno | `TrafficEnvironment()` con la configuración por defecto: red `single-intersection`, demanda asimétrica de `single-intersection.rou.xml`, 300 s simulados | igual |
| Pasos por episodio | 60 | 60 |
| Semilla de SUMO | la del episodio i es `seed_start + i`; los 80 oficiales usan las **0–79**, con `episode_id` = semilla | `seed_start = 9000`, **semillas 9000–9035** |
| Acciones | `ProjectActionSpace.sample()`, uniforme en {0, 1} con `np.random` global **sin fijar** | igual, pero con `np.random.seed(9000)` una vez antes de recolectar, para que se pueda reproducir. La distribución no cambia |
| Normalización | `datasets/processed/scaler.pkl`, ajustado solo con train | el **mismo** `scaler.pkl`, sin reajustar |

- **Semillas nuevas.** 9000–9035 no aparecen en ninguna parte del proyecto. Las usadas son el dataset
  (0–79), las evaluaciones de controladores (3000–3014, 5000–5014 y 7000–7029) y `ReseedingWrapper`
  (20000–20004).
- `collect_dataset` numera los episodios desde 0. Después de recolectar se reescribe
  `episode_id` = semilla de SUMO (9000 + i), igual que en el oficial, para que no colisione con los
  ids originales.

## Cuántos episodios: 36 nuevos (12 + 36 = 48)

- Con 48 episodios, la predicción de √n es un ancho de IC por episodios de √(12/48) = 0.5 veces el de
  12, fácil de contrastar.
- Los 36 nuevos solos triplican el test original. Forman una **réplica independiente**: permiten ver
  si las estimaciones hechas con los 12 originales caen donde su bootstrap decía (la prueba más
  directa de la sección 15).
- El costo es moderado: 36 × 60 pasos de SUMO más ~143 evaluaciones sobre 36 episodios. Ir mucho más
  allá no cambiaría la respuesta cualitativa.

## Controles antes de usar los episodios nuevos (obligatorios; si alguno falla, se detiene)

1. **El simulador es el mismo.** Se reproducen los episodios originales de test 8 y 53: reset con su
   semilla y las mismas acciones registradas en `test_raw.npz`. Estados, recompensas y `next_states`
   deben coincidir **exactamente** con los guardados. Si no coinciden, SUMO o sumo-rl ya no generan
   lo mismo que generó el dataset, y los episodios nuevos no serían comparables.
2. **La normalización es la misma.** Aplicar `scaler.pkl` a `test_raw.npz` reproduce exactamente
   `test.npz`.
3. **Las codificaciones son las mismas.** Codificar el test original con el Autoencoder oficial y con
   cada uno de los 10 de la exploración (`ae12*` y `ae16*`, sobre 24 dims) reproduce exactamente los
   `test_latent.npz` ya guardados. La codificación es la de `scripts/encode_latent_dataset.py` y
   `sweep.py train-ae`: `load_autoencoder` y `encode`.
4. **Formato.** 36 episodios × 60 pasos, sin NaN ni Inf. Las columnas 22 y 23 siempre en 0 (supuesto
   del estado de 24 dims).

## Archivos (carpeta exploratoria)

- Todo en `docs/results/exploratory/latent_sweep/test_episodes_expanded/`:
  - episodios crudos por semilla;
  - `test_new_raw.npz`, `test_new.npz` (normalizado, 26 dims) y `test_new_latent.npz` (Autoencoder
    oficial);
  - `test_new_raw_seq.npz` (26 dims);
  - las versiones de 24 dims (`test_new24_raw_seq.npz`);
  - `ae{12,16}*/test_new_latent.npz`.
- **Los `.npz` no se versionan** (regla del proyecto): un `.gitignore` local los excluye. Se versiona
  un `MANIFEST.json` con las semillas, los conteos y los md5 de cada `.npz`.
- `datasets/processed/` no se toca.

## Qué se mide

- **Modelos:** los mismos 143 checkpoints de la sección 15:
  - los 3 oficiales (LSTM, Transformer y TSMixer);
  - el Experimento 0 oficial (5 z y 5 crudos);
  - la grilla de `latent_dim` = 12 (25 z);
  - las grillas de `latent_dim` = 16 (3 × 25 z y 3 × 10 crudos).
- **Evaluación:** la oficial importada, igual que en la sección 15. Se calcula E[e, h] para cada
  episodio nuevo. Los de los 12 originales se **reutilizan** de `episode_errors.json`.
- **Tres conjuntos de test:** (a) los 12 originales; (b) los 36 nuevos; (c) los 48. Con los mismos
  estadísticos que en la sección 15:
  - niveles de los 3 checkpoints oficiales;
  - E2: Experimento 0, z − crudo;
  - E3: `latent_dim` = 12, z − crudo;
  - E4: `latent_dim` = 16, z − crudo, por arquitectura;
  - E4': los tres pares entre arquitecturas, también con Bonferroni al 98.33%.

  Para cada uno, el IC del 95% solo por semillas, solo por episodios y conjunto, con B = 10,000 y
  `default_rng(12345)`. El bootstrap de episodios sortea n de n (12, 36 o 48), emparejado entre
  corridas.

## Preguntas y criterios (fijados)

- **P1. ¿Se reduce el ancho a la mitad?** Para cada estadístico, la razón entre el ancho del IC solo
  por episodios con 48 y con 12. Se reportan la razón de cada uno y la mediana.

| Mediana de la razón de anchos | Lectura |
|---|---|
| 0.40 a 0.60 | **se cumple aproximadamente** |
| > 0.60 | **se reduce menos de lo esperado**, típico de colas pesadas (episodios raros con mucho peso) |
| < 0.40 | **se reduce más** |

- **P2. ¿Acertaba el bootstrap de 12?** Se cuenta en cuántos estadísticos la estimación con los 36
  nuevos cae dentro del IC solo por episodios calculado con los 12 originales. Si el bootstrap estuviera
  bien calibrado se esperaría ~95%. Como los 36 también tienen ruido, el umbral debería ser algo menor.
  - Si caen dentro menos de la mitad, el IC de 12 **subestimaba** la incertidumbre por episodios.
- **P3. ¿Cambia alguna conclusión?** Por contraste, se compara si el IC conjunto excluye o incluye 0
  con 12 y con 48. Para los pares entre arquitecturas se usa también Bonferroni al 98.33%.
  - Un cambio **no** reescribe nada de lo anterior: se reporta.
  - Con 48, la conclusión de referencia es la del IC conjunto sobre 48.
- **P4. ¿Siguen dominando ep53, ep47, ep12 y ep77?** (descriptivo)
  - la fracción del error en h = 10 de la LSTM oficial que concentran esos 4 en el conjunto de 48;
  - su rango entre los 48 episodios;
  - cuántos episodios nuevos superan al menor de los cuatro;
  - el jackknife sobre 48 para Δ de la LSTM z16 − crudo y para LSTM − Transformer.
- **Descriptivo de comparabilidad:** media, desviación estándar y mínimo de la recompensa por
  episodio en los 12 originales, los 36 nuevos y el train oficial. Sirve para ver si los nuevos
  vienen de la misma distribución.

## Reglas

- No se simula nada antes de commitear este addendum.
- Ningún archivo oficial cambia. Se verifican los md5 y el inventario de `.pt`/`.zip`/`.npz`/`.pkl`
  de `models/checkpoints/` y `datasets/` antes y después.
- Script: `test_expanded.py` en `latent_sweep/`. Salidas: `test_expanded.out` y
  `test_expanded.json`, y `episode_errors_new.json` con E[e, h] de los 36 nuevos para los 143
  checkpoints.
