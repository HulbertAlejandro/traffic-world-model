# Exploración: ¿ayuda el Autoencoder a predecir la recompensa? (`latent_dim` y semillas)

**Registro de investigación exploratoria. No cambia el pipeline oficial ni ningún valor oficial:**
`latent_dim` sigue siendo 16, el estado sigue teniendo 26 dimensiones, y el Experimento 0 oficial
(`docs/results/experiment_0_multiseed_300ep.json`, PROJECT_STATUS.md, "Auditoría técnica y
correcciones", punto 5) queda como está. Todo el detalle vive en la rama
`exploratory/latent-dim-sweep`, en `docs/results/exploratory/`.

Se usó siempre la misma métrica que el Experimento 0: `reward_mse` en el split de test (12
episodios), en rollouts autorregresivos con las acciones reales, h = 1..10. La **reducción** es
(crudo − z)/crudo; positiva significa que el Autoencoder ayuda. La **reducción por semilla** es la
mediana de esa reducción sobre los 10 horizontes.

## 1. La pregunta y por qué se reabrió

El Experimento 0 pregunta si el modelo temporal predice mejor la recompensa sobre el latente z del
Autoencoder que sobre el estado crudo. Su versión final (5 semillas de LSTM por rama, convergencia
igualada) concluyó que el Autoencoder ayuda "de forma moderada y mayoritaria": 37/50 pares semilla ×
horizonte, +10.9%, con reducciones por semilla de -3.4%, +7.4%, +10.0%, +29.4% y +26.5%.

Que la semilla 0 fuera negativa llevó a revisar si había una asimetría entre las ramas. Luego, si el
resultado resistía más semillas.

## 2. Rondas, en orden

### Ronda 0: auditoría de simetría entre ramas (sin reentrenar)

- **Pregunta:** ¿las ramas z y cruda reciben el mismo trato salvo la entrada?
- **Hecho:** se ejecutó el `main()` real de los dos scripts con sondas. Se verificó que:
  - el estado del RNG antes de construir la LSTM es idéntico en las dos ramas;
  - el split, el escalador de recompensa y los hiperparámetros efectivos son los mismos;
  - la semilla 0 de las dos ramas se reproduce bit a bit.
- **Resultado:** no hay bug. El orden de los lotes y los pesos iniciales difieren entre ramas porque
  la entrada tiene otro tamaño (26 frente a 16), algo estructural. Eso hace que el emparejamiento
  "semilla k de z con semilla k cruda" sea nominal: son sorteos independientes.

### (a) Diez semillas de LSTM con `latent_dim` = 16 (Autoencoder oficial, estado de 26 dims)

- **Pregunta:** ¿el valor 0 de la semilla tiene algo especial, o es variabilidad normal?
- **Diseño:** se fijaron de antemano las semillas nuevas 10, 20, 30, 40 y 50 (`MANIFEST.md`), que
  se suman a las 0–4 oficiales. Se revisó además que el código no trate `seed = 0` como falso: no
  lo hace.
- **Resultado:** las nuevas dan -6.6%, -3.9%, +22.1%, -25.4% y +9.9%. Con las 10: z gana 60/100
  pares, 6/10 semillas positivas, prueba de signo p = 0.75, Wilcoxon exacto p = 0.19. La semilla 0 es
  la cuarta más favorable al crudo de las 10.
- **Conclusión:** la semilla 0 es típica. Con 10 semillas **no se detecta** una ventaja del
  Autoencoder: las 5 originales (+14.0% de media) habían caído del lado favorable (las nuevas,
  -0.8%).

### Hallazgo sobre el estado: dos dimensiones siempre en cero

Verificado sobre los 80 episodios (4,800 estados):

- Las posiciones 22 y 23 del one-hot de fase **nunca se activan**.
- El tiempo restante de fase vale 0 en 4,720 de 4,800 estados; los otros 80 son el paso 0 de cada
  episodio.
- Las posiciones 20 y 21 son complementarias: juntas aportan un solo bit.

Desde la ronda (b) se usa un estado de 24 dimensiones (el oficial normalizado sin las columnas 22 y
23). **Quitarlas no cambia los resultados a nivel de grupo:** 10 corridas crudas de 24 frente a 10 de
26, permutación exacta p = 0.997, diferencia del 0.0%. Sí cambia cada corrida individual (hasta 49
puntos en una semilla), porque altera la inicialización. Es una advertencia más sobre comparar
semilla con semilla.

### (b) Barrido de `latent_dim` (estado de 24 dims, 3 semillas)

- **Pregunta:** ¿16 dimensiones comprimen demasiado poco para que se note el efecto?
- **Diseño (MANIFEST preregistrado):**
  - `latent_dim` ∈ {4, 8, 12, 16, 20}, con `hidden_dim = latent_dim`.
  - Un Autoencoder por valor (semilla 0), con el protocolo oficial.
  - LSTM con semillas 0–2, y rama cruda de 24 con semillas 0–2 compartida.
  - Ganador: la mayor mediana de las reducciones por semilla.
  - Antes de correr, los ejecutores se validaron reproduciendo tensor a tensor los pesos oficiales.
- **Resultado (mediana; pares ganados de 30):** 4: -13.6% (9); 8: +25.4% (24); **12: +38.2% (25)**;
  16: +27.5% (18); 20: +28.1% (20). Las 18 LSTM se cortaron por early stopping.
- **Conclusión:** gana 12. Más tarde se vio que las semillas 1 y 2 de la rama cruda compartida
  salieron especialmente malas e inflaron esas dos semillas en todos los valores.

### (c) `latent_dim` = 12 reforzado a 10 semillas de LSTM

- **Diseño:** semillas de LSTM 3–9 nuevas para z12 y para la rama cruda de 24. Las pruebas se
  fijaron de antemano, incluido reportar aparte las semillas no usadas para elegir el ganador.
- **Resultado:** 10/10 semillas positivas, 85/100 pares, mediana +19.9%, signo y Wilcoxon p = 0.002.
  Solo con las semillas 3–9: 7/7, p = 0.016. Control sin emparejar (*post hoc*): -21.8% frente al
  crudo de 24 (p = 0.001) y -21.7% frente al de 26 (p = 0.0002).
- **Conclusión de entonces:** una ventaja significativa **para ese Autoencoder** (semilla 0).

### (d) `latent_dim` = 8 reforzado a 10 semillas de LSTM

- **Diseño:** el mismo. Se fijó de antemano el criterio para comparar 8 con 12 (permutación exacta
  sin emparejar, α = 0.05).
- **Resultado:** 7/10 positivas, signo p = 0.34, Wilcoxon p = 0.19. Solo con las semillas 3–9: 4/7,
  mediana +0.2%, p = 1.00 / 0.69. Comparado con 12, z8 tiene un 13.3% más de error (p = 0.014).
- **Conclusión:** el efecto de 8 no se sostiene con semillas nuevas. 12 se distinguía de 8, pero con
  un único Autoencoder por valor.

### (e) Varianza entre Autoencoders: diseño de 3×3 (`latent_dim` = 12)

- **Diseño (preregistrado):** Autoencoders con semillas 0, 1 y 2 × LSTM con semillas 0, 1 y 2. Con
  12 dimensiones en los tres latentes, la LSTM de semilla s arranca con los mismos pesos y el mismo
  orden de lotes, así que la columna fija de verdad la LSTM. Métrica secundaria sin ruido de la rama
  cruda: el log de la media geométrica del `reward_mse` de z.
- **Resultado:** en la métrica que solo depende de z, la desviación entre Autoencoders es del 11.4% y
  entre semillas de LSTM del 3.9%. **El Autoencoder de semilla 0 es atípico y favorable** según el
  criterio fijado. Con los Autoencoders 1 y 2, z12 no se distingue del crudo. Además, el Autoencoder
  0 es el que peor reconstruye: la pérdida de reconstrucción no predice su utilidad para la
  dinámica.
- **Conclusión:** la ventaja de (c) dependía del Autoencoder concreto.

### (f) Grilla final de 5×5 (`latent_dim` = 12)

- **Diseño (preregistrado):**
  - Autoencoders con semillas 0–4 × LSTM con semillas 0–4, reutilizando lo existente: se
    entrenaron 2 Autoencoders y 14 LSTM nuevas.
  - ANOVA de dos factores sobre la métrica de z.
  - Decisión: IC del 95% por **bootstrap por conglomerados** (se remuestrean los Autoencoders y,
    dentro de cada uno, sus LSTM) de la diferencia entre las 25 corridas z y las 10 crudas de 24. Si
    el IC incluye 0, no hay evidencia.
- **Resultado:**
  - Estimación puntual: z tiene un **7.1% menos** de error.
  - **IC del 95%: de -21.7% a +12.0%**, que incluye 0.
  - ANOVA: el Autoencoder explica una varianza significativa (σ ≈ 16%, p = 0.0007); la LSTM no
    (σ ≈ 6%, p = 0.13).
  - De los cinco Autoencoders, el 0 es el mejor (230, frente a 262–367 de media geométrica) y el 4
    queda peor que el crudo (367 frente a 301).
- **Conclusión:** **sin evidencia** de ventaja ni de desventaja.

## 3. Conclusión (con la LSTM; para las otras arquitecturas, ver la sección 4)

Con la evidencia disponible, y controlando a la vez la varianza del Autoencoder y la del modelo
temporal, **no hay evidencia de que comprimir el estado con el Autoencoder mejore la predicción de la
recompensa frente al estado crudo, para ningún valor de `latent_dim` probado. Tampoco hay evidencia
de que la empeore.** Precisiones sobre el alcance:

- Solo `latent_dim` = 12 se evaluó con el control completo (5 Autoencoders × 5 LSTM). Los demás
  valores se probaron con un único Autoencoder. Ninguno mostró una ventaja que resistiera más
  semillas: 16 (10 semillas, p = 0.19), 8 (10 semillas, p = 0.19), y 4 y 20 solo con 3 semillas. Con
  `latent_dim` = 4 el resultado apuntaba en contra del Autoencoder (-13.6%, 9/30), pero sin prueba
  estadística.
- **La fuente de varianza dominante es qué Autoencoder se entrena**, más que la semilla del modelo
  temporal. El mismo `latent_dim` puede dar un Autoencoder claramente útil o claramente perjudicial,
  y la pérdida de reconstrucción no permite distinguirlos.
- **Relación con el Experimento 0 oficial:** usa un único Autoencoder (semilla 0, `latent_dim` = 16)
  y 5 semillas de LSTM, así que no controla esta fuente de varianza. Esta exploración no lo
  invalida ni lo reemplaza, pero indica que su conclusión ("mejora moderada y mayoritaria") depende
  de ese Autoencoder concreto y no se sostiene al ampliar las semillas.

## 4. Fase 2: LSTM, Transformer y TSMixer a `latent_dim` = 16

Las rondas anteriores solo usaron la LSTM. La Fase 2 repite el diseño de 5×5 con las tres
arquitecturas del Experimento 3.

- **Diseño (preregistrado en `ADDENDUM_transformer_tsmixer.md`):**
  - `latent_dim` = 16 (el valor oficial) y estado de 24 dimensiones.
  - Los **mismos** 5 Autoencoders (semillas 0–4) para las tres arquitecturas, con 5 semillas del
    modelo temporal por Autoencoder, y 10 corridas crudas por arquitectura.
  - Tope de 300 épocas y paciencia 15 para todas las ramas. Las 105 corridas se cortaron por early
    stopping.
  - Decisión con el mismo IC del 95% por bootstrap por conglomerados que la ronda (f).
- **Resultado** (media geométrica del `reward_mse`; negativo significa que el latente predice mejor):

| Arquitectura | Latente (25) | Crudo (10) | Efecto del Autoencoder | IC del 95% | Conclusión |
|---|---|---|---|---|---|
| LSTM | 271 | 301 | −10.0% | [−21.3%, +2.8%] | sin evidencia |
| Transformer | 439 | 329 | +33.4% | [+9.8%, +60.4%] | **el latente predice peor** |
| TSMixer | 679 | 478 | +42.0% | [+21.1%, +65.0%] | **el latente predice peor** |

- **Entre arquitecturas:** la ayuda del Autoencoder difiere entre la LSTM y cada una de las otras
  dos (los IC excluyen 0), pero no entre el Transformer y TSMixer.
- **Comparaciones múltiples:** son tres pruebas por arquitectura, sin corrección, así que el riesgo
  conjunto de algún falso positivo supera el 5% de cada una. Con Bonferroni (98.33%), una
  comprobación añadida después y no preregistrada, **ninguna conclusión cambia**: Transformer
  [+5.3%, +66.4%], TSMixer [+16.8%, +70.0%].
- **Fuentes de varianza (ANOVA 5×5):** en el Transformer domina el Autoencoder (p = 0.001). En la
  LSTM a 16, ninguna. En TSMixer ningún factor es significativo (Autoencoder p = 0.084, semilla
  p = 0.12) y los dos pesan parecido: ahí no domina ninguno.
- **Conclusión de la Fase 2:** con la LSTM el Autoencoder es neutro; con el Transformer y con TSMixer
  **perjudica** la predicción de la recompensa. Sobre el estado crudo las tres quedan en el orden del
  Experimento 3 (LSTM 301, Transformer 329, TSMixer 478), y la distancia crece sobre el latente.
  Buena parte de la desventaja de esas dos arquitecturas en el Experimento 3 viene, entonces, de
  trabajar sobre z.
- **Alcance:**
  - Una sola `latent_dim`.
  - Hiperparámetros del Experimento 3 sin ajustar a ninguna rama.
  - Solo la predicción de la recompensa, no el control.
  - No cambia la decisión del Experimento 3: el LSTM sigue siendo el modelo del sistema, y esta
    exploración refuerza esa elección.
- **Nota del entorno:** la evaluación de TSMixer se hizo con una copia literal de la evaluación
  oficial, porque un bloqueo de Windows impedía cargar matplotlib. Esa copia reprodujo exactamente
  los resultados ya guardados de LSTM y Transformer antes de usarse (detalle en `REPORT.md`,
  sección 13).

## 5. Dónde está el detalle (rama `exploratory/latent-dim-sweep`)

| Ronda | Archivos | Commits |
|---|---|---|
| (a) | [`results/exploratory/seed_zero_check/`](results/exploratory/seed_zero_check/): `MANIFEST.md`, `experiment_0_10seeds_combined.json` | (commiteada junto con este documento) |
| Paso 1, estado de 24 dims, validación | [`latent_sweep/MANIFEST.md`](results/exploratory/latent_sweep/MANIFEST.md), `VALIDATION.md`, `sweep.py` | `b5e1049`, `a68bab8` |
| (b) | `sweep_z{4,8,12,16,20}.json`, `sweep_summary.json` | `778b8d5` |
| (c) | `winner_z12_10seeds.json`, `winner_stats.out`, `calibration_raw24_vs_raw26.*`, `posthoc_unpaired.*` | `3b46dd4` |
| (d) | `ADDENDUM_z8.md`, `winner_z8_10seeds.json` | `7acbe21`, `813677e` |
| (e) | `ADDENDUM_ae_seeds.md`, `ae_variance.*` | `c9f3aa9`, `4b3cf21` |
| (f) | `ADDENDUM_grid5.md`, `grid5.py`, `grid5.json`, `grid5.out` | `fdc148d`, `8fe1ed6` |
| Fase 2 | `ADDENDUM_transformer_tsmixer.md`, `grid_arch.py`, `gridarch_{lstm,transformer,tsmixer}.json`, `grid_arch_A.out`, `grid_arch_B.out`, `grid_arch_lstm_transformer_tsmixer.json`, `multiple_comparisons.*`, `_DESCARTABLE_bloqueo_windows/` | `36ad129`, `78a7ccd`, `5d859df`, `7d317e9` y el commit de cierre |

Informe completo con todas las tablas:
[`results/exploratory/latent_sweep/REPORT.md`](results/exploratory/latent_sweep/REPORT.md)
(secciones 1–13). Los pesos y los datos derivados están en `models/checkpoints/exploratory_*`, no
versionados.
