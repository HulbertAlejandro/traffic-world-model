# Addendum (escrito y commiteado antes de entrenar): ¿ayuda el Autoencoder con Transformer y TSMixer?

Pedido del autor (Fase 2). Pregunta: con el mismo rigor que la grilla de 5×5 de la LSTM, ¿el
latente de `latent_dim` = 16 predice la recompensa mejor que el estado crudo cuando el modelo
temporal es el Transformer o TSMixer? ¿Y la ayuda del Autoencoder cambia según la arquitectura? El
MANIFEST y los addenda anteriores no cambian.

## Alcance (fijado)

- **Solo `latent_dim` = 16**, el valor oficial. Un barrido de `latent_dim` para estas arquitecturas
  es una extensión posible, **no parte de este experimento**, y no se hará sin pedido explícito.
- **Tres arquitecturas con el mismo diseño:** LSTM, Transformer y TSMixer. La LSTM se incluye con su
  propia grilla a 16 (decisión del autor), porque la grilla de 5×5 anterior fue a 12.
- Estado de 24 dimensiones (el oficial sin las columnas 22 y 23, siempre en cero), como toda la
  exploración.

## Protocolo común (fijado; decisión del autor)

- **Tope de 300 épocas y paciencia 15 para las tres arquitecturas y las dos ramas.** El Experimento 3
  oficial usaba un tope de 100, pero TSMixer no converge en 100 (en la verificación su mejor época
  fue la 99). Se iguala al alza para no cortar una rama antes de converger, la asimetría que ya se
  corrigió en el Experimento 0.
- LSTM: el protocolo de `training/train_world_model.py` (ejecutor `sweep.py train-lstm`, validado).
- Transformer y TSMixer: `train()` de `training/train_world_model_{transformer,tsmixer}_raw.py`, la
  **misma función para la rama latente y la cruda**. Con los datos oficiales, reproduce tensor a
  tensor los checkpoints del Experimento 3 (`VALIDATION.md`). El escalador de recompensa es el
  compartido; sus valores coinciden con los del ajuste en train.
- **Autoencoders** de `latent_dim` = 16 (`hidden_dim` = 16, protocolo oficial, estado de 24 dims),
  semillas **0–4**. Son los **mismos 5 Autoencoders para las tres arquitecturas**. No se usa el
  Autoencoder oficial (entrenado sobre 26 dimensiones) porque la rama cruda es de 24.
- 3 corridas en paralelo.

## Inventario, verificado en disco antes de escribir esto

| Qué | Existe | Nuevo |
|---|---|---|
| Autoencoders de 16 | semilla 0 (`ae16/`) | semillas 1–4 (`ae16_seed{1..4}/`): **4** |
| LSTM sobre z16, 5 × 5 | AE 0 × semillas 0–2 (`z16/seed{0,1,2}`, tope 300) | AE 0 × 3, 4 (`z16/`) y AE 1–4 × 0–4 (`z16_ae{a}/`): **22** |
| LSTM cruda de 24 | semillas 0–9 (`raw24/`, tope 300) | **0** |
| Transformer sobre z16, 5 × 5 | — | **25** (`tf16_ae{a}/seed{s}`) |
| Transformer crudo de 24 | — | semillas 0–9: **10** (`tf_raw24/`) |
| TSMixer sobre z16, 5 × 5 | — | **25** (`ts16_ae{a}/seed{s}`) |
| TSMixer crudo de 24 | — | semillas 0–9: **10** (`ts_raw24/`) |

**Total nuevo: 4 Autoencoders y 92 corridas de modelo temporal** (22 LSTM, 35 Transformer y 35
TSMixer).

**Ejecución por partes (decisión del autor):**

- **Parte A:** Autoencoders, LSTM a 16 y Transformer (57 corridas; unas 2–4 h). Después, un reporte
  intermedio.
- **Parte B:** TSMixer (35 corridas; unas 2–3.5 h).

## Análisis por arquitectura (fijado; el mismo que la sección 11 de REPORT.md)

- Métrica de decisión por corrida: g = media sobre h = 1..10 del log del `reward_mse` de test.
  También se reporta, de forma descriptiva, la reducción mediana por celda contra la corrida cruda
  de la misma semilla (0–4).
- ANOVA de dos factores sin réplica (5 × 5) sobre g: F con (4, 16) gl y su p-valor; σ²_AE y σ²_LSTM
  (aquí "LSTM" significa la semilla del modelo temporal).
- Δ = media de g en las 25 corridas latentes − media de g en las 10 crudas de la **misma
  arquitectura**.
- **IC del 95% por bootstrap por conglomerados** (se remuestrean los Autoencoders; dentro de cada
  uno, sus 5 corridas; aparte, las 10 crudas), con B = 100,000 y semilla 12345.
- Permutación por corrida: se reporta, pero no decide (es anticonservadora).
- **Criterio:**
  - IC por debajo de 0: **evidencia de que el latente predice mejor**.
  - IC por encima de 0: **evidencia de que predice peor**.
  - Si incluye 0: **SIN EVIDENCIA**, sin buscar otro corte de los datos.

## Comparación entre arquitecturas (fijada)

- **Tabla única** con Δ (como exp(Δ) − 1) y su IC del 95% para LSTM, Transformer y TSMixer.
- **¿La ayuda del Autoencoder difiere entre arquitecturas?** Para cada par (A, B) se calcula
  Δ_A − Δ_B con un bootstrap por conglomerados **conjunto**: se remuestrean los mismos índices de
  Autoencoder para las dos arquitecturas (comparten los 5 Autoencoders), las semillas del modelo
  temporal dentro de cada uno, y las corridas crudas de cada arquitectura por separado. B = 100,000
  y semilla 12345.
- Mismo criterio: si el IC de Δ_A − Δ_B incluye 0, **no hay evidencia** de que la ayuda difiera
  entre A y B. Son 3 comparaciones; se reportan sin corrección y se señala que son varias.
- La tabla completa espera a la Parte B. En el reporte intermedio de la Parte A solo aparecen la LSTM
  y el Transformer, y su comparación.

## Reglas

Ninguna corrida se repite ni se excluye por su resultado. Una corrida que llegue al tope de 300 sin
que corte el early stopping se reporta como tal. Los md5 oficiales se verifican antes y después.
