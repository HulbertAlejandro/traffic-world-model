# Addendum (escrito y commiteado antes de calcular): ¿cuánta incertidumbre viene de tener solo 12 episodios de test? (Fase 1)

Pedido del autor. Todos los resultados de `reward_mse` del proyecto se evaluaron sobre **el mismo**
split de test de 12 episodios. Los IC de las secciones 11, 13 y 14 remuestrean Autoencoders y
semillas, pero **no** episodios. Esta fase aísla la variabilidad por la composición del test y la
compara con la variabilidad por semilla ya medida.

**No se entrena ni se simula nada.** Solo se hacen rollouts de checkpoints existentes sobre los
datos de test existentes. Los episodios nuevos con SUMO quedan para una Fase 2, solo si este
resultado la justifica.

## Conjunto de test (verificado en disco antes de escribir esto)

- 12 episodios, con `episode_id` 8, 12, 13, 14, 35, 36, 47, 53, 64, 65, 66 y 77.
- 60 pasos cada uno, así que cada episodio aporta 35 ventanas por horizonte (420 en total).
- Son los mismos 12 episodios en `datasets/processed/test{,_latent,_raw_seq}.npz` y en los test de
  24 dims y de los Autoencoders de la exploración.

## Evaluación por episodio

- Se usa la evaluación oficial **importada**: `load_world_model`, `load_reward_scaler`,
  `load_episodes`, `rollout_episode` y `aggregate_by_horizon` de `evaluation/world_model_evaluation.py`.
  matplotlib vuelve a cargar en este equipo, así que no se usa la copia de
  `_DESCARTABLE_bloqueo_windows/`.
- Por checkpoint y episodio, E[e, h] = media del error cuadrático de la recompensa en las 35 ventanas
  del episodio e en el horizonte h.
- **Control obligatorio:** `aggregate_by_horizon` sobre todos los registros debe reproducir
  **exactamente** (igualdad de floats) los valores ya guardados: `gridarch_*.json`,
  `experiment_0_multiseed_300ep.json` y `results/world_model{,_transformer,_tsmixer}_evaluation.json`.
  Si algo no coincide, se detiene y se reporta.
- Como todos los episodios tienen las mismas ventanas, el `reward_mse` de un conjunto de episodios es
  la media de sus E[e, h].

## Bootstrap de episodios (fijado)

- B = 10,000 y `np.random.default_rng(12345)`.
- Cada réplica sortea 12 episodios con reemplazo: `ei = rng.integers(0, 12, (B, 12))`, generado
  **una sola vez**.
- **El mismo sorteo se aplica a todos los modelos y corridas de la réplica**, es decir, emparejado:
  todas las corridas se evaluaron sobre el mismo test, y así se sigue comparando igual con igual.
- En cada réplica: `reward_mse_h` = media de E[e, h] sobre los episodios sorteados, y
  g = media sobre h de log(`reward_mse_h`), la métrica de toda la Fase 2. Se reporta como media
  geométrica, exp(g).
- IC del 95% por percentiles (2.5 y 97.5).

## Qué se calcula

**E1. Checkpoints oficiales, una corrida cada uno:** LSTM (`world_model_best.pt`), Transformer y
TSMixer (`world_model_{transformer,tsmixer}_best.pt`), sobre `test_latent.npz` oficial.

- Por modelo: g con su IC del 95% solo por episodios, y `reward_mse` por horizonte con su IC.
- Contrastes emparejados: LSTM − Transformer y LSTM − TSMixer (diferencia de g), con IC solo por
  episodios. Se comparan con los IC por semillas de la sección 14.
- No existe un checkpoint crudo "oficial" suelto. El contraste z − crudo oficial es el Experimento 0
  de E2.

**E2 a E4. Descomposición por fuentes,** sobre el mismo estadístico Δ ya reportado:

| | Contraste | Corridas |
|---|---|---|
| E2 | Experimento 0 oficial (`exp0_multiseed_300ep`, 26 dims, Autoencoder oficial): media g(z) − media g(crudo) | 5 z (`z/seed0-4`) y 5 crudas (`raw/seed0-4`) |
| E3 | Sección 11 (`latent_dim` = 12): media g(z12) − media g(crudo 24) | 25 z (5 Autoencoders × 5) y 10 crudas |
| E4 | Sección 13 (`latent_dim` = 16), por arquitectura: media g(z16) − media g(crudo 24) | 25 z y 10 crudas por arquitectura |
| E4' | Sección 14: los tres pares entre arquitecturas sobre z16 | 25 y 25 |

Para cada contraste, tres IC del 95% con B = 10,000:

1. **Solo semillas:** el remuestreo ya usado (conglomerados por Autoencoder donde los hay; E2 sin
   conglomerados, remuestreando las 5 semillas de cada rama), con los 12 episodios fijos.
   - Generador `default_rng(12345)` propio, en el orden de las secciones 11, 13 y 14.
   - Con B = 10,000 no coincidirá al dígito con los IC publicados (B = 100,000). Se muestran los dos.
2. **Solo episodios:** todas las corridas fijas, con los episodios remuestreados (emparejados).
3. **Conjunto:** semillas y episodios a la vez. Cada réplica combina el sorteo de semillas de su
   índice con el de episodios de su índice.

**Medida de comparación:** el ancho del IC en log (hi − lo). Se reporta, por contraste:

- la **razón de anchos** entre solo episodios y solo semillas;
- la **fracción de varianza por episodios**, var(solo episodios) / [var(solo episodios) +
  var(solo semillas)], con las varianzas de las réplicas.

Para los niveles de un solo modelo (E1), la desviación estándar de g por episodios se compara con la
desviación estándar de g entre corridas del mismo tipo:

- 5 semillas de z del Experimento 0 para la LSTM;
- 25 corridas z16 por arquitectura, para las tres.

## Criterio de lectura (fijado)

Por contraste, según la razón de anchos (episodios / semillas):

| Razón de anchos | Lectura |
|---|---|
| < 0.33 | **fuente menor** |
| 0.33 a 1 | **fuente relevante**, pero menor que las semillas |
| ≥ 1 | **fuente dominante** o igual a las semillas |

**Lo que decide si hace falta la Fase 2** es si alguna conclusión ya reportada **cambia** con el IC
conjunto, es decir, si un IC que excluía 0 pasa a incluirlo o al revés. Si ninguna cambia y todas las
razones son < 0.33, la recomendación será que la Fase 2 no es prioritaria para estas conclusiones. En
cualquier otro caso, se describe qué conclusiones dependen del test.

## Influencia de cada episodio (descriptivo)

Para g del LSTM oficial, Δ del Experimento 0, Δ de la LSTM a 16 y LSTM − Transformer:

- en las réplicas de las colas (el 2.5% inferior y el 2.5% superior del estadístico en el bootstrap
  de solo episodios), el número medio de apariciones de cada episodio. El valor esperado es 1.0; se
  reportan los más sobrerrepresentados en cada cola;
- jackknife: el cambio del estadístico al quitar cada episodio (11 episodios);
- el peso de cada episodio en el error total de la LSTM oficial en h = 10 (su E[e, 10] / suma).

## Limitaciones declaradas de antemano

- **El bootstrap solo recombina los 12 episodios existentes.** No puede revelar regímenes de tráfico
  que no estén entre ellos: mide la inestabilidad del resultado ante la composición, no la cobertura
  del test.
- **Con n = 12, el bootstrap de percentiles tiende a quedarse corto:** la varianza queda sesgada
  hacia abajo en un factor 11/12 y la cobertura real queda por debajo del 95%. Las razones de anchos
  son, entonces, una cota inferior aproximada de la importancia de esta fuente.
- Las ventanas dentro de un episodio están correlacionadas. Por eso se remuestrean episodios, no
  ventanas.

## Reglas

- Script `episode_bootstrap.py` en esta carpeta. Guarda E[e, h] de cada checkpoint en
  `episode_errors.json`, un archivo pequeño.
- Ningún archivo oficial cambia. Se verifican los md5 y el inventario de `.pt`/`.zip`/`.npz` de
  `models/checkpoints/` antes y después, y que no aparezca ningún checkpoint ni dataset nuevo.
- Ningún número publicado cambia. Si esta fase sugiere que alguna conclusión depende del test, se
  describe; no se reescribe nada sin decisión del autor.
