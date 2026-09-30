# Addendum (escrito y commiteado antes de entrenar): grilla de 5×5 para `latent_dim` = 12

Pedido del autor. Pregunta: controlando a la vez la semilla del Autoencoder y la de la LSTM, ¿el
latente de `latent_dim` = 12 predice la recompensa mejor que el estado crudo de 24 dimensiones? El
MANIFEST y los addenda anteriores no cambian.

## Diseño (fijado)

- **Autoencoders** de `latent_dim` = 12 (`hidden_dim` = 12, protocolo oficial, estado de 24 dims):
  semillas **0, 1 y 2**, reutilizadas (`ae12/`, `ae12_seed1/`, `ae12_seed2/`), y **3 y 4**, nuevas
  (`ae12_seed3/`, `ae12_seed4/`).
- **LSTM** con semillas **0–4** por Autoencoder (25 celdas), con el protocolo de siempre (tope de
  300 épocas, paciencia 15), 3 en paralelo.
- **Rama cruda:** `raw24/seed0..9`, las 10 corridas existentes, sin reentrenar.

**Inventario, verificado en disco antes de escribir esto:**

| AE | LSTM que existen | LSTM nuevas |
|---|---|---|
| 0 (`z12/`) | 0–4 (y 5–9, fuera de la grilla) | ninguna |
| 1 (`z12_ae1/`) | 0–2 | 3, 4 |
| 2 (`z12_ae2/`) | 0–2 | 3, 4 |
| 3 (`z12_ae3/`) | — | 0–4 |
| 4 (`z12_ae4/`) | — | 0–4 |

**Tamaño real: 2 Autoencoders nuevos y 14 LSTM nuevas.** No son 16 como decía el pedido: las
semillas 3 y 4 del AE 0 ya existen (refuerzo a 10 semillas).

## Métricas (fijadas)

- **Principal por celda:** reducción mediana sobre h = 1..10 de (crudo24_s − z_{a,s})/crudo24_s,
  contra `raw24/seed s` (s = 0..4). Solo es descriptiva: incluye el ruido de la rama cruda compartido
  por columna.
- **Secundaria por corrida:** g = media sobre h = 1..10 del log del `reward_mse` de test, es decir,
  el log de la media geométrica. Es la métrica de decisión. Diferencias en g ≈ diferencias relativas;
  se reportan como exp(Δ) − 1.

## Análisis (fijado)

1. **Grilla de 5×5** de las dos métricas.
2. **ANOVA de dos factores sin réplica** sobre g (AE × semilla de LSTM, 5 × 5). Cuadrados medios,
   F con (4, 16) gl y su p-valor (distribución F, calculada con la beta incompleta regularizada y
   verificada contra valores críticos de tabla). Componentes σ²_AE = (CM_AE − CM_res)/5 y σ²_LSTM =
   (CM_LSTM − CM_res)/5, truncadas en 0. Se compara σ²_AE con σ²_LSTM. La misma descomposición sobre
   la métrica principal se reporta como descriptiva.
3. **Comparación contra el crudo** (estadístico Δ = media de g en las 25 corridas de z − media de g
   en las 10 de `raw24`; Δ < 0 significa que z tiene menos error):
   - **Intervalo de decisión: bootstrap por conglomerados.** Las 25 corridas no son independientes
     (vienen en 5 grupos por Autoencoder, la fuente de varianza dominante), así que se remuestrean
     con reemplazo los 5 Autoencoders; dentro de cada uno, sus 5 corridas de LSTM, y aparte las 10
     corridas crudas. Se recalcula Δ. B = 100,000, semilla del generador 12345, IC del 95% por
     percentiles (2.5 y 97.5).
   - **Reportada, no decisoria:** la prueba de permutación sin emparejar por corrida (25 frente a 10;
     Monte Carlo con 10⁶ permutaciones y semilla 12345, porque C(35,10) ≈ 1.8·10⁸). Trata las 35
     corridas como intercambiables e ignora la agrupación por Autoencoder, así que es
     anticonservadora. Se incluye porque la pidió el autor y por continuidad con el control
     *post hoc* anterior.
4. **Criterio de conclusión (fijado):**
   - Si el IC del 95% por conglomerados de Δ **no incluye 0** y está por debajo, hay evidencia de que
     z12 predice mejor la recompensa que el estado crudo.
   - Si no incluye 0 y está por encima, hay evidencia de que predice peor.
   - **Si incluye 0, no hay evidencia**, y se reporta así, sin buscar otro corte de los datos que sí
     salga significativo.
   - La permutación por corrida no puede cambiar esta conclusión.
- Salvedad fijada: con 5 Autoencoders, el bootstrap por conglomerados tiene pocos grupos y su IC
  puede ser algo estrecho. Si la conclusión depende de un margen pequeño, se dirá.
