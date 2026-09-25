# PROJECT_STATUS.md — Estado al momento de este handoff

Última verificación: commit `a224812`, 62/62 tests en verde. **Lo más reciente:
verificación del RL directo con 3x presupuesto** (30,000 pasos de entrenamiento; modelos
fuera del repositorio, checkpoints oficiales intactos). Con ese presupuesto, el RL directo
**alcanza un desempeño comparable** al del World Model (-335.24 frente a -326.79; p = 0.71
a nivel de semilla), pero **consumiendo ~8.5 veces más interacciones reales** (39,000
frente a ~4,600 por semilla). Esto precisa el resultado de la segunda ronda, que sigue
vigente como comparación con el presupuesto original: con 13,000 interacciones por
semilla (~2.8 veces las del World Model), el RL directo queda claramente por debajo en
desempeño medio y en consistencia. La ventaja demostrada del World Model es de
**eficiencia en interacciones reales** (ver la sección siguiente). Antes de esto se cerró
el Experimento 3: la LSTM se mantiene como modelo temporal.

Resultado oficial de la segunda ronda del escenario asimétrico (sin cambios): dataset de
80 episodios (semilla de SUMO por episodio), selección del checkpoint del sueño en SUMO
real y `VecNormalize` en los dos PPO. **Con 3 semillas de entrenamiento por método (90
episodios cada uno): PPO del sueño -326.79, PPO directo (10,000 pasos) -453.74, tiempo
fijo -411.27.** El método World Model supera a tiempo fijo en las tres semillas y usa
menos interacciones reales que el RL directo: 13,800 pasos de SUMO en total para sus 3
semillas (4,800 del dataset, recolectado una sola vez y compartido, más 3 × 3,000 de
selección; ~4,600 por semilla), frente a 39,000 del RL directo (3 × 13,000). Con ese
presupuesto, la ventaja sobre el RL directo es clara a nivel de episodio (4.56 errores
estándar), pero **no concluyente a nivel de semilla** (p = 0.145 con 3 semillas por
método), porque el RL directo varía mucho entre semillas. Siguen abiertos los episodios
catastróficos (menos, pero presentes) y la función de valor del PPO directo. Ver la
sección "Escenario asimétrico, segunda ronda".

## ✅ Verificación: RL directo con 3x presupuesto (30,000 pasos) — alcanza al World Model, con ~8.5 veces sus interacciones reales

### 1. Motivación y protocolo

La segunda ronda asimétrica dejó abierta una limitación explícita (punto 9 de esa
sección): con solo 10,000 pasos reales de entrenamiento no se podía descartar que más
presupuesto cerrara la brecha del RL directo o estabilizara sus semillas. Esta
verificación la responde. Es una **verificación, no un reemplazo**: los checkpoints
oficiales de `models/checkpoints/controller_direct/` (10,000 pasos) no se tocaron (md5 de
los 61 archivos de `controller/` y `controller_direct/` verificado igual antes y después)
y los modelos de 30,000 pasos quedaron fuera del repositorio.

- Las 3 semillas (0, 1, 2) se entrenaron con `training/train_controller_direct.py` sin
  modificar el script. Solo se cambiaron, desde fuera, `DIRECT_TOTAL_TIMESTEPS`
  (10,000 → 30,000), la semilla y el directorio de salida. Todo lo demás es idéntico:
  `normalize_obs=True`, `normalize_reward=True`, `EvalCallback` cada 1,000 pasos con 5
  episodios en las semillas fijas 20000–20004, y el mismo flujo de semillas de
  entrenamiento.
- Chequeo de determinismo: las primeras 10 evaluaciones periódicas de cada semilla
  coinciden exactamente con las de las corridas de 10,000 pasos.
- Evaluación en SUMO real con el protocolo de siempre (`seed_base` 3000 y 5000, 15
  episodios cada una, el mismo bucle y el mismo `load_obs_normalizer` de
  `evaluate_final_comparison.py`). En la misma corrida se reevaluaron las 3 semillas del
  sueño y las 3 del directo de 10,000 pasos: **reproducen exactamente las medias por
  semilla documentadas en la segunda ronda**.
- Los p-valores se calcularon con una implementación propia de Welch y Mann-Whitney
  (`scipy` no está instalado), validada reproduciendo el p = 0.145 ya documentado.

### 2. Entrenamiento (Paso 1)

| Semilla | Mejor t (10k) | Mejor t (30k) | Curva de evaluación periódica con 30k |
|---|---|---|---|
| 0 | 9,000 | 26,000 (-338) | Entre -353 y -444 de t=6,000 a t=21,000; **tres evaluaciones catastróficas** (-7,578 a -7,615 en t=22,000–24,000 y -8,233 en t=29,000), después se recupera |
| 1 | 6,000 | 19,000 (-240) | Mejora sostenida desde t=14,000 y se estabiliza en ~-250/-280 |
| 2 | 3,000 | 29,000 (-443) | Sale de la meseta de la regla trivial hacia t=15,000; su mejor punto está casi al final, así que puede no haber terminado de mejorar |

~20 minutos por semilla (las 3 en paralelo).

### 3. Evaluación en SUMO real (Paso 2)

```
semilla | seed 3000          | seed 5000          | media 30 | mediana 30 | gana t.fijo | < -600 | peor    | acuerdo con la regla*
      0 | -359.01 +/- 188.47 | -352.59 +/- 192.01 |  -355.80 |    -273.95 |    22/30    |  4/30  |  -853.4 | 52.7%
      1 | -285.87 +/- 145.20 | -314.37 +/- 161.09 |  -300.12 |    -243.85 |    26/30    |  3/30  |  -744.2 | 49.7%
      2 | -353.73 +/- 244.57 | -345.88 +/- 176.36 |  -349.80 |    -283.75 |    25/30    |  4/30  | -1058.5 | 55.1%
Media 90 episodios: -335.24 (std 190.15, mediana -271.60)
```

\* Porcentaje de los pasos no bloqueados por `min_green` en que la acción del PPO
coincide con "pedir la fase contraria" (`1 - green_phase`), medido sobre su propia
trayectoria. Con 10,000 pasos: 37.0%, 48.7% y **97.8%** (semillas 0, 1, 2).

### 4. RL directo: 10,000 frente a 30,000 pasos (Paso 3)

| | Media 90 | Mediana | Desv. de las medias por semilla | Rango de medias por semilla | < -600 | Ganan a tiempo fijo |
|---|---|---|---|---|---|---|
| Directo, 10,000 pasos (oficial) | -453.74 | -381.75 | **96.2** | -540.89 a -350.47 | 23/90 | 49/90 |
| **Directo, 30,000 pasos** | **-335.24** | **-271.60** | **30.6** | -355.80 a -300.12 | 11/90 | 73/90 |
| PPO del sueño (referencia) | -326.79 | -293.00 | 19.3 | -345.42 a -306.82 | 5/90 | 75/90 |

- **La semilla 2 dejó de colapsar a la regla trivial.** Con 10,000 pasos, su mejor
  checkpoint era el de t=3,000: todavía dentro de la meseta inicial, donde la política
  pide siempre la fase contraria (97.8% de acuerdo con la regla y 25/30 episodios a menos
  de 1 punto de ella). Las evaluaciones posteriores fueron todas peores (de -576 a -969), así que
  el checkpoint seleccionado se quedó en esa meseta. Con 30,000 pasos, la semilla sale
  de la meseta hacia t=15,000, sigue mejorando hasta t=29,000, y su checkpoint final
  tiene un acuerdo del 55.1% con la regla (el nivel de las otras dos semillas) y 0/30
  episodios pegados a ella. Su media pasa de -540.89 a -349.80. La lectura: el colapso
  no era una propiedad del método, sino un síntoma de presupuesto insuficiente para
  salir de un óptimo local temprano.
- **La consistencia entre semillas mejora mucho**: la desviación de las medias por
  semilla baja de **96.2 a 30.6**, cerca de la del sueño (19.3).
- **Los episodios catastróficos bajan de 23/90 a 11/90**, todavía más del doble que en el
  sueño (5/90). El peor episodio del directo con 30k (-1058.5) es el peor medido en el
  proyecto.

### 5. World Model frente a RL directo (Paso 4)

| Criterio | Sueño vs. directo 10k | Sueño vs. directo 30k |
|---|---|---|
| Diferencia de medias | +126.95 a favor del sueño | **+8.46** a favor del sueño |
| Episodio a episodio, mismo escenario, las 9 combinaciones de semillas | sueño gana 188/270 (70%) | **sueño gana 111/270 (41%)** |
| Episodio a episodio, semilla i vs. semilla i | sueño gana 64/90 | sueño gana 39/90 |
| Nivel episodio (90 vs 90), Welch | t = 4.56, p ≈ 1e-5 | **t = 0.33, p = 0.74** |
| Nivel episodio, Mann-Whitney | p ≈ 1.4e-5 | p = 0.10 (mediana mejor en el directo) |
| Nivel semilla (3 vs 3), Welch | t = 2.24, gl ≈ 2.2, p = 0.145 | **t = 0.41, gl ≈ 3.4, p = 0.71** |

Con 30,000 pasos, el RL directo **alcanza un desempeño comparable** al del World Model:
la diferencia de medias (8.46 puntos) no es significativa a ningún nivel, el directo gana
la mayoría de las comparaciones episodio a episodio y tiene mejor mediana, y el sueño
conserva menos episodios catastróficos (5/90 frente a 11/90; esta diferencia no se
sometió a prueba).

### 6. Contabilidad de interacciones reales

| | Entrenamiento | Selección / evaluación periódica | Por semilla | 3 semillas |
|---|---|---|---|---|
| World Model (PPO del sueño) | 4,800 del dataset, recolectado una vez y compartido (1,600 por semilla) | 3,000 | **~4,600** | **13,800** |
| Directo, 10,000 pasos | 10,000 | 3,000 (10 evaluaciones × 5 × 60) | 13,000 (~2.8x) | 39,000 |
| Directo, 30,000 pasos | 30,000 | 9,000 (30 evaluaciones × 5 × 60) | **39,000 (~8.5x)** | **117,000** |

El overhead de evaluación periódica crece con el presupuesto (una evaluación cada 1,000
pasos) y se cuenta como interacción real, igual que en la segunda ronda.

### 7. Salvedad metodológica: selección del checkpoint

Con 30,000 pasos, cada semilla elige su mejor checkpoint entre 30 evaluaciones
periódicas en vez de 10, sobre los mismos 5 escenarios fijos. Eso le da más
oportunidades de encontrar un candidato que rinda bien en esos 5 escenarios. La mejora
**sí se confirma** en los 30 escenarios de prueba (`seed_base` 3000 y 5000), que son
independientes de los de selección, así que no es un artefacto de la selección. Pero una
parte de la ganancia puede venir de esa búsqueda más amplia y no solo de más
entrenamiento, y esta verificación no separa ambos efectos.

### 8. Interpretación

Esta verificación **precisa** el resultado de la segunda ronda, no lo corrige. La
comparación con el presupuesto original sigue siendo válida: con 13,000 interacciones
reales por semilla (~2.8 veces las del World Model), el RL directo queda claramente por
debajo del sueño en desempeño medio y en consistencia. Lo que se añade es qué pasa al
triplicar el presupuesto: **el RL directo alcanza un desempeño comparable, pero
consumiendo ~8.5 veces más interacciones reales que el World Model** (39,000 frente a
~4,600 por semilla). La conclusión del proyecto se formula entonces así: el World Model
llega al mismo nivel de control con una fracción de las interacciones reales. Su
ventaja demostrada es de **eficiencia en interacciones**, que es exactamente lo que
pregunta la pregunta de investigación; no es una ventaja de control a igualdad de
interacciones ilimitadas. Queda sin medir la curva completa de desempeño frente a
interacciones del RL directo (solo hay dos puntos, 13,000 y 39,000 por semilla), que
diría con más precisión cuántas interacciones necesita para alcanzar al World Model.

## ✅ Experimento 3: Transformer y TSMixer como alternativas a la LSTM — se mantiene la LSTM

### 1. Qué se construyó

Commits `90d7641` a `f49cf09` (62/62 tests):

- `LatentDynamicsTransformer` (`models/world_model/transformer.py`): proyección lineal de
  `(z, a)` a `d_model=128`, codificación posicional sinusoidal, 2 capas
  `TransformerEncoderLayer` (4 cabezas, `dim_feedforward=256`), salida del último paso,
  sin máscara causal (el objetivo está fuera de la ventana, no hay fuga de futuro).
- `LatentDynamicsTSMixer` (`models/world_model/tsmixer.py`): 2 bloques de mezcla temporal
  (`LayerNorm` + `Linear(16, 16)` + ReLU sobre la secuencia transpuesta, residual) y de
  mezcla de variables (`LayerNorm` + MLP `18 → 128 → 18`, residual), salida del último
  paso. Exige `sequence_length` fijo. La dimensión de variables se queda en
  `latent_dim + action_dim = 18`, como en el TSMixer original.
- Ambas implementan `TemporalModel`, con las mismas dos cabezas (`ẑ`, `r̂`) y las mismas
  validaciones de forma que la LSTM.
- `build_world_model` en `evaluation/world_model_evaluation.py` elige la clase según la
  clave `"architecture"` del `.json` del checkpoint (sin clave → `"lstm"`, así que los
  checkpoints anteriores y el Dream Environment no cambian).
- `training/train_world_model_{transformer,tsmixer}.py` **importan** el protocolo de
  `train_world_model.py` (semilla, Adam con `weight_decay=1e-4`, lr, batch, 100 épocas,
  early stopping con paciencia 15, pérdida) y **leen** el `reward_scaler.json` existente
  en vez de recalcularlo. La LSTM **no se reentrenó**: se usó `world_model_best.pt` tal
  cual (md5 verificado igual antes y después; su reporte de evaluación regenerado es
  idéntico byte a byte al anterior).
- `scripts/evaluate_world_model_{transformer,tsmixer}.py` (espejo de
  `evaluate_world_model.py`: mismo test split, mismos horizontes, mismo baseline) y
  `scripts/compare_experiment_3.py`.

**Criterio de decisión, fijado antes de ver resultados**: un candidato reemplaza a la
LSTM solo si gana la mayoría estricta de los horizontes en `reward_mse` **y** su
reducción mediana frente a la LSTM es ≥ 7.3% (la mitad del 14.7%, la reducción mediana
que justificó mantener el Autoencoder en el Experimento 0, segunda ronda). Si la LSTM
gana la mayoría, se mantiene por predecir mejor. Si el resultado está dividido, se
mantiene por estar ya integrada y validada en todo el pipeline.

### 2. Entrenamiento

| | Parámetros | Épocas | Mejor validación (época) | Train en esa época |
|---|---|---|---|---|
| LSTM (existente) | 77,969 | 100 | 0.140486 | — |
| Transformer | 269,585 | 74 (early stopping) | **0.120795** (59) | 0.052817 |
| TSMixer | 10,519 | 100 (tope) | 0.188491 (99) | 0.151450 |

El Transformer tuvo un pico de inestabilidad entre las épocas 40 y 45, del que se
recuperó. TSMixer tiene una curva suave, casi sin brecha entre train y validación, pero su
mejor época fue la 99 de 100: había que verificar si estaba subentrenado (punto 4).

### 3. Resultado (`reward_mse` en test, 12 episodios, 420 ventanas por horizonte)

```
  h |     LSTM | Transformer |  TSMixer | gana
  1 |   57.051 |      59.311 |   79.075 | LSTM
  2 |  127.288 |     167.358 |  217.487 | LSTM
  3 |  204.492 |     309.029 |  445.732 | LSTM
  4 |  279.212 |     489.156 |  564.000 | LSTM
  5 |  349.542 |     673.581 |  728.701 | LSTM
  6 |  409.317 |     819.326 |  976.982 | LSTM
  7 |  494.946 |     895.629 | 1369.184 | LSTM
  8 |  591.838 |    1026.694 | 1740.368 | LSTM
  9 |  757.966 |    1089.673 | 2123.745 | LSTM
 10 |  993.291 |    1155.337 | 2530.505 | LSTM
```

**La LSTM gana en 10/10 horizontes frente a ambas alternativas**, con una reducción
mediana de `reward_mse` de 38.1% frente al Transformer y de 56.1% frente a TSMixer
(medida relativa a cada candidato). Ninguno de los dos se acerca al umbral. **Decisión:
se mantiene la LSTM porque predice mejor**, con el argumento adicional de que ya está
integrada y validada en todo el pipeline. Reemplazarla exigiría repetir la cadena
completa (Dream Environment, selección por SUMO real, PPO con 3 semillas).

Hipótesis de la propuesta, evaluadas en este escenario: **H4** (el Transformer tiene un
comportamiento predictivo distinto al de la LSTM) se confirma, y la diferencia va en
contra del Transformer en la recompensa. **H6** (TSMixer iguala el error de la LSTM y del
Transformer) se rechaza: es el peor de los tres en todos los horizontes.

### 4. Verificación de convergencia de TSMixer

- **Con el protocolo compartido y `EPOCHS=300`** (override solo para esa corrida; el
  default del script sigue en 100): el early stopping cortó en la época 114. Las primeras
  100 épocas fueron idénticas a la corrida original (misma semilla) y la mejor siguió
  siendo la 99, así que el checkpoint oficial y su reporte no cambiaron (verificado byte
  a byte). Esto no probaba 300 épocas de entrenamiento.
- **Corrida complementaria de 300 épocas completas, sin early stopping**, escrita fuera
  de `models/checkpoints/` y `results/` (el checkpoint oficial no se tocó, md5
  verificado). La validación se estanca desde la época ~100 (media 0.194 entre 101 y 150,
  0.192 entre 151 y 200, 0.194 entre 201 y 250, 0.198 entre 251 y 300; mejor 0.184008 en
  la época 188) mientras train sigue bajando (0.138 → 0.096): el modelo ya había
  convergido y al final empieza a sobreajustar. En test, ese mejor checkpoint mejora algo
  a TSMixer (h=1: 79.1 → 70.9; h=10: 2530.5 → 1710.8), pero **la LSTM sigue ganando en
  los 10 horizontes**, con una reducción mediana de 44.5% frente a él. **La conclusión se
  mantiene.** El checkpoint oficial de TSMixer sigue siendo el del protocolo compartido,
  el mismo de la tabla del punto 3.

### 5. Hallazgo: el Transformer predice mejor el estado latente, pero peor la recompensa

Frente a la LSTM, el Transformer tiene **menor `latent_mse` en los 10 horizontes**
(h=1: 0.0894 frente a 0.1035; h=10: 0.4089 frente a 0.4954) y también **menor pérdida de
validación combinada** (0.1208 frente a 0.1405). Esa pérdida es la suma del MSE latente y
del MSE de recompensa normalizada, y es la métrica con la que se elige el mejor
checkpoint. Sin embargo, su **`reward_mse` es peor en los 10 horizontes**: casi empata a
un paso (59.3 frente a 57.1, +4%), la diferencia crece hasta h=6 (+100%, 819 frente a
409) y h=8 (mayor diferencia absoluta, 435), y después se estrecha (+44% en h=9, +16% en
h=10) sin llegar a invertirse.

Es **el mismo patrón de fondo que el precedente del PPO del sueño** (sección "Escenario
asimétrico, segunda ronda", punto 4): allí el reward imaginado del Dream Environment tenía
una correlación de Pearson de +0.08 con el reward real en SUMO, y el checkpoint con mejor
reward imaginado fue el peor en SUMO real. En los dos casos, **la métrica que se optimiza
o con la que se selecciona durante el entrenamiento no predice la métrica que realmente
importa**. Aquí, una mejor pérdida de validación no se tradujo en una mejor predicción de
la recompensa, que es lo que el Dream Environment entrega al controlador. Consecuencia
práctica: comparar arquitecturas del modelo temporal por pérdida de validación habría
elegido al Transformer, la opción equivocada. La comparación tiene que hacerse sobre
`reward_mse` en rollouts autorregresivos, como se hizo.

No verificado: la pérdida de validación no se descompuso en sus dos términos, así que la
lectura de que el término latente explica la ventaja del Transformer en validación se
apoya en el `latent_mse` de test, no en una medición directa sobre validación.

## ✅ Escenario asimétrico, segunda ronda: dataset de 80 episodios, selección por SUMO real y `VecNormalize` — resultado con 3 semillas por método

### 1. Recapitulación

La demanda asimétrica (500/150 veh/h) rompió la regla trivial "pedir siempre la fase
contraria" y los dos PPO dejaron de seguirla, pero aparecieron episodios catastróficos
que no se explicaron del todo (ver "Escenario asimétrico, primera ronda", más abajo).
Esta sección documenta todo lo que se investigó y corrigió después, en orden.

**Estado de los checkpoints** (los `.zip`/`.npz`/`.pkl` no se versionan; los `.json` sí):

- `controller/best_model.zip`: PPO del sueño **oficial**, semilla de entrenamiento 1,
  `VecNormalize` de recompensa. Archivados: `*_seed0_worse.*` (semilla 0), `*_seed2.*`
  (semilla 2), `*_asym40.*` (dataset de 40 episodios), `*_v1_dream7.*` y
  `*_v2_dream20_deprecated.*` (escenario simétrico).
- `controller_direct/best_model.zip`: PPO directo **oficial**, semilla 0, `VecNormalize`
  de recompensa y de observaciones, con sus estadísticas en `best_model_vecnormalize.pkl`.
  Archivados: `*_seed1.*`, `*_seed2.*`, `*_no_obsnorm.*` (solo recompensa normalizada),
  `*_asym40.*`.
- **El resultado de cada método se reporta como la media de sus 3 semillas de
  entrenamiento**, no con el checkpoint oficial solo (punto 7).

### 2. Mejora del dataset

- **Bug encontrado:** `scripts/collect_dataset.py` llamaba a `env.reset()` sin semilla, y
  sumo-rl reutiliza la última semilla cuando no recibe una. Los 40 episodios compartían
  la semilla de SUMO 42: la variedad venía solo de las acciones aleatorias. **Arreglado**
  (commit `2a63328`): el episodio `i` reinicia con la semilla `seed_start + i`, el mismo
  principio que `ReseedingWrapper`. Hay un test con un entorno sustituto que lo comprueba.
  El estado inicial es idéntico con cualquier semilla (en t=0 no hay azar que la semilla
  controle), pero las trayectorias difieren a partir del paso 10.
- **De 40 a 80 episodios** (`DEFAULT_NUM_EPISODES`), divididos en 56/12/12 episodios
  (3360/720/720 transiciones), 0 NaN/Inf.
- **La cola fuera de rango del test desapareció.** Con 40 episodios, el mínimo del test
  (-765.1) quedaba por debajo del de train (-504.1). Ahora train llega a -857.1 y ningún
  reward de validación ni de test queda por debajo de ese mínimo.
- **Recorte de recompensa del Dream Environment recalculado** (commit `6c753d2`):
  **[-326.33, 1.00]** (antes [-266.13, 1.00]).

| | 40 episodios, semilla fija | **80 episodios, semilla por episodio** |
|---|---|---|
| Autoencoder, validación (final / mejor) | 0.021138 / 0.021030 | **0.007646 / 0.007445** |
| LSTM, validación (final / mejor) | 0.157874 / ≈0.1565 | **0.141777 / 0.140486** |
| Error de reward a un paso (h=1), como % del baseline persistente | 23.4% | **2.7%** |
| Ídem en h=10 | 11.9% | **9.2%** |
| Error latente h=1 / h=10 | 0.2486 / 0.6966 | **0.1035 / 0.4954** |

Evaluación del LSTM en test (80 episodios, 420 muestras por horizonte):

```
  h | latent (modelo) | latent (baseline) | reward (modelo) | reward (baseline) | modelo/baseline
  1 |     0.1035      |     0.8797        |      57.05      |     2130.94       |      2.7%
  2 |     0.1573      |     1.6391        |     127.29      |     5997.40       |      2.1%
  3 |     0.1977      |     1.9122        |     204.49      |     8752.99       |      2.3%
  4 |     0.2470      |     1.9055        |     279.21      |     9875.39       |      2.8%
  5 |     0.2992      |     1.8538        |     349.54      |    10027.76       |      3.5%
  6 |     0.3463      |     1.8216        |     409.32      |    10105.19       |      4.1%
  7 |     0.3865      |     1.8547        |     494.95      |    10669.87       |      4.6%
  8 |     0.4033      |     1.8787        |     591.84      |    10991.98       |      5.4%
  9 |     0.4422      |     1.8852        |     757.97      |    10896.91       |      7.0%
 10 |     0.4954      |     1.7985        |     993.29      |    10817.11       |      9.2%
```

**Matiz honesto:** el 2.7% queda incluso por debajo del 5.8% del escenario simétrico,
pero **parte de la mejora viene de un test set distinto**, no solo de un modelo mejor.
Ahora son 12 episodios en vez de 6, con semillas variadas y sin la cola fuera de rango
que penalizaba al modelo anterior. Por eso se comparan proporciones frente al baseline y
no MSE absolutos. El error latente acumulado crece 4.79x de h=1 a h=10 (antes 2.80x),
porque h=1 mejoró mucho más que h=10, no porque h=10 empeorara.

### 3. Experimento 0 repetido con el dataset nuevo

Mismos scripts, sin modificar. Además se corrió `scripts/evaluate_world_model_raw.py`,
sin el cual `compare_experiment_0.py` habría comparado el `z` nuevo con el reporte crudo
del escenario simétrico. LSTM sobre estado crudo: pérdida final de train 0.026895 y de
validación 0.099583 (mejor 0.095438, época 68).

```
  h |  reward_mse (z) |  reward_mse (crudo) | gana
  1 |          57.051 |              63.436 |  z
  2 |         127.288 |             139.760 |  z
  3 |         204.492 |             235.490 |  z
  4 |         279.212 |             334.497 |  z
  5 |         349.542 |             417.238 |  z
  6 |         409.317 |             505.745 |  z
  7 |         494.946 |             622.507 |  z
  8 |         591.838 |             713.016 |  z
  9 |         757.966 |             808.376 |  z
 10 |         993.291 |             937.508 |  crudo
```

**El Autoencoder sigue ganando, en 9/10 horizontes** (en el escenario simétrico original
fueron 10/10). Su ventaja es mayor en horizontes cortos y medios, se estrecha en h=9 y
**se invierte en h=10**, coherente con el mayor error acumulado del modelo en `z`. La
decisión de mantener el Autoencoder se sostiene.

### 4. Hallazgo crítico: el criterio de selección del PPO del sueño estaba roto

- **Síntoma.** El primer PPO del sueño reentrenado con el Autoencoder y el LSTM nuevos
  empeoró en SUMO real: su `best_model`, elegido por reward imaginado (t=26000), dio
  **-536.06 / -547.37** (13/30 episodios catastróficos). El checkpoint final (t=50176) dio
  -432.11 / -430.63. Al mismo tiempo, el reward imaginado *mejoraba* (-119.28 → -86.24).
- **Investigación** (`CheckpointCallback` cada 2000 pasos, 25 checkpoints evaluados en SUMO
  real, seed 3000). **Correlación de Pearson entre reward imaginado y real: +0.077**
  (Spearman -0.062; +0.229 desde t=10000). Es esencialmente nula. El checkpoint con mejor
  reward imaginado (t=38000, -91.67) fue **el peor en SUMO real (-684.91)**; el mejor en
  real (t=6000, -398.10) tenía uno de los peores rewards imaginados. Además, el reward
  real empeoraba con más entrenamiento en el sueño (-448.8 de media entre t=2000 y 12000,
  -535.6 entre t=14000 y 50000) mientras el imaginado mejoraba (-140.4 → -122.3). Es lo
  esperable si el PPO explota errores del World Model; es una tendencia de una sola
  corrida, no una prueba.
- **Fix** (commit `174a40a`): `EvalCallback` evalúa en **SUMO real** a través de
  `EncodedTrafficEnvironment` (Encoder congelado), envuelto en `ReseedingWrapper` con las
  semillas fijas 20000–20004, con `eval_freq=5000` y `n_eval_episodes=5`. El entrenamiento
  sigue sin usar pasos reales, pero **la selección cuesta 3,000 pasos reales por corrida**.
  Resultado, antes de normalizar recompensas: -421.60 / -468.25 (17/30 ganan a tiempo
  fijo, 4/30 catastróficos).
- **Efecto colateral corregido** (commit `7e4a2ae`): `DreamEnvironment` creaba su
  generador aleatorio sin semilla, así que la evaluación imaginada elegía ventanas
  distintas en cada corrida. El mismo modelo obtuvo -86.24 en una corrida y -104.41 en
  otra; en toda la curva, la diferencia media entre corridas fue de 34 puntos (máximo
  234), un ruido mayor que las diferencias entre checkpoints. Se añadió `eval_seed`
  opcional (por defecto `None`, sin cambio de comportamiento), con su test.

### 5. Hallazgo crítico: la función de valor de PPO no aprendía

- **Diagnóstico.** `explained_variance` ≈ 0 en **todas** las corridas del proyecto hasta
  este punto, en los dos PPO. Los retornos no estaban normalizados y eran de cientos a
  miles (en el dataset, el retorno por episodio tiene desviación estándar 1488.56 y el
  retorno descontado 1046.06), y `value_loss` era del orden de la varianza real de los
  retornos: lo esperable si la red de valor predice aproximadamente la media.
- **Fix** (commit `782e52c`): `VecNormalize(norm_obs=False, norm_reward=True,
  clip_reward=10.0, gamma=config.gamma)` en los dos entrenamientos, con
  `ControllerConfig.normalize_reward` y `reward_clip` registrados en el `.json` de cada
  checkpoint. Detalles técnicos:
  - `EvalCallback` llama a `sync_envs_normalization` antes de cada evaluación, y esa
    función **lanza un `AssertionError` si el entorno de evaluación no está envuelto
    igual** (también en `VecNormalize`). Por eso el entorno de evaluación usa
    `VecNormalize(training=False, norm_reward=False)`: estadísticas congeladas y reward
    real, comparable con todas las tablas anteriores.
  - `Monitor` va **debajo** de `VecNormalize`, para que los registros de episodio guarden
    el reward real. PPO solo lo añade por sí mismo cuando no recibe un `VecEnv`.
  - Un test arma la pila completa con modelos sintéticos y cruza evaluaciones. Se
    comprobó aparte que con el entorno de evaluación mal envuelto `learn()` falla.
- **Resultado en el PPO del sueño:** `explained_variance` sube a **~0.95** (media por
  cuartos 0.818 / 0.941 / 0.952 / 0.955). Con la semilla 0: -351.61 / -339.23, 25/30
  episodios ganan a tiempo fijo, 3/30 catastróficos. **Es la primera vez con el dataset
  de 80 episodios que supera a tiempo fijo en las dos semillas de evaluación**, y el
  punto 7 confirma que lo hace en las tres semillas de entrenamiento.
- **Resultado en el PPO directo (solo recompensa normalizada, semilla 0):** mejora
  parcial y muy inestable de `explained_variance` (media por cuartos -0.978 / -0.050 /
  0.046 / 0.151; máximo 0.584), y **empeoró en SUMO real**: -436.25 / -349.79, frente a
  -381.01 / -328.02 sin normalizar, con 7/30 episodios catastróficos (el peor, -1184.9).
  Eso llevó a la siguiente investigación.

### 6. Hallazgo: el PPO directo necesitaba también normalizar observaciones

- **Diagnóstico.** `TrafficEnvironment` le entrega al PPO directo el estado crudo de 26
  dimensiones, sin `scaler.pkl` (confirmado en el código). La desviación estándar por
  dimensión va de 0.021 (ocupaciones) a 37.88 (tiempo de espera del carril Sur): **un
  rango de ~1764x**, frente a ~3x en `z` (0.65–2.12). Además, una política mala genera
  estados muy fuera del rango del dataset. Tras 30 pasos pidiendo siempre la fase 1, los
  tiempos de espera por carril llegan a **1508 y 1515**, frente a un máximo de **467** en
  los datos de entrenamiento (~3.2 veces más).
- **Fix** (commit `c074b9e`): `normalize_obs=True` **solo para el PPO directo**. El del
  sueño sigue con `norm_obs=False`, porque `z` ya tiene escala unitaria. Detalles:
  - Con `norm_obs=True`, la política depende de las estadísticas de normalización del
    momento en que se guardó, y esas estadísticas siguen cambiando después.
    **`SaveVecNormalizeOnBest`** (`callback_on_new_best`) guarda una copia
    (`best_model_vecnormalize.pkl`) en el instante exacto en que se guarda el mejor
    modelo: son las mismas que `EvalCallback` sincronizó para la evaluación que lo eligió.
  - `load_obs_normalizer()` lee `normalize_obs` del `.json` del checkpoint. Si es falso,
    devuelve la observación sin cambios (así funcionan todos los checkpoints anteriores);
    si es verdadero, carga el `.pkl` con `VecNormalize.load(...)`, `training=False`, y
    aplica `normalize_obs()` a cada observación antes de `model.predict()`. **Si falta el
    `.pkl`, falla** en vez de evaluar sin normalizar.
  - Se actualizaron `evaluate_final_comparison.py` (tabla y comparación contrafactual) y
    `evaluate_direct_vs_dream.py`. `evaluate_controller_sumo.py` no evalúa el PPO directo.
  - 2 tests nuevos.
- **Resultado** (semilla 0 fija, única variable cambiada frente a la corrida anterior):
  -351.68 / -349.26 (media de 30 episodios **-350.47**, antes -393.02), episodios
  catastróficos **3/30** (antes 7/30), peor episodio -716.8 (antes -1184.9), desviación
  reducida a menos de la mitad en seed 3000, y throughput 13.00 / 12.93 (antes 11.53 /
  11.40, casi al nivel de tiempo fijo). La curva de `EvalCallback` se estabiliza en ~-360
  a partir del timestep 6000. **`explained_variance` no mejoró** (media del último
  cuarto 0.106, máximo 0.809). El punto 7 muestra además que **la semilla 0 fue la mejor
  de las tres**.

### 7. Verificación de robustez: 3 semillas de entrenamiento por método

Una sola corrida no basta para confiar en una mejora. Cada PPO se reentrenó con las
semillas de entrenamiento 0, 1 y 2, y cada checkpoint se evaluó en las dos semillas de
evaluación (15 episodios cada una).

**PPO del sueño (`VecNormalize` de recompensa):**

```
semilla | seed 3000          | seed 5000          | media 30 | mediana 30 | gana t.fijo | < -600 | peor   | best en t=
      0 | -351.61 +/- 194.53 | -339.23 +/- 167.01 |  -345.42 |    -294.55 |    25/30    |  3/30  | -973.1 | 15000
      1 | -291.26 +/-  56.86 | -364.99 +/- 189.84 |  -328.13 |    -302.60 |    25/30    |  1/30  | -998.7 | 45000
      2 | -289.04 +/-  70.18 | -324.59 +/- 124.71 |  -306.82 |    -290.90 |    25/30    |  1/30  | -685.1 | 15000
Media 90 episodios: -326.79 (std 147.42, mediana -293.00, error estándar 15.63)
```

Las tres semillas superan a tiempo fijo, con medias parecidas (dispersión entre semillas
de 39 puntos) y `explained_variance` ~0.95 en las tres. **Sobre el checkpoint oficial:**
la semilla 1 se adoptó mirando solo la seed 3000, donde parecía la mejor (-291.26). Con
las dos semillas de evaluación, la mejor en promedio es la semilla 2 (-306.82), y la 1 es
la peor en seed 5000 (-364.99, con un episodio en -998.7). Las diferencias entre semillas
están dentro del ruido. Se decidió mantener la semilla 1 como checkpoint oficial y
**reportar como resultado del método la media de las tres semillas en 90 episodios
(-326.79)**; el desglose por semilla es la evidencia de consistencia, no el número
principal.

**PPO directo (`VecNormalize` de recompensa y observaciones):**

```
semilla | seed 3000          | seed 5000          | media 30 | mediana 30 | gana t.fijo | < -600 | peor    | best en t=
      0 | -351.68 +/- 146.55 | -349.26 +/- 110.13 |  -350.47 |    -329.45 |    22/30    |  3/30  |  -716.8 | 9000
      1 | -500.61 +/- 306.10 | -439.08 +/- 250.12 |  -469.85 |    -300.05 |    18/30    | 10/30  | -1023.7 | 6000
      2 | -527.25 +/- 145.69 | -554.54 +/- 180.24 |  -540.89 |    -543.95 |     9/30    | 10/30  |  -918.5 | 3000
Media 90 episodios: -453.74 (std 217.13, mediana -381.75, error estándar 23.02)
```

**El resultado del RL directo varía mucho entre semillas** (dispersión de 190 puntos):

- **La semilla 2 colapsó a la regla trivial "pedir siempre la fase contraria"**, el mismo
  patrón del escenario simétrico. Coincide con la regla a menos de 1 punto en 25 de 30
  episodios (con diferencias de 0.1–0.4, la penalización de `phase_change`), y los otros
  5 son peores que la regla. Su curva de evaluación se queda plana en -576 durante cinco
  evaluaciones y termina en -832.
- **La semilla 1 es bimodal:** tiene buenos episodios (~-200/-300) y 10 catastróficos,
  incluido el peor episodio medido en el proyecto (-1023.7).
- `explained_variance` sigue sin aprender en ninguna semilla (media del último cuarto:
  0.106, -0.04 y 0.21; máximos puntuales 0.81–0.85).

El checkpoint oficial del directo (semilla 0) es la mejor de sus tres semillas. Por eso,
igual que con el sueño, el resultado del método es la media de las tres (-453.74), no la
semilla 0 sola.

### 8. Resultado final de la comparación

SUMO real, `scripts/evaluate_final_comparison.py` y el mismo protocolo. Los PPO se
promedian sobre 3 semillas × 2 semillas de evaluación × 15 episodios; tiempo fijo y la
regla son deterministas: 2 semillas de evaluación × 15 episodios.

| Política | Episodios | Media | Desv. estándar | Mediana | Espera media | Ganan a tiempo fijo | < -600 | Peor | Pasos reales de SUMO (3 semillas) |
|---|---|---|---|---|---|---|---|---|---|
| **PPO del sueño (World Model)** | 90 | **-326.79** | 147.42 | -293.00 | 4.43 | **75/90** | **5/90** | -998.7 | **13,800** en total: 4,800 de dataset (una vez) + 3 × 3,000 de selección → **~4,600 por semilla** |
| PPO directo (RL directo) | 90 | -453.74 | 217.13 | -381.75 | 6.15 | 49/90 | 23/90 | -1023.7 | 39,000 en total: 3 × (10,000 de entrenamiento + 3,000 de evaluación) → 13,000 por semilla |
| Tiempo fijo (ciclo=5) | 30 | -411.27 | 51.75 | -399.70 | 5.58 | — | 0/30 | -592.2 | — |
| Regla "pedir fase contraria" | 30 | -502.53 | 118.11 | -511.10 | 6.68 | 9/30 | 7/30 | -703.1 | — |

La espera media es el promedio de las 6 evaluaciones de cada PPO y de las 2 de tiempo
fijo y la regla. **Contabilidad de interacciones:** el dataset del World Model (80
episodios × 60 pasos = 4,800 transiciones) se recolecta una sola vez y lo reutilizan las
3 semillas del sueño, así que se cuenta una vez; en el RL directo, cada semilla entrena
desde cero en SUMO sin nada compartido, así que su costo se repite por semilla. Con una
sola semilla, el World Model costaría 7,800 pasos (4,800 + 3,000) frente a 13,000: la
ventaja en interacciones crece con el número de entrenamientos que reutilizan el mismo
dataset.

**Cuánta evidencia hay de que la ventaja del World Model es real (con el presupuesto
de 10,000 pasos del RL directo):**

- **A nivel de episodio** (90 frente a 90, tratados como independientes): la diferencia
  de 126.95 puntos equivale a **4.56 errores estándar** (Welch t = 4.56, p ≈ 1e-5;
  Mann-Whitney p ≈ 1.4e-5). No es azar entre episodios.
- **A nivel de semilla de entrenamiento** (3 medias frente a 3 medias): **Welch t = 2.24,
  gl ≈ 2.2, p = 0.145, no significativo.** Los 15 episodios de una semilla no son
  independientes entre sí, y el RL directo varía mucho entre semillas (desviación de las
  medias por semilla: 96.2, frente a 19.3 en el sueño). Toda semilla del sueño supera a
  toda semilla del directo, pero por muy poco en el peor caso (la peor del sueño, -345.42,
  frente a la mejor del directo, -350.47).
- **Lectura honesta:** con el presupuesto usado, el método World Model obtiene mejor
  control medio, es **mucho más consistente entre semillas** y tiene muchos menos
  episodios catastróficos, con **~35% de las interacciones reales por semilla** (4,600
  frente a 13,000, con el dataset compartido entre las 3 semillas). Con solo 3
  semillas por método, la ventaja en la media no se puede afirmar con significancia
  estadística a nivel de semilla. La diferencia más robusta es la **consistencia**: el
  RL directo puede salir tan bien como el sueño (semilla 0) o colapsar a la regla trivial
  (semilla 2). **Precisión posterior:** todo esto vale para este presupuesto (13,000
  interacciones por semilla). Con 3x presupuesto, el RL directo alcanza un desempeño
  comparable (-335.24), su desviación entre semillas baja a 30.6 y la semilla 2 deja de
  colapsar, a cambio de ~8.5 veces las interacciones del World Model (sección
  "Verificación: RL directo con 3x presupuesto").
- **Throughput:** el PPO del sueño sigue algo por debajo de tiempo fijo (12.20–12.33
  frente a 13.27–13.73 con el checkpoint oficial), la misma salvedad de todo el proyecto.

### 9. Limitaciones que quedan abiertas

- **Los episodios catastróficos se redujeron, pero no desaparecieron** en ninguno de los
  dos métodos (5/90 en el sueño, 23/90 en el directo). Su causa completa sigue sin
  identificarse; ver la primera ronda para lo que ya se descartó.
- **La función de valor del PPO directo sigue sin aprender** incluso con las dos
  normalizaciones: la media del último cuarto de `explained_variance` está entre -0.04 y
  0.21 según la semilla, con máximos puntuales de 0.81–0.85. Causa no identificada;
  hipótesis sin confirmar: muchas menos actualizaciones de gradiente que el PPO del sueño
  (39 frente a 195), y más variabilidad de escenarios en SUMO real que en el sueño.
- **Presupuesto del RL directo:** con solo 10,000 pasos reales de entrenamiento no se
  puede descartar que más presupuesto hubiera cerrado la brecha o estabilizado sus
  semillas. La comparación vale **para el presupuesto usado**; no es una afirmación
  general de que el RL directo sea inferior en cualquier condición. **Verificado
  después:** con 30,000 pasos el RL directo alcanza un desempeño comparable, con ~8.5
  veces las interacciones del World Model (sección "Verificación: RL directo con 3x
  presupuesto").
- **Solo 3 semillas por método:** suficiente para ver la diferencia de consistencia, no
  para afirmar significancia a nivel de semilla (punto 8).
- **Límite de 8 s en la fase 1:** medido con los PPO de la primera ronda; no se volvió a
  medir con los checkpoints actuales.

**Notas técnicas menores:** la nota al pie de `compare_experiment_0.py` dice "8 vs. 26
dimensiones", pero el espacio latente tiene 16 (corregido después: la nota y el docstring
del script ya dicen 16). La primera
propuesta de conteos de tests para los 8 commits de este bloque tenía un error (commit 2
= 40, no 41); cada commit se verificó sobre su propio árbol en un `git worktree`.

## 🗄️ Escenario asimétrico, primera ronda (40 episodios, semilla de SUMO fija): la regla trivial se rompe, aparecen episodios catastróficos

> **Registro histórico, superado por la sección anterior.** Los checkpoints, el dataset y
> los números de esta primera ronda fueron reemplazados. Se conserva porque documenta
> decisiones que siguen vigentes (la elección de la demanda 500/150, descartando
> 700/150; la verificación de inserción de vehículos) y la investigación de los
> episodios catastróficos (cola invisible y rachas de fase 1 descartadas). Sus
> conclusiones sobre la función de valor y el trabajo futuro se resolvieron después.

> **Hallazgos de proceso anotados sin corregir** (ver también "🟡 Pendiente"):
> 1. `scripts/collect_dataset.py` llama a `env.reset()` sin semilla, así que los 40
>    episodios del dataset comparten la misma semilla de SUMO (42). La variedad del
>    dataset viene solo de las acciones aleatorias, no de distintas realizaciones de
>    tráfico. Pendiente decidir si esto limita al World Model.
> 2. En los 60 episodios evaluados (punto 5), ninguno de los dos PPO sostiene el verde
>    de la fase 1 (Este/Oeste) más de **8 s**.

### 1. Motivación y diseño

En el escenario simétrico (sección siguiente), los dos PPO convergieron a la misma regla
trivial y el escenario no permitía comparar calidad de control. Se cambió la demanda a
una vía principal Norte/Sur cargada y una secundaria Este/Oeste ligera, donde cambiar de
fase lo antes posible ya no debería ser lo mejor. La demanda simétrica original se
conserva en `single-intersection_symmetric_backup.rou.xml`.

**Proceso de verificación de la demanda (Fase 0)**, con una primera propuesta descartada:

- **Primera propuesta, 700/150 veh/h por brazo (descartada).** Los `vehicle_counts` en
  carril daban una razón de 12.32:1, fuera del rango esperado de 3:1 a 6:1. El conteo de
  vehículos insertados mostró la causa: la entrada real era 3.53:1, pero el carril Sur
  solo aceptó 47 de los ~58 vehículos programados. La vía principal (un carril por
  brazo) quedaba saturada y los vehículos esperaban **fuera de la red**, donde no
  aparecen ni en el estado ni en la recompensa: una cola invisible que una política
  podría explotar.
- **Demanda elegida, 500/150 veh/h por brazo**: Norte/Sur 286 recto + 129 izquierda +
  85 derecha; Este/Oeste 85 + 40 + 25. Razón configurada 3.33:1. Los criterios se
  redefinieron según lo que se puede medir de verdad:
  - Razón de entrada real **2.87:1** en episodios de 300 s. Es menor que la configurada
    por el redondeo: cada flujo inserta su primer vehículo en t=0, lo que pesa más en
    los flujos pequeños. Aceptado como límite estructural de episodios cortos.
  - **Insertados = programados** en los 12 flujos, con la política de recolección
    (acciones aleatorias) y con la regla "fase contraria": ningún vehículo queda fuera.
    Pendientes de inserción: como máximo 2 vehículos durante ≤3 s. Se deben al arranque
    en t=1–2 s, cuando los 3 flujos de cada brazo comparten un solo carril, y a
    coincidencias puntuales entre flujos. **La demanda simétrica original tenía el mismo
    patrón con más frecuencia** (11 y 14 segundos con pendientes, frente a 5 y 9 ahora).
  - Solo la política de referencia deliberadamente mala "siempre fase 1" deja vehículos
    fuera (hasta 36 pendientes). Esto confirma que la **fase 0 da verde a Norte/Sur** y
    la **fase 1 a Este/Oeste**.

### 2. Pipeline completo re-ejecutado (desde cero, sin commits intermedios)

**Cambio de checkpoints, importante para leer las secciones anteriores.** Desde este
bloque, `models/checkpoints/controller/best_model.zip` y
`models/checkpoints/controller_direct/best_model.zip` contienen los modelos del
**escenario asimétrico**. El v1 del escenario simétrico se conserva como
`controller/best_model_v1_dream7.zip` (y v2 como `_v2_dream20_deprecated`). El
checkpoint del RL directo simétrico se borró: sus números siguen documentados en la
sección siguiente, y su `.json` queda en el historial de git. Las secciones anteriores
que dicen "checkpoint oficial" se refieren al escenario simétrico.

**Dataset** (40 episodios, 28/6/6 por episodio, 0 NaN/Inf, acciones balanceadas):

| split | reward medio | std | min | max |
|---|---|---|---|---|
| train | -43.01 | 56.78 | -504.1 | 2.0 |
| validation | -34.23 | 43.71 | -309.1 | 1.0 |
| test | -66.74 | 102.87 | -765.1 | 2.0 |

La asimetría llega a los datos: `vehicle_counts` medio de Sur 8.18, Norte 4.81, Oeste
0.97 y Este 0.93. Sur carga casi el doble que Norte con la misma demanda (no
investigado). El split de test tiene una cola más pesada que train: su mínimo, -765.1,
queda fuera del rango de entrenamiento.

**Autoencoder**: pérdida final de train 0.016387 y de validación 0.021138 (mejor
0.021030). Comparable al escenario simétrico (MSE de 0.0186 sobre train).

**Recorte de recompensa del Dream Environment, recalculado** (commit `6c753d2`): los
percentiles 1 y 99 del nuevo `train_latent.npz` son **[-266.13, 1.00]**, frente a
[-165.05, 1.00] antes. Con los límites viejos se habría recortado el 4.8% de los rewards
reales de entrenamiento; con los nuevos, el 1.2%.

**LSTM**: pérdida final de train 0.069640 y de validación 0.157874 (mejor ≈0.1565 en la
época 90); razón validación/train ~2.3x. Evaluación en test frente al baseline
persistente, con el escenario simétrico como referencia:

```
  h | latent (modelo) | latent (baseline) | reward (modelo) | reward (baseline) || simetrico: latent modelo | reward modelo | reward baseline
  1 |      0.2486     |      0.9475       |      756.3      |      3233.8       ||      0.1943    |     41.9      |     716.6
  2 |      0.3868     |      1.9580       |     1622.1      |      9817.6       ||      0.2625    |     67.5      |    1203.9
  3 |      0.4874     |      2.5791       |     2535.9      |     16186.3       ||      0.3154    |     86.0      |    1359.3
  4 |      0.5399     |      2.8341       |     2762.2      |     20843.4       ||      0.3724    |    111.7      |    1379.2
  5 |      0.5646     |      2.9350       |     2493.4      |     24089.5       ||      0.4261    |    144.8      |    1279.8
  6 |      0.6066     |      3.0184       |     2364.4      |     25927.3       ||      0.4553    |    187.3      |    1069.3
  7 |      0.6469     |      3.1227       |     2518.6      |     27267.6       ||      0.4729    |    200.5      |    1065.5
  8 |      0.6887     |      3.1174       |     2736.1      |     27475.3       ||      0.4715    |    198.3      |    1157.7
  9 |      0.7221     |      3.0030       |     2866.6      |     26181.6       ||      0.4729    |    181.3      |    1300.9
 10 |      0.6966     |      2.9775       |     2885.7      |     24190.0       ||      0.4742    |    167.0      |    1303.2
```

El modelo supera al baseline en todos los horizontes. El error latente acumulado es algo
mayor (crece 2.80x de h=1 a h=10, frente a 2.44x antes). **La predicción de reward a un
paso es relativamente peor**: 23% del error del baseline, frente al 6% del escenario
simétrico. En h=10 la proporción es similar (12% frente a 13%).

**PPO del sueño** (`dream_max_steps=7`, 50,176 timesteps; reward imaginado, 20 episodios
de validación por evaluación; `*` = `best_model.zip`):

```
 1000 -265.93   2000 -156.18   3000 -162.73   4000 -155.14   5000 -143.39   6000 -153.34   7000 -137.99   8000 -132.61
 9000 -144.77  10000 -140.01  11000 -144.03  12000 -152.58  13000 -173.60  14000 -160.65  15000 -160.71  16000 -136.84
17000 -122.21  18000 -162.12  19000 -149.10  20000 -136.24  21000 -149.57  22000 -119.28* 23000 -148.65  24000 -175.37
25000 -158.04  26000 -132.42  27000 -142.79  28000 -150.63  29000 -137.49  30000 -160.72  31000 -139.37  32000 -136.61
33000 -162.88  34000 -145.18  35000 -133.46  36000 -154.68  37000 -138.27  38000 -144.98  39000 -122.26  40000 -144.84
41000 -119.51  42000 -125.91  43000 -144.58  44000 -131.89  45000 -135.31  46000 -148.30  47000 -148.61  48000 -141.42
49000 -147.75  50000 -136.99
```

Curva ruidosa y no monótona, como en el escenario simétrico. `explained_variance` entre
-0.0006 y 0.0017 durante todo el entrenamiento.

**PPO de RL directo** (10,000 timesteps reales; semillas verificadas: 171 de
entrenamiento únicas y 5 de evaluación que ciclan; `*` = `best_model.zip`):

```
 1000 -23143.46 +/- 10925.07     6000 -284.00 +/-  75.62 *
 2000  -2105.66 +/-   603.68     7000 -301.76 +/-  52.45
 3000   -362.88 +/-    74.07     8000 -310.24 +/-  65.79
 4000   -311.30 +/-    50.81     9000 -309.84 +/- 115.45
 5000   -477.68 +/-   300.22    10000 -299.76 +/-  51.07
```

A diferencia del escenario simétrico, la curva no se congela: mejora hasta el timestep
6000 y después fluctúa. **`explained_variance` se mantuvo entre -0.0016 y 0.00014 durante
todo el entrenamiento**, igual que antes: la función de valor no aprende (ver punto 5).

### 3. Resultado principal: la regla trivial se rompe

`scripts/evaluate_final_comparison.py` (commit `f58347e`), 15 episodios por semilla base:

```
seed_base=3000                 |               reward |      espera_prom |      cola_prom |       throughput
PPO (sueno)                    |   -357.89 +/-  202.71 |    4.87 +/-  2.81 |   1.26 +/- 0.58 |   11.73 +/-  1.73
    [-180.6, -431.2, -918.8, -211.6, -241.6, -332.5, -316.5, -253.8, -272.8, -524.8, -171.5, -398.0, -221.7, -682.2, -210.8]
PPO (RL directo)               |   -381.01 +/-  275.58 |    5.25 +/-  3.85 |   1.27 +/- 0.75 |   11.00 +/-  1.71
    [-148.1, -928.0, -192.9, -396.2, -509.2, -872.0, -248.0, -195.0, -213.9, -801.9, -143.0, -573.1, -166.9, -188.0, -139.0]
Tiempo fijo (ciclo=5)          |   -397.07 +/-   57.40 |    5.39 +/-  0.76 |   1.44 +/- 0.18 |   13.73 +/-  2.95
    [-360.2, -383.2, -364.2, -369.2, -421.2, -592.2, -397.2, -427.2, -372.2, -343.2, -362.2, -404.2, -369.2, -374.2, -416.2]
Regla: pedir fase contraria    |   -509.57 +/-  122.55 |    6.76 +/-  1.62 |   1.88 +/- 0.43 |   11.93 +/-  3.13
    [-559.1, -416.1, -700.1, -563.1, -400.1, -418.1, -386.1, -435.1, -356.1, -675.1, -632.1, -332.1, -696.1, -536.1, -538.1]

seed_base=5000                 |               reward |      espera_prom |      cola_prom |       throughput
PPO (sueno)                    |   -411.83 +/-  214.76 |    5.58 +/-  2.94 |   1.46 +/- 0.66 |   12.60 +/-  2.36
    [-632.2, -220.6, -378.9, -359.7, -823.0, -242.8, -676.1, -197.6, -269.0, -273.9, -164.6, -369.8, -700.1, -661.6, -207.6]
PPO (RL directo)               |   -328.02 +/-  178.20 |    4.51 +/-  2.43 |   1.14 +/- 0.53 |   12.00 +/-  2.85
    [-726.2, -707.0, -198.0, -343.9, -134.0, -222.0, -300.1, -225.0, -359.0, -313.0, -192.0, -205.0, -525.0, -289.0, -181.1]
Tiempo fijo (ciclo=5)          |   -425.47 +/-   40.71 |    5.77 +/-  0.55 |   1.52 +/- 0.13 |   13.27 +/-  2.46
    [-402.2, -425.2, -393.2, -446.2, -413.2, -537.2, -362.2, -458.2, -423.2, -456.2, -397.2, -458.2, -387.2, -426.2, -396.2]
Regla: pedir fase contraria    |   -495.50 +/-  113.06 |    6.59 +/-  1.48 |   1.84 +/- 0.39 |   13.13 +/-  2.39
    [-531.1, -660.1, -421.1, -327.1, -391.1, -608.1, -328.1, -580.1, -466.1, -703.1, -549.1, -463.1, -491.1, -357.1, -556.1]
```

- **La regla "pedir siempre la fase contraria" pasa de ser la mejor política (escenario
  simétrico) a ser la PEOR** (-509.57 y -495.50), peor incluso que tiempo fijo.
- **Tiempo fijo mejora frente a la regla**: su ciclo no es simétrico (63% del tiempo en
  fase 0, la de Norte/Sur), lo que por casualidad favorece a la vía principal.
- **Los PPO ya no siguen la regla.** Comparación contrafactual sobre los mismos estados
  reales (el PPO del sueño conduce SUMO en las 30 semillas; 1800 pasos, 666 bloqueados
  por `min_green`, 37.0%):

  ```
  Par                  | acuerdo total | desacuerdos | en pasos bloqueados | acuerdo en pasos NO bloqueados
  sueno / directo      |     81.6%     |     332     |    182 (54.8%)      |   86.8%  (150 de 1134)
  sueno / regla        |     61.1%     |     700     |    189 (27.0%)      |   54.9%  (511 de 1134)
  directo / regla      |     47.7%     |     942     |    343 (36.4%)      |   47.2%  (599 de 1134)
  ```

  En el escenario simétrico el acuerdo con la regla en pasos no bloqueados era del 100% y
  el 99.2%; ahora es del 54.9% y el 47.2%. Los dos PPO coinciden entre sí en el 86.8% de
  esas decisiones: aprendieron comportamientos parecidos por caminos distintos.

### 4. Hallazgo nuevo: mejores en promedio, pero con episodios catastróficos

| | Episodios que ganan a tiempo fijo (misma semilla) | Episodios peores que -600 | Peor episodio |
|---|---|---|---|
| PPO sueño | 11/15 + 10/15 = **21/30** | **7/30** | -918.8 |
| PPO directo | 9/15 + 12/15 = **21/30** | **5/30** | -928.0 |
| Tiempo fijo | — | **0/30** | -592.2 |

En sus buenos episodios, los PPO llegan a -130 / -250, muy por encima de tiempo fijo; en
los malos caen a -680 / -930. Tiempo fijo es entre 4 y 5 veces más estable (desviación
de 41–57 frente a 178–276). El throughput de los PPO es menor (11.0–12.6 frente a
13.3–13.7), la misma salvedad que en el escenario simétrico. Los episodios catastróficos
dependen de la política, no solo del tráfico: el PPO del sueño falla en 3002, 3013, 5000,
5004, 5006, 5012 y 5013, y el directo en 3001, 3005, 3009, 5000 y 5001. Solo comparten la
semilla 5000.

### 5. Investigación de los episodios catastróficos

Cada PPO conduce su propia trayectoria real, que reproduce exactamente la tabla del
punto 3. Se compararon los 12 episodios catastróficos con los 5 mejores de cada semilla
(10 por política), midiendo por segundo simulado.

- **Cola invisible (vehículos pendientes de inserción): DESCARTADA.** El máximo es de 8
  pendientes en los 60 episodios, catastróficos y buenos, en las dos políticas. Ese
  máximo es el efecto de arranque en t=1 s. El total es de 14–24 vehículo·s por episodio
  en ambos grupos, y su correlación con el reward en los 30 episodios es -0.09 (sueño) y
  -0.13 (directo).
- **Racha de fase 1 sostenida: DESCARTADA como causa.** Las dos políticas tienen una
  racha máxima de verde en fase 1 de **exactamente 8 s en los 60 episodios**, así que no
  distingue episodios catastróficos de buenos. Sobre el origen de esos 8 s: no es un
  tope que imponga el entorno, porque la política "siempre fase 1" de la Fase 0 sostuvo
  esa fase todo el episodio. Coincide con la duración mínima posible del verde tras un
  cambio: 3 s de verde en el paso del cambio (5 s menos 2 de amarillo) más un paso
  completo de 5 s bloqueado por `min_green`. Eso sugiere que **ambos PPO abandonan la
  fase 1 en cuanto se lo permiten**. No se investigó a fondo.
- **Fracción de tiempo en fase 1: señal PARCIAL, solo en el PPO del sueño.**
  Catastróficos 29.9% frente a 25.8% en los buenos; correlación con el reward en los 30
  episodios de -0.59. En el PPO directo no hay señal: 23.3% frente a 23.1%, correlación
  -0.05.
- **Escala de la función de valor (`explained_variance` ≈ 0 en AMBOS entrenamientos):
  confirmada en magnitud, NO en causalidad.** Los retornos no están normalizados y son
  de cientos a miles: en el dataset, el retorno por episodio tiene desviación estándar
  1488.56 y el retorno descontado (γ=0.99) 1046.06. En los dos entrenamientos,
  `value_loss` es del orden de la varianza real de los retornos:
  - RL directo: `value_loss` ~1e4–4e4 al final, con varianza de sus retornos de 1.9e4
    (desviación estándar 139.09).
  - Sueño: `value_loss` entre 5,220 y 45,000 (mediana 10,100), con varianza de los
    retornos imaginados de 5.2e3 (desviación estándar 72.27).

  Esto es lo esperable si la red de valor predice aproximadamente la media. No se
  implementó `VecNormalize` ni otra normalización para comprobar si eso arregla la
  función de valor o elimina los episodios catastróficos.

**Conclusión honesta: la causa completa de los episodios catastróficos NO se
identificó.** Queda documentada como limitación abierta, no como resuelta.

### 6. Trabajo futuro identificado, en orden de prioridad

1. **Normalizar recompensas o retornos (por ejemplo `VecNormalize`) en los dos
   entrenamientos de PPO**, y volver a evaluar si `explained_variance` sube y si
   desaparecen los episodios catastróficos. Es el experimento más prometedor y mejor
   fundamentado de los que quedaron pendientes.
2. **Si (1) no lo explica todo: investigar el momento de los cambios de fase** respecto a
   las colas de cada brazo (cola del brazo que pasa a verde frente a la del que pasa a
   rojo en cada cambio), comparando episodios catastróficos con buenos.

### 7. Nota sobre el costo de interacciones reales (sin conclusión firme)

El RL directo llegó a su mejor evaluación (-284.00, semillas 20000–20004) en **6,000
pasos reales de entrenamiento más 1,800 de evaluación**. El método World Model usó
**2,400 transiciones reales** para el dataset. Es el mismo orden de magnitud que en el
escenario simétrico. **Esta comparación es prematura mientras los episodios catastróficos
no se entiendan**: podrían estar distorsionando cualquiera de los dos promedios, y
ninguno de los dos métodos produce todavía una política estable.

## ✅ Baseline de RL directo y hallazgo final: ambos métodos convergen a la misma regla, sin ahorro de interacciones demostrable en este escenario

### 1. Qué se construyó

- **`environments/reseeding_wrapper.py`** (`ReseedingWrapper`). sumo-rl solo cambia la
  semilla de tráfico cuando `reset()` la recibe explícitamente; si no, reutiliza la
  última (`SumoEnvironment.reset`). Stable-Baselines3 llama a `reset()` sin semilla en
  cada frontera de episodio, así que sin el wrapper el PPO directo habría entrenado y
  se habría evaluado siempre sobre la misma realización de tráfico. El wrapper asigna
  una semilla nueva en cada episodio de entrenamiento (10000, 10001, …) y un conjunto
  fijo que cicla en `EvalCallback` (20000–20004); una semilla explícita siempre gana.
  Verificado con un registro de cada `reset()` real:
  - Entrenamiento: 171 reinicios, 171 semillas únicas. La primera es 0 (la pasa SB3
    explícitamente por `PPO(seed=0)`), después 10000–10169 consecutivas.
  - `EvalCallback`: 60 reinicios. Cada una de las 10 evaluaciones cubre exactamente una
    vez las 5 semillas, rotadas una posición porque `DummyVecEnv` reinicia
    automáticamente tras el quinto episodio y consume una semilla que se descarta. Esto
    solo funciona porque `n_eval_episodes` (5) coincide con la longitud del ciclo.
  - 3 tests nuevos (`tests/test_reseeding_wrapper.py`), sin SUMO.
- **`training/train_controller_direct.py`**: PPO (`MlpPolicy`, mismos hiperparámetros de
  `ControllerConfig`) entrenado directamente sobre el estado crudo de 26 dimensiones de
  `TrafficEnvironment`, sin Autoencoder ni Dream Environment. Presupuesto deliberadamente
  modesto: 10,000 timesteps (SB3 completa hasta 10,240, múltiplo de `n_steps`), más
  10 evaluaciones × 5 episodios × 60 pasos = 3,000 pasos de evaluación. Duró 262 s. Su
  `.json` de hiperparámetros registra `total_timesteps=10000`; `dream_max_steps` aparece
  porque es un campo de `ControllerConfig`, pero no tiene efecto en esta corrida.
- **`scripts/evaluate_direct_vs_dream.py`**: v1, PPO directo y tiempo fijo contra SUMO
  real, mismo protocolo de siempre (15 episodios, `seed_base=3000` y `5000`, semillas
  distintas de las de entrenamiento y `EvalCallback`).

### 2. Primera señal de alarma

- **PPO directo y v1 dan un reward casi idéntico episodio por episodio**: 22 de 30
  episodios difieren en menos de 1 punto (por ejemplo -294.0/-293.7, -257.0/-257.0,
  -261.1/-261.1), a pesar de haber aprendido en espacios completamente distintos (`z`
  imaginado frente a estado crudo real).
- **`explained_variance` se mantuvo entre -0.002 y 0.0003 durante todo el
  entrenamiento directo**: la función de valor no aprendió nada.
- **La curva de `EvalCallback` se estanca desde el timestep 2000**:

  ```
  timestep | reward (5 episodios, semillas 20000-20004)
     1000  | -549.12 +/- 287.15
     2000  | -285.20 +/-  47.56
     3000  | -285.06 +/-  35.52
     4000  | -285.14 +/-  35.51
     5000  | -285.18 +/-  35.50
     6000  | -285.02 +/-  35.59
     7000  | -285.02 +/-  35.55
     8000  | -286.08 +/-  35.13
     9000  | -284.34 +/-  35.60
    10000  | -284.10 +/-  35.58   <- best_model.zip
  ```

  Después del timestep 2000 las mejoras son de décimas. El reward del rollout
  (política estocástica, media de los últimos 100 episodios) baja de -2020 a -330.

### 3. Investigación: dos pruebas

**(a) Reglas triviales, evaluadas de forma independiente** (su propio
`TrafficEnvironment`, mismo `run_policy` del script de evaluación):

```
                                          |             reward |    espera_prom |    cola_prom |   throughput
"Pedir siempre la fase contraria", 3000   |  -287.83 +/-  23.87 |  3.82 +/- 0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
"Pedir siempre la fase contraria", 5000   |  -293.43 +/-  41.28 |  3.90 +/- 0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
"Alternar 0/1 cada paso", 3000            |  -733.53 +/-  64.83 | 10.50 +/- 0.93 | 1.90 +/- 0.15 | 13.40 +/- 2.47
"Alternar 0/1 cada paso", 5000            |  -748.60 +/-  55.87 | 10.69 +/- 0.81 | 1.96 +/- 0.13 | 13.47 +/- 3.10
```

"Pedir siempre la fase contraria" (acción = `1 - green_phase`, es decir, cambiar en
cuanto `min_green` lo permite) **reproduce a v1 casi exacto**: la misma espera, cola y
throughput a dos decimales, la misma desviación del reward, y un reward por episodio
0.0–0.1 peor que v1. "Alternar 0/1 cada paso" es mucho peor que tiempo fijo, porque la
mitad de sus peticiones caen en pasos bloqueados por `min_green`.

**(b) Comparación contrafactual sobre las trayectorias reales de v1** (v1 conduce SUMO
en las semillas 3000–3014 y 5000–5014; en cada estado real se pregunta qué elegirían
v1, el PPO directo y la regla, sin ejecutar nada más). Los 30 rewards de v1 reproducen
sus valores oficiales. Un paso está "bloqueado" cuando
`time_since_last_phase_change < yellow_time + min_green`: ahí ninguna acción cambia el
tráfico, solo la penalización de 0.1 (ver nota técnica, punto 7).

```
1800 pasos, 930 bloqueados por min_green (51.7%; 31 de 60 en cada episodio)

Par                   | acuerdo total | desacuerdos | en pasos bloqueados | acuerdo en pasos NO bloqueados
v1 / PPO directo      |     78.7%     |     383     |     376 (98.2%)     |   99.2%  (7 de 870)
v1 / regla            |     97.5%     |      45     |      45 (100%)      |  100.0%  (0 de 870)
PPO directo / regla   |     79.8%     |     364     |     357 (98.1%)     |   99.2%  (7 de 870)
```

En los pasos donde la acción sí afecta al tráfico, **v1 coincide con la regla en 870 de
870** y el PPO directo en 863 de 870. Los desacuerdos restantes caen casi enteramente en
pasos bloqueados, donde la acción no tiene efecto real y solo mueve la penalización de
0.1.

### 4. Conclusión honesta

**Ninguno de los dos PPO aprendió control dependiente del estado del tráfico.** Ambos
redescubrieron la misma regla simple, "pedir siempre la fase contraria", que es la mejor
política encontrada para este escenario concreto (demanda baja y simétrica, dos fases).
No está demostrado que sea óptima: supera a tiempo fijo y a "alternar 0/1", y dos
optimizadores independientes convergieron a ella, lo que la señala como la mejor
estrategia alcanzable aquí, pero no es una prueba de optimalidad.

**En este escenario no hay ahorro demostrable de interacciones reales con SUMO.**
Contando todo lo que cada método necesitó de SUMO real para llegar a esa regla:

| Método | Pasos reales de SUMO hasta la regla |
|---|---|
| World Model (v1) | **2,400** (dataset de 40 episodios × 60 pasos con acciones aleatorias, usado para entrenar y seleccionar el Autoencoder y el LSTM, sembrar el Dream Environment y validar el PPO) + 0 durante el entrenamiento del PPO |
| RL directo | **≤2,000** de entrenamiento + 600 de evaluación = **≤2,600** (primera evaluación en el timestep 2000 ya en -285.20; como se evaluaba cada 1000 pasos, 2000 es una cota superior) |

Los dos costos son del mismo orden de magnitud, y el del RL directo es una cota superior.
El PPO v1 sí entrenó con cero pasos reales, pero el método World Model completo no: su
costo en SUMO está en la recolección del dataset.

La pregunta de investigación ("¿puede un modelo aprendido de la dinámica reducir las
interacciones necesarias con SUMO sin perder desempeño?") se responde así para este
escenario: **ambos métodos alcanzan el mismo desempeño con un costo de interacción
comparable; no hay reducción demostrable.** Además, la comparación sostiene conclusiones
sobre el costo para llegar a la regla, no sobre la calidad de control, porque ninguno de
los dos aprendió control. El ahorro de interacciones que predice la propuesta solo sería
demostrable en un escenario donde el RL directo necesitara **sustancialmente más de
2,400 pasos reales** para converger. Este escenario, por ser demasiado simple, nunca lo
exige (ver punto 6).

### 5. Tabla final (SUMO real, 15 episodios por semilla, mismo protocolo)

```
                                       |             reward |     espera_prom |     cola_prom |    throughput
PPO v1 (World Model), seed_base=3000   |  -287.76 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
PPO v1 (World Model), seed_base=5000   |  -293.35 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
PPO RL directo, seed_base=3000         |  -293.57 +/-  37.21 |  3.91 +/-  0.49 | 1.16 +/- 0.12 | 13.47 +/- 2.31
PPO RL directo, seed_base=5000         |  -297.63 +/-  38.60 |  3.96 +/-  0.52 | 1.18 +/- 0.10 | 13.67 +/- 2.65
Tiempo fijo (ciclo=5), seed_base=3000  |  -570.27 +/-  86.70 |  8.05 +/-  1.25 | 1.66 +/- 0.21 | 13.67 +/- 2.55
Tiempo fijo (ciclo=5), seed_base=5000  |  -605.27 +/- 123.47 |  8.57 +/-  1.75 | 1.73 +/- 0.28 | 13.87 +/- 2.63
Regla "fase contraria", seed_base=3000 |  -287.83 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
Regla "fase contraria", seed_base=5000 |  -293.43 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
```

La regla "fase contraria" es la referencia del techo alcanzado en este escenario (la
mejor política encontrada, sin aprendizaje). Frente a ella, v1 empata y el PPO directo
queda levemente por debajo: su peor episodio es -403.6 (semilla 3009), frente a -314.1
de la regla en la misma semilla. Las ventajas de v1 sobre tiempo fijo documentadas en
la sección anterior (~50% menos espera, ~31% menos cola) siguen siendo ciertas
numéricamente, pero son ventajas de la regla, no de control aprendido.

### 6. Limitación más importante del proyecto (trabajo futuro)

El escenario actual (demanda baja y simétrica: 200/90/60 vehículos por hora por acceso
para recto/izquierda/derecha, dos fases) es **demasiado simple para que el control
dependiente del estado aporte ventaja sobre una regla fija**: cambiar de fase en cuanto
se pueda ya alcanza el mejor resultado encontrado. Un escenario con **demanda asimétrica
o variable en el tiempo** es necesario para que la comparación distinga los métodos por
calidad de control, y no solo por costo de aprendizaje. También es la condición para
poder medir el ahorro de interacciones que predice la propuesta: requiere un problema
donde el RL directo necesite sustancialmente más interacciones reales que las 2,400 del
dataset del World Model.

### 7. Nota técnica adicional (sin corregir)

`ProjectRewardFunction` resta `delta × phase_change` (`delta=0.1`), y
`TrafficEnvironment.step()` define `info["phase_change"] = float(action == 1)`. Es decir,
se penaliza **pedir la fase 1**, no cambiar efectivamente de fase. Tiene la misma raíz
que la confusión de semántica de acción ya documentada (la acción es el índice de fase
verde destino, no "mantener/cambiar"). Es la única señal que distingue entre sí las
acciones en pasos bloqueados por `min_green`, y es la explicación más probable (no
verificada paso a paso) de las diferencias de 0.0–0.1 por episodio entre v1 y la regla.

## ✅ Controlador PPO contra SUMO real: historia completa y resultado oficial (v1)

> **Actualización posterior**: el bloque del baseline de RL directo (sección de arriba)
> mostró que v1 equivale en la práctica a la regla "pedir siempre la fase contraria"
> (100% de acuerdo en los pasos donde la acción afecta al tráfico). Los números de esta
> sección siguen siendo correctos, pero describen esa regla, no control aprendido
> dependiente del estado.

Esta sección reemplaza por completo la versión anterior. Cuenta en orden cronológico lo
que realmente pasó. **Los pasos (a)–(d) se hicieron sin saberlo con un puente de
evaluación defectuoso**; sus números se conservan como registro, no como resultados
válidos. El resultado oficial está en la tabla final de esta sección.

**Infraestructura de evaluación**: `environments/encoded_traffic_environment.py`
(`EncodedTrafficEnvironment`, commit `915bd79`) envuelve `TrafficEnvironment` y corre el
Encoder congelado en vivo para traducir el estado crudo de 26 dimensiones a `z` antes de
que la política PPO (entrenada enteramente en `z` dentro del Dream Environment) lo vea.
Junto con él, `scripts/evaluate_controller_sumo.py` compara PPO contra tiempo fijo y
acción aleatoria con métricas reales de tráfico (espera, cola, throughput) de
`TrafficEnvironment`, no solo con el reward abstracto. Protocolo: 15 episodios por
semilla base, semillas de evaluación `seed_base=3000` y `seed_base=5000`.

### (a) v1 entrenado (`max_dream_steps=7`) y primera evaluación, con el bug sin saberlo

v1 es el PPO entrenado dentro del Dream Environment con su horizonte por defecto de 7
pasos; el detalle del entrenamiento está en la sección "Controlador PPO — implementado y
entrenado dentro del Dream Environment", más abajo. Primera evaluación en SUMO: con 5
episodios PPO parecía ganar a tiempo fijo (-453.18 frente a -547.40). Con 15 episodios
(`seed_base=3000`) se invirtió: **-682.09 ± 652.41** frente a -570.27 ± 86.70, con un
patrón bimodal. 11 de 15 episodios eran mejores que tiempo fijo (media ≈-348), pero 4
eran catastróficos: -927.80, -1033.40, -2540.60 y -1902.50.

### (b) Diagnóstico, también con el bug sin saberlo

Tres de los cuatro episodios catastróficos (seeds 3001, 3011, 3013) tenían rachas de la
misma acción de 6, 9 y 14 pasos, más largas que el horizonte de 7 pasos del Dream
Environment. Se concluyó que la política no había podido experimentar esas rachas
durante el entrenamiento. El cuarto (seed 3005) se clasificó como un modo de fallo
distinto: un pico puntual de `waiting_total=83.00` en el paso 30, sin racha anormal
(`max_run=4`). Quedó documentado como limitación conocida no resuelta. **Ver (i): todo
este diagnóstico resultó ser un artefacto del bug.**

### (c) Fix aplicado por ese diagnóstico: `dream_max_steps=20` → v2

Nuevo campo `ControllerConfig.dream_max_steps=20` (commit `8d84d52`), exclusivo de este
entrenamiento, sin tocar el default de `DreamEnvironment` (7). Se respaldó v1 como
`best_model_v1_dream7.zip`, `ppo_controller_final_v1_dream7.zip` y
`evaluations_v1_dream7.npz`, y se reentrenó completo como v2 (50,176 timesteps).

### (d) v2 evaluado, todavía con el bug: parecía mejor que v1

```
                              |            reward |    espera_prom |    cola_prom |   throughput
v2 (con bug), seed_base=3000  | -290.05 +/-  57.70 | 3.92 +/- 0.81 | 1.09 +/- 0.16 | 13.40 +/- 3.88
v2 (con bug), seed_base=5000  | -316.76 +/-  52.76 | 4.28 +/- 0.73 | 1.16 +/- 0.16 | 12.40 +/- 2.92
```

Sin episodios catastróficos en ninguna semilla. Se documentó como resultado final del
método (commit `a5483e1`). **Estos números no son válidos**: ver (e)–(f).

### (e) Se encuentra el bug real: el puente no normalizaba con `scaler.pkl`

Al preparar el baseline de RL directo se vio que `EncodedTrafficEnvironment` codificaba
el estado crudo de SUMO **sin aplicar `scaler.pkl`**, el scaler con el que
`scripts/normalize_dataset.py` normalizó los datos de entrenamiento del Autoencoder. El
`z` resultante caía fuera del espacio latente donde aprendieron el LSTM y el PPO. Fix
(commit `990c6e5`): cargar `scaler.pkl` en `__init__` y aplicar
`(raw - state_mean) / state_std` antes de `encode()`, con el test
`test_state_is_normalized_with_scaler_before_encode`.

Antes de diagnosticar, se descartó que `scaler.pkl` y `autoencoder_best.pt` vinieran de
datasets distintos. Ambos salen de la misma pasada del pipeline (21/09, 22:26:08 a
22:27:06), sobre los 40 episodios (28/6/6) generados con el fix de fase ya aplicado: la
fase alterna (52.2%/47.8%) y `remaining` está en [0, 5], no en el 86400 del bug antiguo.
Numéricamente: `scaler.mean` coincide exactamente con la media de `train_raw.npz`;
`train.npz` es exactamente `(train_raw - mean) / std`; el Autoencoder reconstruye
`train.npz` con MSE 0.0186 frente a 15.83 sobre el estado crudo; y `train_latent.npz` es
bit a bit `encode(train.npz)`.

### (f) v2 re-evaluado con el puente corregido: peor de lo documentado, aún mejor que tiempo fijo

```
                              |            reward |    espera_prom |    cola_prom |   throughput
v2 (corregido), seed_base=3000| -421.69 +/-  98.90 | 5.82 +/- 1.44 | 1.38 +/- 0.23 | 13.60 +/- 2.50
v2 (corregido), seed_base=5000| -419.65 +/-  75.22 | 5.81 +/- 1.07 | 1.36 +/- 0.18 | 13.40 +/- 2.65
```

Sigue superando a tiempo fijo, comparando episodio por episodio con la misma semilla
(15/15 y 14/15), pero los rangos ya se solapan y el margen es menor: ~28-32% menos espera
y ~17-21% menos cola. Sin episodios catastróficos.

### (g) Por qué el `z` con bug daba mejor reward en v2: investigado, causa no identificada

Hipótesis probadas en orden:

1. **"El `z` con bug está degenerado/saturado": descartada.** Sobre 10 pasos reales
   (seed 3000, acciones aleatorias), el `z` sin normalizar tenía **mayor** desviación
   estándar que el correcto en **16/16** dimensiones. No era degeneración sino estar
   fuera de rango: [-20.72, 22.54] frente a [-13.51, 11.30] en todo `train_latent.npz`
   (dim 15: media 10.25 frente a un máximo de entrenamiento de 8.24). El `z` correcto
   cae en [-6.18, 4.30].
2. **Qué decide la política con cada `z` sobre los mismos estados.** Comparación
   contrafactual: 15 trayectorias guiadas por la versión correcta, 900 pasos, y en cada
   paso se pregunta qué habría elegido cada versión. Discrepan en el 32.2% de los pasos,
   con un sesgo en una sola dirección: cuando la correcta elige 1, la del bug elige 0 en
   273/453 (60%); a la inversa, solo en 17/447 (3.8%). Las rachas contrafactuales llegan
   a 12, frente a un máximo de 6 en la versión correcta. Estas cifras son decisiones
   sobre estados ajenos, no el comportamiento de la versión con bug en su propia
   trayectoria.
3. **"El bug hace menos cambios de fase y ahorra amarillo": descartada** con
   trayectorias propias (cada política conduce SUMO, seeds 3000–3014; el amarillo se
   mide por segundo simulado leyendo `TrafficSignal.is_yellow`):

   ```
   Politica                 |       reward medio | cambios/ep | amarillo_s/ep | verde F0 % | verde F1 %
   v2 corregido             |  -421.69 +/-  98.90 | 24.0 +/- 1.1 |  48.0 +/- 2.2 |       50.4 |       49.6
   v2 con bug               |  -290.05 +/-  57.70 | 26.5 +/- 0.6 |  52.9 +/- 1.2 |       56.3 |       43.7
   Tiempo fijo (ciclo=5)    |  -570.27 +/-  86.70 | 22.0 +/- 0.0 |  44.0 +/- 0.0 |       65.6 |       34.4
   ```

   La versión con bug hacía **más** cambios y más amarillo, no menos. Estas corridas
   reproducen exactamente los rewards ya reportados de las tres políticas.
4. **Mecanismo causal: NO identificado.** El bug sesgaba la política hacia la fase 0,
   pero por qué ese patrón puntuaba mejor queda como **curiosidad abierta**, no
   investigada a fondo. La única hipótesis no descartada, que no se midió, es el
   *momento* de los cambios respecto al estado de las colas.

### (h) Re-evaluación de v1 con el puente corregido, por si el diagnóstico original también estaba contaminado

Mismo protocolo, sin reentrenar. Como control, el mismo checkpoint v1 con el puente
viejo reproduce **exactamente** -682.09 ± 652.41, con los mismos cuatro episodios
catastróficos en las mismas semillas (3001, 3005, 3011, 3013). El checkpoint es el
diagnosticado en (a)–(b), y lo único que cambia entre corridas es la normalización.

### (i) Resultado: v1 corregido es el mejor controlador del proyecto; el diagnóstico de (b) era un artefacto

- **Cero episodios catastróficos en v1**: el peor de 30 es -366.00. Las seeds que antes
  fallaban dan ahora -271.00 (3001), -312.00 (3005), -263.10 (3011) y -312.10 (3013).
  Esto incluye la seed 3005, el "modo de fallo distinto": también era el bug.
- **v1 supera a v2 corregido en las 30 comparaciones** episodio por episodio (15/15 en
  cada semilla), por unos 130 puntos de reward y con menos varianza.
- **v1 supera a tiempo fijo en las 30 comparaciones, sin solapamiento de rangos**:
  v1 [-338.00, -257.00] frente a tiempo fijo [-766.20, -425.20] (`seed_base=3000`);
  v1 [-366.00, -232.00] frente a [-808.20, -418.20] (`seed_base=5000`).
- Conclusión: las rachas de acción largas de (b) no eran una limitación real de
  `max_dream_steps=7`. Eran el efecto de alimentar a la política con un `z` fuera de
  distribución. Con el `z` correcto, el horizonte de 20 **empeoró** la política real.

### (j) Decisión final

Se revierte `ControllerConfig.dream_max_steps` a 7 (commit posterior a `990c6e5`) y v1
pasa a ser el checkpoint oficial (`models/checkpoints/controller/best_model.zip`). v2 se
conserva como `best_model_v2_dream20_deprecated.zip`, como evidencia de un cambio que
parecía buena idea y no lo fue.

### Resultado oficial del método World Model (PPO v1, puente corregido)

```
                               |             reward |     espera_prom |     cola_prom |    throughput
PPO v1, seed_base=3000         |  -287.76 +/-  23.87 |  3.82 +/-  0.32 | 1.15 +/- 0.09 | 13.47 +/- 2.03
PPO v1, seed_base=5000         |  -293.35 +/-  41.28 |  3.90 +/-  0.56 | 1.17 +/- 0.11 | 13.87 +/- 2.53
Tiempo fijo, seed_base=3000    |  -570.27 +/-  86.70 |  8.05 +/-  1.25 | 1.66 +/- 0.21 | 13.67 +/- 2.55
Tiempo fijo, seed_base=5000    |  -605.27 +/- 123.47 |  8.57 +/-  1.75 | 1.73 +/- 0.28 | 13.87 +/- 2.63
Aleatoria, seed_base=3000      | -1753.63 +/- 946.86 | 26.29 +/- 14.62 | 3.12 +/- 1.17 | 14.47 +/- 2.80
Aleatoria, seed_base=5000      | -1702.63 +/- 879.77 | 25.51 +/- 13.62 | 3.03 +/- 1.05 | 12.87 +/- 2.25
```

Frente a tiempo fijo: reward ~50% mejor (49.5% y 51.5%), **espera ~53-55% menor** (52.5%
y 54.5%), **cola ~31-32% menor** (30.7% y 32.4%). **Throughput: PPO sigue sin ser mejor
que tiempo fijo** (13.47 frente a 13.67; empate exacto en 13.87). Esta salvedad se ha
mantenido en todas las corridas del proyecto, con y sin bug.

### Lección metodológica

Se investigó un fallo y se "arregló" (`dream_max_steps=20`, un reentrenamiento completo)
**sin haber descartado antes un bug en la propia herramienta de evaluación**. El
diagnóstico de (b) era internamente coherente (rachas más largas → peores episodios) y
por eso pareció confirmado, pero medía un síntoma del puente defectuoso, no del
controlador. Además, el "fix" pareció funcionar en (d) porque se evaluó con la misma
herramienta rota. Lección: cuando una evaluación da resultados inesperados, verificar
primero que la herramienta de evaluación reproduce fielmente las condiciones de
entrenamiento (aquí, la misma normalización de entrada) antes de cambiar el modelo o su
entrenamiento. Y re-evaluar todo lo medido con una herramienta después de corregirla, no
solo el último resultado.

### Notas técnicas (observaciones, no corregidas)

- **Semántica de la acción**: la acción **no** significa "mantener/cambiar" como dice el
  docstring de `ProjectActionSpace`. sumo-rl la trata como **índice de fase verde
  destino** (`TrafficSignal.set_next_phase`): solo hay cambio si
  `new_phase != green_phase` y ya pasaron `yellow_time + min_green`. Con 2 fases,
  `acción=1` equivale a "cambiar" solo cuando la fase actual es la 0; cuando es la 1,
  `acción=1` significa mantener y `acción=0` significa cambiar. En las trayectorias
  medidas cada fase ocupa ~50% del tiempo, así que la coincidencia con la documentación
  es de alrededor de la mitad de los pasos. El pipeline es coherente internamente
  (dataset, LSTM y PPO usan la misma convención), pero hay que leer retroactivamente así
  los nombres ya usados: "Siempre cambiar (1)" = "siempre pedir la fase 1"; "Siempre
  mantener (0)" = "siempre pedir la fase 0"; una racha de acción = "sostener una fase";
  "Tiempo fijo (ciclo=5)" no es un ciclo simétrico (63.3% de los pasos en fase 0). Queda
  pendiente en TODO.md.
- **La política aleatoria se reproduce exactamente entre corridas** (-1753.63 ± 946.86 y
  -1702.63 ± 879.77 en todas las ejecuciones, incluso en procesos distintos), aunque usa
  `np.random` sin semilla explícita. Explicación más probable, no verificada: al cargar
  el modelo, `PPO.load` llama a `set_random_seed` con la semilla guardada, lo que fija la
  semilla global de numpy antes de que corra la política aleatoria. No se investigó más.

**Tests**: 36/36 en verde (`test_controller_config_rejects_invalid_dream_max_steps`,
`test_state_is_normalized_with_scaler_before_encode`, más los previos).

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

**⚠️ Salvedad importante, planteada sin resolver en la corrida original**: la tasa de
`reward_clipped` de PPO (13.3%) es **más alta** que la de la política "alternando"
(4.8%), a pesar de que PPO evita rachas largas por completo. Se planteó como posible
síntoma de que la política está explotando el recorte de recompensa para volver
artificialmente barato el error de extrapolación del modelo, en vez de aprender control
de tráfico genuino.

**Investigación de seguimiento — commit `8044262`**: antes de gastar cómputo en SUMO
real, se investigó la hipótesis con una medición más precisa: la **magnitud** del
recorte (`|raw_predicted_reward - bound|`), no solo su frecuencia. Se expuso
`info["raw_predicted_reward"]` (el valor crudo del LSTM antes de `torch.clamp`) en
`DreamEnvironment.step()`, y se creó `scripts/analyze_controller_actions.py` para
comparar PPO contra las políticas de referencia:

```
Rango de recorte: [-165.05, 1.0]

Politica                  |  n_clips |  mag_media |    mag_max |  accion=0 |  accion=1
-------------------------------------------------------------------------------------
PPO (entrenado)           |       25 |       3.05 |      18.25 |       100 |       110
Alternando cada paso      |        9 |       2.75 |       4.89 |       120 |        90
Siempre mantener (0)      |       29 |      10.15 |      31.60 |       210 |         0
Siempre cambiar (1)       |       34 |      32.84 |      91.25 |         0 |       210
```

**Resultado, sin suavizar**: la magnitud del recorte en PPO (`mag_media=3.05`,
`mag_max=18.25`) es cercana a la de "alternando" (`2.75`/`4.89`) y muy por debajo de
las políticas constantes (`10-33` de media, hasta `91` de máximo). Cuando PPO dispara
el recorte, el valor crudo queda apenas fuera del rango empírico — no profundamente en
territorio alucinado, como sí ocurre con las políticas constantes. La distribución de
acciones de PPO tampoco es degenerada (100 vs. 110, similar al balance de "alternando").
Esto **debilita, pero no descarta por completo**, la hipótesis de explotación del
recorte: no se puede descartar con este análisis que PPO esté eligiendo secuencias de
acción específicas (no necesariamente rachas largas ni una acción constante) que
empujan la predicción justo más allá del borde del rango con más frecuencia que
"alternando" — el conteo marginal de acciones no distingue eso de un patrón temporal
particular más sutil.

**Conclusión de esta investigación**: no resuelve la pregunta de fondo (¿control
genuino o artefacto del Dream Environment?), pero sí reduce la prioridad de la
sospecha más grave (explotación profunda del recorte) lo suficiente como para proceder
a la validación contra SUMO real sin gastar más tiempo en análisis dentro del Dream
Environment. **El resultado del PPO sigue sin estar validado** hasta esa evaluación en
`TrafficEnvironment`.

**Tests**: `test_raw_predicted_reward_exposed_in_info` — **34/34 tests en verde** (33
previos + 1 nuevo).

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

1. ~~Cuenta o app desconocida en GitHub~~ — **resuelto**: era Codex Connector; su
   acceso se revocó en ambas capas, y los colaboradores humanos se confirmaron como
   legítimos.
2. ~~`DOCUMENTACION_PROYECTO.md` sigue desactualizado~~ — **resuelto** en el commit
   `9ca20ac` (documentación final subida).
3. ~~Documentación de `ProjectActionSpace` engañosa~~ — **resuelto** en el commit
   `c253d88`: los docstrings y comentarios ya dicen que la acción es el índice de fase
   verde destino, no "mantener/cambiar" (solo documentación, sin cambio de
   comportamiento). La Sección 12 de la propuesta se corrigió en `30c3fae`.
4. `ProjectRewardFunction.phase_change` penaliza pedir la fase 1, no cambiar
   efectivamente de fase (misma raíz que el punto 3; ver la sección del baseline de RL
   directo, punto 7).
5. Ambos PPO de la primera ronda asimétrica nunca sostenían el verde de la fase 1 más
   de 8 s (la duración mínima posible). No se volvió a medir con los checkpoints
   actuales; no investigado a fondo.
6. **Episodios catastróficos: reducidos pero no eliminados** (5/90 en el PPO del sueño,
   23/90 en el directo); causa no identificada por completo.
7. **Función de valor del PPO directo:** `explained_variance` sigue bajo e inestable
   incluso con `VecNormalize` de recompensa y observaciones; causa no identificada.

## ⚪ No implementado todavía

- Más semillas por método (siguen siendo 3), para poder afirmar o descartar diferencias
  a nivel de semilla. El presupuesto del RL directo ya se verificó (30,000 pasos).
- Curva de desempeño frente a interacciones reales del RL directo (hoy solo hay dos
  puntos: 13,000 y 39,000 por semilla), para saber cuántas interacciones necesita para
  alcanzar al World Model.
- Análisis del momento de los cambios de fase respecto a las colas de cada brazo, para
  los episodios catastróficos que quedan.
- Demanda variable en el tiempo (el escenario asimétrico ya está implementado; la
  variación temporal no).

## Qué se estaba haciendo justo antes de este handoff

Se verificó si más presupuesto cierra la brecha del RL directo (limitación abierta de la
segunda ronda). Las 3 semillas del directo se reentrenaron con 30,000 pasos en vez de
10,000, sin tocar los checkpoints oficiales, y se evaluaron con el protocolo de siempre
(reproduciendo antes las medias documentadas del sueño y del directo de 10k). Resultado:
el directo pasa de -453.74 a -335.24, la desviación de sus medias por semilla baja de
96.2 a 30.6, la semilla 2 deja de colapsar a la regla trivial y los episodios
catastróficos bajan de 23/90 a 11/90. Frente al sueño (-326.79), la diferencia ya no es
significativa (p = 0.74 por episodio y 0.71 por semilla) y el directo gana 159 de 270
comparaciones episodio a episodio. Pero cada semilla del directo consumió 39,000
interacciones reales, frente a ~4,600 del World Model (~8.5 veces). La conclusión del
proyecto se precisa: el World Model alcanza un control equivalente con una fracción de
las interacciones reales; su ventaja es de eficiencia, no de control a presupuesto
ilimitado. Antes de esto se había cerrado el Experimento 3 (se mantiene la LSTM) y se
había corregido la documentación de la semántica de la acción (`c253d88`, `30c3fae`).
