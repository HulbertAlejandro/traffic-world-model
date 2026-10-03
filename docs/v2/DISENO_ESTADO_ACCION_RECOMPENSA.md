# v2 — Estado, acción conjunta y recompensa del corredor de 4 intersecciones

Fase 1 de la versión 2.0. Código:

| | |
|---|---|
| Estado, acción, recompensa y entorno Gym | `environments/corridor_environment.py` (`CorridorStateBuilder`, `CorridorActionSpace`, `CorridorRewardFunction`, `CorridorTrafficEnvironment`) |
| Coeficientes de la recompensa | `configs/corridor_reward.py` (`CorridorRewardConfig`) |
| Políticas de recolección | `scripts/v2/corridor_policies.py` |
| Tests | `tests/test_corridor_environment.py`, `tests/test_corridor_concurrent_environments.py` |

La demanda y la red son las de la Fase 0 (`docs/v2/DISENO_RED_4_INTERSECCIONES.md`). El offset
del pulso se sortea con la semilla de cada episodio (sección 7.1 de ese documento); la decisión final sobre la demanda está en la sección 7.11.

## 1. Estado: el de la v1, por intersección, concatenado (104 dimensiones)

`CorridorStateBuilder` crea un `CustomStateBuilder` de la v1 por semáforo, sin modificarlo.
Concatena los cuatro `TrafficState` de 26 dimensiones en el orden A0, B0, C0, D0 (de oeste a
este). Las columnas del semáforo k son `26*k : 26*(k+1)`, y dentro de cada bloque siguen el orden
de la v1:

| Columnas del bloque | Variable |
|---|---|
| 0–3 | `vehicle_counts` de los 4 carriles de entrada (ordenados por id) |
| 4–7 | `queue_lengths` |
| 8–11 | `waiting_times` |
| 12–15 | `mean_speeds` |
| 16–19 | `occupancies` |
| 20–23 | `phase_one_hot` (4 posiciones: 2 fases verdes + 2 amarillas de sumo-rl) |
| 24 | `elapsed_phase_time` |
| 25 | `remaining_phase_time` = `max(0, min_green − elapsed)` |

Usar las mismas variables y el mismo constructor garantiza el mismo criterio de diseño que la
v1, y que cada bloque se pueda comparar con el estado de la v1. Las lecturas pasan por la
conexión TraCI propia de cada entorno (`env.sumo`), es decir, el arreglo del bug C1.

**El estado no incluye el tiempo del episodio.** `elapsed_phase_time` es el tiempo en la fase
actual. Junto con el offset aleatorio del pulso, el controlador no tiene un reloj con el que
memorizar el patrón de demanda.

### Columnas constantes

COLUMNAS_CONSTANTES

## 2. Acción conjunta: mantener/cambiar por semáforo (decisión deliberada)

`MultiDiscrete([2, 2, 2, 2])`: una decisión por semáforo, aplicada a los cuatro a la vez.

- **0 = mantener** la fase verde actual.
- **1 = cambiar** a la otra fase verde.

Como en la v1, sumo-rl ignora un cambio pedido antes de `yellow_time + min_green` = 7 s desde el
anterior. `info["phase_switched"]` registra los cambios reales y `info["switch_requested"]` los
pedidos.

**No es la convención de la v1, y es a propósito.** La acción de la v1 es el **índice de la fase
verde de destino** (`ProjectActionSpace`, CLAUDE.md). Con 2 fases verdes, las dos codificaciones
son una biyección **dada la fase actual**:

```
destino   = fase_actual        si acción = mantener (0)
            1 − fase_actual    si acción = cambiar (1)
acción    = 1 si destino ≠ fase_actual, 0 si no
```

La fase actual de cada semáforo está en el estado (`phase_one_hot` de su bloque), así que no se
pierde información. `CorridorActionSpace.to_target_phases` y `from_target_phases` implementan las
dos direcciones, y un test verifica la ida y vuelta en los 16 vectores de acción con dos
combinaciones de fases. La decisión la confirmó el autor: mantener/cambiar es más simple como
acción conjunta (el significado de cada bit no depende de la fase en que está cada uno de los
otros semáforos) y evita la confusión que la convención de la v1 ya causó (CLAUDE.md,
`info["phase_change"]`).

Consecuencia práctica: una política aleatoria uniforme es la misma distribución en las dos
codificaciones (50% de pedidos de cambio por semáforo), pero los datos y los modelos de la v1 y
de la v2 **no** son intercambiables bit a bit en la columna de acción.

## 3. Recompensa: la "recompensa estilo v1" de la Fase 0, sumada sobre los 4 semáforos

```
R_t = − Σ_{semáforo s}  ( α · Σ_carriles waiting_time_s  +  β · Σ_carriles halted_s ),   α = β = 1
```

Se mide después del paso, en los carriles de entrada de cada semáforo, con los mismos valores
`waiting_times` y `queue_lengths` del estado. Es exactamente la `reward_v1_style` con la que se
calibró la demanda. Un test lo comprueba: con la demanda de la Fase 0 (offset 0) y la misma
semilla, el retorno del entorno reproduce el valor que el validador midió para `fijo_2_3` y para
`cola_mas_larga` (tolerancia relativa de 1e-5).

**Por qué la suma y no el promedio.** El promedio es la suma dividida por 4: las dos tienen la
misma política óptima y solo cambian la escala, que VecNormalize o la normalización del dataset
absorben. Se elige la suma por tres razones:

1. **Unidades interpretables.** La suma son vehículos·s de espera más vehículos detenidos de toda
   la red, las mismas unidades del retorno medido en la Fase 0, que el test reproduce. Cada bloque
   tiene la escala de una intersección de la v1.
2. **El objetivo es la red.** Un vehículo detenido en C0 cuesta lo mismo que uno en B0. La suma
   no le pide a ninguna intersección que "promedie bien".
3. **Lo que sabemos de C0 no se arregla con el promedio, y no conviene esconderlo.** En la Fase 0,
   la referencia gana en A0, B0 y D0 pero pierde en C0, la intersección más cargada. La suma y el
   promedio dan el mismo peso relativo a cada intersección, así que la diferencia de C0 no
   distingue entre las dos. Lo que sí la escondería es normalizar cada intersección por su propia
   escala, porque C0 aportaría menos de lo que pesa en la red. Eso queda descartado: con la suma,
   C0 domina la recompensa en proporción a su tráfico, que es justo donde hay más margen de mejora.

`info["reward_per_signal"]` guarda los cuatro términos por separado, para diagnóstico o para una
asignación de crédito por intersección si una fase posterior la necesita.

**Términos de la v1 que se omiten:**

- **Throughput:** la medida de la v1 está rota (cuenta solo el último segundo de cada paso) y se
  mantuvo allí solo por compatibilidad. `info["arrivals_total"]` registra las llegadas reales,
  con la misma definición correcta de la v1, sin entrar en la recompensa.
- **Penalización de fase:** en la v1 penalizaba **pedir** la fase 1, un efecto de su codificación
  de acción. No se usó al calibrar la demanda; agregarla cambiaría la recompensa con la que se
  validó.

## 4. Conexiones concurrentes

`tests/test_corridor_concurrent_environments.py` abre dos `CorridorTrafficEnvironment` en el
mismo proceso, con semillas y offsets de pulso distintos (19900 y 19901), y los avanza
intercalados 14 pasos con acciones distintas. Los estados y las recompensas de cada uno son
idénticos, bit a bit, a los de correrlo solo. Como el segundo entorno arranca al final, el módulo
`traci` global apunta a él: si algún componente leyera por el módulo global (el bug C1), el
primero vería la simulación del segundo y el test fallaría. Cada instancia escribe además su
propio archivo de rutas en un directorio temporal privado. **Pasa.**
