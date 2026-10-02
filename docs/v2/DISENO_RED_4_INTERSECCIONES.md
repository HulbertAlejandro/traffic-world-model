# v2 — Diseño de la red de 4 intersecciones y calibración de la demanda

Fase 0 de la versión 2.0. Este documento cubre **solo la red de SUMO y la demanda**. El vector de
estado, la acción conjunta y la recompensa de la v2 son la Fase 1 y no están definidos.

Archivos:

| | |
|---|---|
| Red | `environments/four-intersection-corridor/four-intersection-corridor.net.xml` |
| Demanda | `environments/four-intersection-corridor/four-intersection-corridor.rou.xml` (generada, no editar a mano) |
| Configuración de SUMO | `environments/four-intersection-corridor/four-intersection-corridor.sumocfg` (para abrirla en `sumo-gui`) |
| Fábrica del entorno | `environments/four_intersections.py` (`make_corridor_env`, solo infraestructura) |
| Candidatas de demanda y generador | `scripts/v2/corridor_demand.py` |
| Validación contra políticas triviales | `scripts/v2/validate_corridor_demand.py` |
| Resultados por episodio | `docs/results/v2/demand_calibration/<candidata>_<segundos>s.json` |
| Test de humo | `tests/test_four_intersections.py` |

## 1. Topología: corredor de 4 intersecciones en línea

```
            top0        top1        top2        top3
             |           |           |           |
 left0 ---- A0 -------- B0 -------- C0 -------- D0 ---- right0     (arterial Este/Oeste)
             |           |           |           |
          bottom0     bottom1     bottom2     bottom3
```

Las 4 intersecciones están separadas 200 m. La arterial va de Este a Oeste y hay 4 calles
transversales Norte/Sur, todas de un carril por sentido a 13.89 m/s. Los brazos externos miden
200 m.

**Por qué un corredor y no una cuadrícula 2x2:**

1. **Cada intersección es idéntica a la de la v1.** Tiene 4 accesos de un carril, el mismo
   programa de 2 fases verdes (0 = Norte/Sur, 1 = Este/Oeste) y los mismos parámetros. La red se
   genera con el mismo comando `netgenerate` de la v1, cambiando solo `--grid.x-number 4
   --grid.y-number 1` (comando completo en la sección 4). Lo aprendido en la v1 sobre una
   intersección (la fase 0 sirve a Norte/Sur, los giros a la izquierda permisivos en carril
   compartido, la cola invisible al saturar un acceso) se aplica igual a cada intersección.
2. **La interacción entre intersecciones es real y fácil de explicar.** Los pelotones que suelta
   A0 llegan a B0 unos 14 s después (200 m a 13.89 m/s, unos 3 pasos de control). Cada tramo
   interno almacena unos 26 vehículos antes de bloquear la intersección de aguas arriba. La
   coordinación de una arterial ("onda verde") es el problema clásico de control multi-
   intersección.
3. **No hay elección de ruta.** En una línea cada origen-destino tiene un único camino. En una
   2x2, el mismo O-D tiene dos caminos de igual longitud. Repartir el tráfico entre ellos sería
   una decisión de la demanda que cambia lo que ve cada semáforo, y la cuadrícula tiene ciclos
   que pueden bloquearse (gridlock) en las dos direcciones. Eso agrega fuentes de variación que
   no son control.
4. **El costo es menor.** Hay 3 tramos internos en vez de 4, y la red tiene 26 aristas.

Limitación aceptada: las calles transversales solo interactúan entre sí a través de la arterial,
y solo A0 y D0 reciben tráfico arterial externo.

## 2. Parámetros de SUMO y sumo-rl (iguales a la v1)

| Parámetro | Valor | Nota |
|---|---|---|
| `delta_time` | 5 s | un paso de control |
| `yellow_time` | 2 s | se descuenta del paso en que ocurre el cambio |
| `min_green` | 5 s | un cambio exige `yellow_time + min_green` = 7 s desde el anterior |
| `max_green` | 50 s | sumo-rl no lo impone en las acciones (igual que en la v1) |
| Duración del episodio | 300 s = 60 pasos | la validación se repite a 900 s |
| Teleport | desactivado (`time_to_teleport=-1`) | |
| Tipo de vehículo | el `passenger` de la v1 | `sigma=0.5`, `tau=1.0` |

Están escritos de forma explícita en `environments/four_intersections.py`, para que una
actualización de sumo-rl no los cambie en silencio. Como en la v1, sumo-rl reemplaza el programa
estático del `.net.xml`: usa sus 2 fases verdes y construye los amarillos, y la acción de cada
semáforo es el **índice de la fase verde de destino**.

Con estos parámetros, "cambiar apenas se cumpla el verde mínimo" equivale exactamente a tiempo
fijo con 2 pasos por fase (`fijo_2_2`). Las dos dan números idénticos en todas las iteraciones.

## 3. Calibración de la demanda

### 3.1 Protocolo y criterios (fijados antes de correr la primera candidata)

Los criterios se fijaron en el código (`validate_corridor_demand.py`) antes de correr it1. No hay
un commit intermedio que lo pruebe: todo se commiteó al final.

**Semillas de calibración: 11000–11019 (20 episodios).** No se cruzan con las de la v1 (3000,
5000, 7000) y **ninguna evaluación de la v2 debe usarlas**, porque la demanda se ajustó mirándolas.
Es la misma lección de la nota del 26 de septiembre en PROJECT_STATUS.md.

**Políticas triviales** (lo que la demanda NO debe volver óptimo). Las 4 señales usan el mismo
programa, con desfase 0:

- `min_verde_y_cambiar`: pide siempre la otra fase. Es la regla que la demanda simétrica de la v1
  volvió óptima.
- `fijo_N0_N1`: tiempo fijo con N0 pasos en la fase 0 y N1 en la fase 1. Se prueba una grilla de
  6 x 6 con N ∈ {2, 3, 4, 6, 8, 10}. La mejor celda se elige **sobre las mismas semillas**, así
  que la comparación favorece al tiempo fijo (es conservadora).
- `fijo_v1`: el baseline de tiempo fijo de la v1, tal cual.
- `siempre_arterial`: no deja nunca la fase de la arterial. Es un control de cordura.

**Referencias dependientes del estado** (sin entrenamiento). Solo sirven para mostrar que existe
una política que mira el tráfico y le gana a toda trivial; no son óptimas:

- `cola_mas_larga`: da verde a la fase cuyos carriles de entrada tienen más vehículos detenidos.
- `max_presion`: max-pressure por enlace, (vehículos en el carril de entrada) − (vehículos en el
  de salida).

**Métricas.** Ninguna es la recompensa de la v2, que es de la Fase 1:

- **Demora** (vehículos·s detenidos): cada segundo simulado se suman los vehículos detenidos en
  todos los carriles más los **vehículos pendientes de inserción**. Así se cuenta la cola fuera
  de la red que la v1 descubrió con la demanda 700/150.
- **Recompensa al estilo de la v1:** −(tiempo de espera + vehículos detenidos) en los carriles de
  entrada de cada semáforo, sumada sobre los 4 semáforos y los 60 pasos. Son los términos alfa y
  beta de la recompensa de la v1; los términos de throughput y de penalización de fase se omiten.
  Se incluye porque es el candidato natural de la Fase 1, y una demanda que solo es no trivial
  bajo otra métrica volvería a caer en la trampa.

**Criterios de aceptación.** Se exigen los tres:

1. **Sin cola invisible:** con la mejor referencia, como máximo 5 vehículos pendientes de
   inserción después de los primeros 30 s, en todos los episodios.
2. **La demora no es trivial:** la mejor referencia le gana a **cada** política trivial, con una
   reducción media pareada ≥ 10% y p < 0.01 en la t pareada y en el Wilcoxon pareado.
3. **La recompensa al estilo de la v1 no es trivial:** la mejor referencia mejora frente a
   **cada** política trivial, con p < 0.01 en la t pareada.

Los tests son las funciones del proyecto: `paired_t` de `scripts/evaluate_multiseed_statistical.py`
y el Wilcoxon validado de `docs/results/ppo_10_seeds/analyze.py`.

### 3.2 Modelo de demanda

Parámetros de `scripts/v2/corridor_demand.py`, en veh/h:

- **Arterial:** `arterial_west` entra por el extremo oeste (hacia el Este) y `arterial_east` por el
  extremo este (hacia el Oeste). En cada una de las 4 intersecciones, el 3.75% del flujo que entró
  gira hacia el Norte y otro 3.75% hacia el Sur. El 70% restante cruza el corredor entero.
- **Calles transversales:** `cross[J] = (norte, sur)`. El 60% sigue recto y el 40% gira hacia la
  arterial, mitad hacia cada extremo.
- **Llegadas:** `uniform` (`vehsPerHour`, espaciado regular, como la v1) o `poisson`
  (`period="exp(tasa)"`).
- **Segmentos** (opcional): demanda que varía en el tiempo y se repite cíclicamente.

### 3.3 Iteraciones

Medias de 20 episodios de 300 s. Las mejoras son las de la mejor referencia (siempre
`cola_mas_larga`) frente a la **mejor política trivial**, pareadas por semilla. "Pend." es el
máximo de vehículos pendientes de inserción después de los 30 s, con la mejor referencia.

| Iteración | Demanda (veh/h) | Mejor trivial | Mejora en demora | Mejora en recompensa v1 | Pend. | ¿Aceptada? |
|---|---|---|---|---|---|---|
| it1 | uniforme; arterial 600/400; transversales A0 250/150, B0 100/60, C0 300/200, D0 80/120 (2,260 en total) | `min_verde_y_cambiar` | +14.5% (t p = 4e-7) | +3.5% (p = 0.55) | 13 | No |
| it2 | it1 con Poisson | `min_verde_y_cambiar` | +9.4% (p = 0.014) | +1.6% (p = 0.79) | 4 | No |
| it3 | it2 con arterial 750/450 (2,460) | `fijo_2_3` | +8.0% (p = 0.057) | +1.3% (p = 0.88) | 8 | No |
| it4 | it2 con transversales A0 200/120, B0 40/30, C0 350/250, D0 40/30 (2,060) | `min_verde_y_cambiar` | +17.1% (p = 1e-4) | +10.0% (p = 0.13) | 3 | No |
| **it5** | it4 con pulsos de 150 s (ver 3.4) (2,060 de media) | `fijo_2_3` | **+20.2%** (t p = 1e-5, W p = 3e-5) | **+23.9%** (p = 2e-4) | 4 | **Sí** |

**it1, rechazada por cola invisible y porque la regla trivial es casi óptima.** Quedaron 12–14
vehículos pendientes de inserción después de los 30 s con **todas** las políticas, incluso con
`siempre_arterial`. La causa no es el control: los 9 flujos de cada entrada arterial tienen
espaciado regular y los 8 de giro comparten período, así que insertan sus vehículos en ráfagas
sincronizadas. Además, los 26 flujos insertan su primer vehículo en t = 0 (32 pendientes al
arranque, el mismo artefacto de la v1 multiplicado). Con la recompensa al estilo de la v1,
`min_verde_y_cambiar` queda dentro del ruido de la mejor referencia: **es la trampa de la v1 otra
vez**.

**it2, rechazada.** Las llegadas de Poisson eliminan la cola invisible y la ráfaga del arranque
(como máximo 4 pendientes, incluidos los primeros 30 s). Pero la ventaja de la referencia baja al
9.4% y sigue sin ser significativa con la recompensa al estilo de la v1. Con una demanda
estacionaria y parecida en las 4 intersecciones, un ciclo corto y uniforme está cerca de lo mejor
posible.

**it3, rechazada.** Más carga en la arterial no rompe la trivialidad: la referencia gana menos
(8%) y vuelve la cola fuera de la red (8 pendientes con la mejor referencia). La saturación no es
la palanca.

**it4, rechazada por poco.** Con transversales muy distintas entre sí (B0 y D0 casi vacías, C0
cargada), ningún programa compartido acierta en las 4 intersecciones. La demora pasa el criterio
(+17.1%), pero la recompensa al estilo de la v1 no (+10.0%, p = 0.13).

**it5, aceptada.** Se agrega variación temporal: el sentido dominante de la arterial y el pico de
la transversal de C0 cambian cada 150 s, y ningún programa fijo sirve para las dos mitades.

### 3.4 Demanda final (it5)

La base es it4. Cada episodio de 300 s tiene dos mitades, y el patrón se repite cada 300 s hasta
el final del archivo de rutas (3,600 s). En veh/h:

| | Arterial desde el oeste (→E) | Arterial desde el este (→O) | A0 N/S | B0 N/S | C0 N/S | D0 N/S |
|---|---|---|---|---|---|---|
| 0–150 s | 750 | 250 | 200/120 | 40/30 | 200/150 | 40/30 |
| 150–300 s | 450 | 550 | 200/120 | 40/30 | 500/350 | 40/30 |

Total medio: 2,060 veh/h. Llegadas de Poisson y reparto de giros como en 3.2.

Resultados de it5 a 300 s (20 semillas):

| Política | Demora (veh·s) | Recompensa v1 | Llegados | Cambios de fase |
|---|---|---|---|---|
| `cola_mas_larga` (referencia) | 1,288 | −1,189 | 119.8 | 55.8 |
| `fijo_2_3` (mejor tiempo fijo uniforme) | 1,614 | −1,561 | 116.5 | 92.0 |
| `min_verde_y_cambiar` = `fijo_2_2` | 1,694 | −1,498 | 111.2 | 116.0 |
| `max_presion` (referencia) | 1,762 | −7,288 | 121.8 | 39.4 |
| `fijo_v1` | 3,202 | −3,861 | 101.7 | 88.0 |
| `siempre_arterial` | 11,332 | −206,783 | 61.9 | 4.0 |

Frente a cada trivial, la referencia mejora:

| Política trivial | Demora | Recompensa v1 |
|---|---|---|
| `fijo_2_3` | +20.2% | +23.9% |
| `min_verde_y_cambiar` | +24.0% | +20.7% |
| Resto de la grilla | +37.8% o más | +54.8% o más |

Todos los p son < 4e-4. Gana en 16–20 de las 20 semillas.

**Lo que no se afirma.** `cola_mas_larga` no es óptima. Su demora por intersección es A0 332,
B0 129, C0 651 y D0 147 veh·s; la de `min_verde_y_cambiar` es 671, 319, **368** y 305. La
referencia gana en A0, B0 y D0, pero en C0, la intersección más cargada, es peor que cambiar
rápido. Ninguna de las políticas probadas domina en las 4 intersecciones: queda margen para un
controlador aprendido, y la regla trivial no está "a un paso" de lo mejor posible. `max_presion`
reduce poco la demora y es muy mala con la recompensa al estilo de la v1: mantiene verdes largos y
acumula esperas largas en pocos vehículos.

**Consecuencia para la Fase 1.** Los pulsos son periódicos y están alineados con el inicio del
episodio. Un controlador puede inferir en qué mitad está por las colas, o por el tiempo si el
estado lo incluyera. Hay que decidirlo al diseñar el estado: incluir el tiempo del episodio haría
que parte de la política se aprendiera "de memoria".

### 3.5 Robustez a 900 s

Se usa la misma demanda; los pulsos se repiten 3 veces por episodio. Mismas semillas y
políticas, `--seconds 900`:

| Política | Demora (veh·s) | Recompensa v1 | Pend. máx. después de 30 s |
|---|---|---|---|
| `cola_mas_larga` | 4,956 | −5,082 | 4 |
| `fijo_2_3` (mejor trivial) | 6,523 | −6,444 | 5 |
| `max_presion` | 6,562 | −28,041 | 4 |
| `min_verde_y_cambiar` | 8,803 | −7,244 | **31** |
| `fijo_v1` | 23,434 | −23,039 | 57 |

Pasan los tres criterios, con márgenes iguales o mayores que a 300 s:

| Frente a | Demora | Recompensa v1 |
|---|---|---|
| `fijo_2_3` | +24.0% (t p = 2e-11, Wilcoxon p = 2e-6, 20/20 semillas) | +21.1% (p = 2e-5, 18/20 semillas) |
| `min_verde_y_cambiar` | +43.7% | +29.9% |

En un episodio largo, "cambiar apenas se pueda" ni siquiera mantiene la red al día: acumula hasta
31 vehículos fuera de la red. El ciclo corto pierde demasiada capacidad en amarillos. Que la
demanda no sea trivial no depende de la duración del episodio.

## 4. Reproducir

```
# Red (el mismo generador de la v1, con 4 x 1)
netgenerate --grid true --grid.x-number 4 --grid.y-number 1 --grid.length 200 --grid.attach-length 200 \
    --default.lanenumber 1 --default.speed 13.89 --tls.set A0,B0,C0,D0 \
    --output-file environments/four-intersection-corridor/four-intersection-corridor.net.xml

# Demanda oficial (FINAL_CANDIDATE = "it5")
python scripts/v2/corridor_demand.py

# Validación de cualquier candidata (unos 14 min por candidata a 300 s, con 10 procesos)
python scripts/v2/validate_corridor_demand.py --candidate it5 [--seconds 900]

# Test de humo
pytest tests/test_four_intersections.py -v
```

## 5. Costo computacional

Medición secuencial, sin otros procesos de simulación en paralelo. Son 20 episodios por
configuración, intercalados, con acciones aleatorias en la misma máquina de la v1. Se reporta la
mediana.

| Configuración | Episodio | Con la creación del entorno |
|---|---|---|
| v1: `TrafficEnvironment`, 300 s (60 pasos, con estado y recompensa) | 0.98 s | 1.59 s |
| v2: corredor, sumo-rl multiagente, 300 s | 1.54 s | 2.14 s |
| v2 + `CustomStateBuilder` de la v1 en los 4 semáforos, 300 s | 1.71 s | 2.31 s |
| v2: corredor, 900 s | 5.01 s | 5.61 s |

- **Un episodio de la v2 cuesta unas 1.75 veces uno de la v1** (1.71 s frente a 0.98 s), o 1.45
  veces si se cuenta la creación del entorno. Esta incluye dos arranques de SUMO por entorno
  (sumo-rl abre uno solo para leer la red) y pesa lo mismo en las dos versiones. La aproximación
  del estado (las lecturas de la v1 repetidas en las 4 señales) es una cota inferior: el estado,
  la acción y la recompensa reales de la Fase 1 pueden costar más.
- Un paso de control de la v2 (las 4 decisiones a la vez) cuesta unos 28 ms de simulación, frente
  a unos 16 ms en la v1. Ejemplo: los 39,208 pasos del RL directo de 30k de la v1 pasarían de unos
  10.5 a unos 18.5 minutos de simulación por semilla, sin contar PPO.
- Pasar a episodios de 900 s cuesta unas 3.3 veces más por episodio, algo más que el factor 3,
  porque hay más vehículos en la red.
- La validación completa de una candidata (41 políticas × 20 semillas, 300 s, con métricas
  segundo a segundo) tarda unos 14 minutos con 10 procesos, y unos 44 minutos a 900 s.

## 6. Pendiente para la Fase 1 (no se decide aquí)

- Vector de estado, acción conjunta y recompensa de la v2.
- Si el estado incluye el tiempo del episodio (ver 3.4, pulsos alineados con el inicio).
- Duración del episodio: 300 s, como en la v1, o más larga. La demanda vale para las dos.
- Semillas de evaluación de la v2, distintas de las de calibración (11000–11019).
