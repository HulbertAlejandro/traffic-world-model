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

## 7. Fase 1: offset aleatorio del pulso y re-validación de la demanda

### 7.1 Decisión: el offset del pulso se sortea con la semilla de cada episodio

Con la demanda it5 tal cual, el pulso arranca siempre igual al inicio del episodio. Un controlador
que pudiera inferir en qué momento del episodio está podría memorizar el patrón en vez de
reaccionar al tráfico. Por eso cada episodio arranca el patrón en una fase sorteada con **su
propia semilla de SUMO**, sin una semilla aparte (`scripts/v2/corridor_demand.py`,
`pulse_offset`):

```
offset = numpy.random.default_rng(seed).integers(0, P)        P = 300 s (las dos mitades de it5)
demanda en el instante t = demanda del patrón en (t + offset) mod P
```

El rango es [0, 300) y no [0, 150): el patrón completo dura 300 s, y con [0, 150) el episodio
arrancaría siempre dentro de la mitad con pico hacia el Este. Con `offset = 0`, el generador
reproduce el archivo oficial de rutas carácter por carácter (salvo los finales de línea). Lo
verifiqué además en 2,000 instantes aleatorios: la demanda en t es la del patrón en
(t + offset) mod 300.

### 7.2 it5 con offset aleatorio no pasa el tercer criterio

Se usó el mismo protocolo de la sección 3.1 (20 semillas de calibración, 300 s, las mismas 41
políticas), con `--random-offset`
(`docs/results/v2/demand_calibration/it5_offset_300s.json`):

| Criterio | it5, offset 0 | it5, offset aleatorio |
|---|---|---|
| Sin cola invisible | ✅ 4 | ✅ 4 |
| Demora | ✅ +20.2% | ✅ +15.7% (t p = 7e-4, W p = 1e-3) |
| Recompensa v1 | ✅ +23.9% (p = 2e-4) | ❌ +5.5% (p = 0.44) |

En los dos casos la mejor trivial es `min_verde_y_cambiar`.

### 7.3 Iteraciones it6–it9: pulsos más marcados, misma demanda media

Todas con offset aleatorio. La demanda media es la de it5 (arterial 600/400; C0 350/250; A0, B0
y D0 constantes): solo cambia cuánto se separan las dos mitades. Siguiendo la indicación del
autor, se probó primero subir el contraste del pulso y no la carga total, que ya había fallado en
it3.

| Iteración | Mitad Este (arterial O→E / E→O; C0 N/S) | Mitad Oeste/C0 | Demora | Recompensa v1 | Pend. | ¿Pasa? |
|---|---|---|---|---|---|---|
| it6 | 850/200; 150/100 | 350/600; 550/400 | +17.4% (p = 6e-4) | +3.5% (p = 0.67) | 4 | No |
| it7 | 950/150; 100/80 | 250/650; 600/420 | +20.9% (p = 2e-5) | +10.3% (p = 0.076) | 5 | No |
| it8 | 1050/100; 50/40 | 150/700; 650/460 | +19.0% (p = 0.007) | **−26.7%** (p = 0.23) | **8** | No |
| it9 | 950/150; 50/40 | 250/650; 650/460 | +21.6% (p = 0.001) | −6.2% (p = 0.75) | **16** | No |

La mejora es la de `cola_mas_larga` frente a la mejor trivial, que es `min_verde_y_cambiar` en
todos los casos. Subir el contraste no arregla el criterio de la recompensa, y en it8 e it9 lo
empeora y vuelve a saturar las entradas.

### 7.4 Mecanismo: por qué "cola más larga" pierde con la recompensa al estilo de la v1

Se re-simularon las 20 semillas de calibración con `cola_mas_larga` y `min_verde_y_cambiar`, paso a
paso y por semáforo (`scripts/v2/analyze_reference_failures.py`, resultados en
`analysis_reference_failures_*.json`). Las trazas reproducen exactamente la diferencia de
recompensa que midió el validador en todas las semillas (error 0.0).

Brecha media por episodio, referencia menos trivial (> 0: la referencia es mejor):

| Candidata | A0 | B0 | C0 | D0 | Total |
|---|---|---|---|---|---|
| it5, offset 0 (pasa) | +305 | +160 | **−263** | +108 | +310 |
| it5, offset aleatorio | +160 | +64 | **−327** | +180 | +77 |
| it7 | +124 | +87 | **−270** | +215 | +155 |
| it8 | +30 | +138 | **−524** | −81 | −437 |
| it9 | +77 | +139 | **−554** | +242 | −96 |

1. **No depende del offset.** Los episodios extremos aparecen con cualquier offset, y la misma
   semilla no es mala de forma consistente entre candidatas: son realizaciones concretas del
   tráfico.
2. **No es el borde del pulso, salvo con contraste extremo.** Alineada con el tiempo del patrón,
   la brecha de it5–it7 no se acumula después de los bordes (0 y 150). En it8 e it9 sí aparece un
   efecto de borde, pero solo en C0: al final del pico de C0 y justo después (tramos 240–300 y
   0–30 del patrón), cuando llega la oleada hacia el Este con la cola transversal de C0 todavía
   sin despejar.
3. **Es C0, en todas las candidatas y en todos los tramos del patrón, también con offset 0.**
   Entre el 85% y el 90% de la pérdida en C0 viene del término de espera, no del de detenidos.
   El término de espera (`lane.getWaitingTime`) suma, en cada paso, el tiempo que cada vehículo
   lleva detenido sin moverse, así que castiga las paradas largas más que proporcionalmente.
   `cola_mas_larga` decide por número de detenidos e ignora cuánto lleva esperando cada uno:
   cambia de fase cuando la otra cola ya la superó en número, y para entonces esos vehículos
   acumularon paradas largas. La parada individual máxima en C0 es de 13–19 s con la referencia
   y de 9 s con la trivial.
4. **C0 es la única intersección con las dos fases muy cargadas a la vez** (el pico transversal
   coincide con el de la arterial hacia el Oeste). Con carga alta y equilibrada, alternar rápido y
   en partes iguales es casi lo correcto para una recompensa basada en espera. En B0 y D0, con
   transversales casi vacías, la referencia gana manteniendo el verde de la arterial (verdes
   medios de 9–10 pasos, hasta 50).
5. **El resultado neto es lo que gana en A0, B0 y D0 menos lo que pierde en C0.** Con offset 0, la
   mitad oeste/C0 caía siempre con la red ya llena, donde las ganancias en A0 y B0 son mayores.
   Al aleatorizar el offset esas ganancias se reducen a la mitad, la pérdida en C0 no cambia, y
   unas pocas paradas largas puntuales (28–42 s en una intersección) disparan la varianza.

**Conclusión.** La demanda no se volvió trivial: en demora, la referencia gana en todas las
candidatas (+15.7% a +21.6%, p ≤ 0.007). Lo que falla es el tercer criterio, y por cómo se
construyó: `cola_mas_larga` es una mala sonda de margen para una recompensa basada en espera,
justo en la intersección donde está ese margen. El defecto en C0 ya existía en la Fase 0; it5 pasó
porque las ganancias en otras intersecciones lo compensaban con la alineación del offset 0.

**Papel de `cola_mas_larga` desde aquí.** Sigue en el conjunto de referencias y sigue siendo
válida para el criterio de demora, que mide vehículos detenidos y donde ignorar la antigüedad de
la espera no la penaliza. No se usa como sonda decisiva del criterio de la recompensa al estilo de
la v1, por el mecanismo de los puntos 3 y 4.

### 7.5 Pre-registro: referencia "espera más larga" (escrito antes de correrla)

Fijado el 2 de octubre de 2026, a las 20:25 (hora local), sobre el commit `3609e89` más los
cambios sin commitear de esta fase. **No se ha corrido sobre ninguna semilla.**

**Definición de `espera_mas_larga`.** En cada paso de control y en cada semáforo s, para cada
fase verde p ∈ {0, 1}:

```
puntaje(p) = Σ  ( α · lane.getWaitingTime(l)  +  β · lane.getLastStepHaltingNumber(l) )
             l ∈ carriles de entrada de s con verde (G o g) en la fase p, cada carril una vez
```

- α y β son los coeficientes `alpha` y `beta` de `CorridorRewardConfig`, leídos de la
  configuración por defecto (α = β = 1). Son exactamente los dos términos, y los mismos pesos, de
  la recompensa de la v2, restringidos a los carriles que sirve cada fase. Se leen en el momento
  de decidir, es decir, con el estado que dejó el paso anterior.
- En esta red cada carril de entrada tiene todos sus movimientos en verde en una sola fase: la
  fase 0 suma los carriles Norte y Sur, y la fase 1 los Este y Oeste.
- **Decisión:** se pide la fase con el mayor puntaje.
- **Empate:** si el puntaje de la fase actual es igual al máximo (en particular, si los dos son 0)
  se mantiene la fase actual. Es la misma regla de desempate de `cola_mas_larga` y `max_presion`
  (`_choose` en el validador).
- Como en todas las políticas, un cambio pedido antes de los 7 s (`yellow_time + min_green`) lo
  ignora sumo-rl.

**Alcance.** Se aplica **solo a it5**, sin tocar su demanda. No se aplica a it6–it9 ni a ninguna
otra candidata en esta ronda.

**Protocolo.** El de la sección 3.1 sin otro cambio que este: las mismas 20 semillas de
calibración (11000–11019), offset aleatorio (`--random-offset`), 300 s, las mismas políticas
triviales (incluida la grilla 6 × 6 de tiempo fijo) y los mismos umbrales. El conjunto de
referencias pasa a ser {`cola_mas_larga`, `max_presion`, `espera_mas_larga`}, y cada criterio se
evalúa con la mejor referencia **para su propia métrica**:

1. **Sin cola invisible:** como máximo 5 vehículos pendientes de inserción después de los 30 s,
   en todos los episodios, con **cada** referencia que se use para cumplir el criterio 2 o el 3.
2. **Demora:** la referencia con menor demora media le gana a cada política trivial por ≥ 10%,
   con p < 0.01 en la t pareada y en el Wilcoxon pareado.
3. **Recompensa al estilo de la v1:** la referencia con mejor recompensa media le gana a cada
   política trivial, con p < 0.01 en la t pareada.

En la Fase 0 la mejor referencia se elegía por demora y se usaba para los dos criterios; elegirla
por la métrica de cada criterio es parte de este cambio de protocolo. Se reporta además, como
dato informativo y no decisivo, si `espera_mas_larga` sola pasa los tres criterios.

**Criterio de decisión.**

- **Si it5 pasa los tres criterios, se adopta definitivamente como la demanda de la v2,** con el
  offset aleatorio de 7.1.
- **Si no pasa, se detiene el trabajo y se reporta.** En esta ronda no se prueba una
  "espera más larga mejorada", ni otra referencia, ni ningún ajuste de la demanda.

### 7.6 Resultado

**it5 no pasa. Según el criterio de decisión de 7.5, el trabajo se detiene aquí** y no se adopta
it5 como la demanda de la v2. Se corrió con el protocolo de 7.5 (`--random-offset
--reward-aligned-reference`, `docs/results/v2/demand_calibration/it5_offset_espera_300s.json`).
La sección 7.5 tiene el mismo SHA-256 antes y después de la corrida (`7a2a3e85…`).

| Criterio | Mejor referencia para la métrica | Resultado |
|---|---|---|
| Sin cola invisible | `cola_mas_larga` (las dos métricas) | ✅ 4 |
| Demora | `cola_mas_larga` | ✅ +15.7% frente a `min_verde_y_cambiar` (t p = 7e-4, W p = 1e-3, 17/20 semillas) |
| Recompensa v1 | `cola_mas_larga` | ❌ +5.5% frente a `min_verde_y_cambiar` (t p = 0.44) |

`espera_mas_larga` no mejora a `cola_mas_larga` en la recompensa: obtiene −1,404 de media, frente
a −1,317 de `cola_mas_larga` y −1,394 de `min_verde_y_cambiar`. Sola, frente a
`min_verde_y_cambiar`, logra −0.7% en recompensa (p = 0.95) y +14.9% en demora. Gana en la
mediana de los episodios (+116, 12/20 semillas), pero tiene un episodio de −2,494.

Controles: `cola_mas_larga` y `min_verde_y_cambiar` reproducen exactamente sus valores de la
corrida de 7.2. `espera_mas_larga` es una política distinta de `cola_mas_larga`: coinciden en todas
las decisiones en 6 de los 20 episodios y difieren en los otros 14.

Que una regla codiciosa sobre los propios términos de la recompensa tampoco le gane a alternar
rápido refuta la explicación de 7.4, punto 3, como causa suficiente: no basta con que la
referencia mire la espera en vez de la cola. Queda como hipótesis **no probada** que, en C0, una
regla que decide por el estado actual cambia tarde por construcción, sea cual sea el puntaje.
En esta ronda no se prueba nada más, como se fijó en 7.5.

### 7.7 Criterio 3 desagregado por intersección (sin simulaciones nuevas)

El autor pidió recalcular el criterio 3 por separado en cada intersección: ¿la mejor referencia
le gana a la mejor política trivial con p < 0.01? Su hipótesis, escrita antes de ver el dato, era
que A0, B0 y D0 pasarían con claridad y C0 no pasaría con ninguna referencia. Se acordó que, si el
dato era otro, no se adoptaría nada.

**Fuente de los datos.** El validador guarda la recompensa solo agregada por episodio. La
recompensa por intersección sale de las trazas de 7.4 (`analysis_reference_failures_it5_it6_it7.json`,
it5 con offset aleatorio), que cubren `cola_mas_larga` y `min_verde_y_cambiar`. Esas dos son la
mejor referencia y la mejor trivial por recompensa en la corrida de 7.6, y las trazas reproducen
exactamente la diferencia de recompensa de esa corrida en las 20 semillas (error 0.0). Para
`espera_mas_larga` y `max_presion` no hay recompensa por intersección sin simular de nuevo.

**Recompensa al estilo de la v1 por intersección,** `cola_mas_larga` menos `min_verde_y_cambiar`
(> 0: la referencia es mejor), 20 semillas, t pareada y Wilcoxon pareado:

| Intersección | Media | Mediana | Gana | t p | W p | Peor episodio | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| A0 | +160.4 | +179 | 17/20 | 1e-4 | 8e-4 | −102 | ✅ |
| B0 | +63.9 | +63 | 14/20 | 0.33 | 0.11 | −717 | ❌ |
| C0 | **−327.4** | −259 | 3/20 | 4.5e-4 | 4.5e-4 | −1,282 | ❌ (la referencia es significativamente **peor**) |
| D0 | +179.7 | +141 | 17/20 | 4e-4 | 4e-4 | −147 | ✅ |

**Demora por intersección** (vehículos·s detenidos en los carriles de entrada,
`min_verde_y_cambiar` menos la referencia, > 0: la referencia es mejor), con las tres
referencias. Estos datos sí están en el JSON del validador:

| Referencia | A0 | B0 | C0 | D0 |
|---|---|---|---|---|
| `cola_mas_larga` | +201 (p = 7e-6) | +117 (p = 0.02) | −284 (p = 5e-5) | +215 (p = 2e-5) |
| `espera_mas_larga` | +205 (p = 8e-6) | +128 (p = 0.01) | −319 (p = 5e-5) | +219 (p = 8e-6) |
| `max_presion` | +88 (p = 0.01) | +18 (p = 0.7) | −381 (p = 6e-6) | +93 (p = 0.04) |

**Resultado: la hipótesis se confirma solo en parte, y no se adopta nada.**

- **A0 y D0 pasan con claridad.**
- **C0 no pasa, y es más que eso:** la referencia es significativamente peor que alternar rápido
  (3/20 semillas). En demora, las tres referencias pierden en C0 (p ≤ 5e-5).
- **B0 no pasa.** La ventaja es positiva en la media y la mediana (14/20 semillas), pero no es
  significativa (p = 0.33). Hay un episodio de −717 (semilla 11013, la de la parada de 30 s en B0
  de 7.4). En demora, B0 tampoco llega a p < 0.01 con ninguna referencia (la mejor, p = 0.01).

Por lo acordado, la conclusión de "cumplido en 3 de 4 intersecciones, con C0 como caso límite
esperado" **no se redacta ni se adopta**: el dato es 2 de 4. La decisión queda pendiente del
autor.

### 7.8 B0 y la semilla 11013 (con los datos guardados, sin simulaciones nuevas)

Las trazas de 7.4 guardaron agregados por semilla e intersección (brechas, términos de espera y
de detenidos, parada individual máxima), no la serie paso a paso.

**La semilla 11013:**

- **Es el episodio más cargado de los 20.** Entraron 204 vehículos con `min_verde_y_cambiar`,
  frente a una media de 174.3 y una desviación de 14.4 (rango 149–204).
- **Momento del patrón (offset 29):** mitad con pico hacia el Este (arterial 750/250) hasta
  t = 121 s, mitad oeste/C0 (450/550) hasta t = 271 s, y otra vez la mitad Este. La transversal
  de B0 tiene 40/30 veh/h constantes.

| En B0 | `cola_mas_larga` | `espera_mas_larga` | `min_verde_y_cambiar` |
|---|---|---|---|
| Brecha de recompensa frente a la trivial | −717 (−665 de espera, −52 de detenidos) | sin dato por intersección | — |
| Parada individual máxima | 30 s | sin dato | 9 s |
| Demora (vehículos·s detenidos) | 687 | 562 | 430 |

**Las paradas largas en B0 son recurrentes, no exclusivas de esa semilla.** Con
`cola_mas_larga`, la parada individual máxima en B0 llega a 26, 30 y 33 s en 3 de las 20
semillas. Con `min_verde_y_cambiar` nunca pasa de 9 s.

**B0 sin la semilla 11013 (dato informativo, no adoptado):**

| | 20 semillas | 19 semillas |
|---|---|---|
| Media | +63.9 | +105.0 |
| Mediana | — | +74 |
| Gana | 14/20 | 14/19 |
| t p | 0.33 | 0.055 |
| Wilcoxon p | 0.11 | 0.032 |
| Peor episodio | −717 | −371 |

Quitando cualquier semilla, la mejor p de B0 es 0.055. Con 19 semillas, A0 (p = 2e-4) y D0
(p = 9e-4) siguen pasando y C0 sigue siendo peor para la referencia (p = 3e-4). **No es ruido de
una sola semilla:** la ventaja en B0 es real en la mediana, pero las paradas largas recurrentes la
vuelven no significativa con la recompensa al estilo de la v1.

### 7.9 Traza paso a paso de la semilla 11013 en B0 (re-simulación puntual autorizada)

El autor autorizó re-simular solo este episodio, que es determinista: it5, semilla 11013,
offset 29. Se corrió dos veces:

1. `scripts/v2/trace_episode_signal.py` con `cola_mas_larga` y `espera_mas_larga` (resultado en
   `trace_it5_seed11013_B0.json`). Registra en B0, en cada paso, lo que ve cada regla al decidir,
   sus puntajes y la parada más larga por carril.
2. El mismo episodio con `cola_mas_larga` y registro de los carriles de salida de B0, la fase de
   A0 y el vehículo de cabeza del acceso desde C0 (`spillback_check_seed11013.py`, resultado en
   `trace_it5_seed11013_B0_spillback.json`).

**Control:** la recompensa total del episodio reproduce exactamente la del validador (−3,107 con
`cola_mas_larga` y −2,100 con `espera_mas_larga`) en las dos corridas.

**El evento de 30 s (t = 165–200 s, mitad oeste/C0 del patrón).** B0 tiene **verde para la
arterial durante todo el evento**:

- **La arterial está detenida a pesar del verde.** El vehículo de cabeza del acceso desde C0 (un
  vehículo que cruza todo el corredor hacia el Oeste) está detenido a 3.5 m de la línea de
  detención, a velocidad 0, desde t ≈ 165 hasta t ≈ 195 s. Detrás se forma una cola de hasta 7
  vehículos con 83 s de espera acumulada en el carril.
- **No es un bloqueo desde A0.** El carril de salida B0→A0 tiene 0 o 1 vehículos durante todo el
  evento y ninguno detenido; los otros carriles de salida tampoco están llenos. La causa de esa
  detención no queda identificada con lo registrado. Determinarla requeriría registrar los
  carriles internos de la intersección; no se hizo.
- **La transversal espera.** Un vehículo del acceso sur llega hacia t ≈ 170 s y espera 30 s; uno
  del acceso norte espera 15 s.
- **Las dos reglas mantienen el verde de la arterial.** En cada decisión entre t = 170 y 195 s,
  el puntaje de la arterial supera al de la transversal: 2 a 7 detenidos contra 1 a 2 con
  `cola_mas_larga`, y 14 a 90 contra 6 a 37 con `espera_mas_larga`. En t = 200 s la cola de la
  arterial arranca (el de cabeza a 5.25 m/s), quedan 1 a 2 detenidos, y las dos reglas cambian a
  la transversal.

**Conclusión sobre la hipótesis del punto 2.** Se confirma en su forma: la transversal nunca supera
a la arterial en número, así que la regla no le da paso durante 30 s. Pero con una precisión
importante: **los vehículos de la arterial que la regla "servía" no se movían a pesar del verde.**
Ese verde se desperdició. Una regla que solo mira el estado actual del semáforo no distingue una
cola que avanza de una cola bloqueada: las dos suman detenidos y espera. Alternar rápido habría
dado paso a la transversal en ≤ 10 s.

**¿`espera_mas_larga` evita este patrón en B0?** Solo en parte:

- **En el evento de 30 s, no.** Toma exactamente las mismas decisiones que `cola_mas_larga`,
  porque los vehículos bloqueados de la arterial también acumulan espera y su puntaje sigue siendo
  mayor.
- **Evita un segundo evento** que `cola_mas_larga` sí sufre: paradas de 28 y 24 s en la
  transversal hacia t = 245 s. `espera_mas_larga` cambia a la transversal en t = 230 s. En todo el
  episodio, las dos reglas difieren en solo 7 de 60 pasos en B0, todos desde t = 230 s.
- **En este episodio**, la recompensa en B0 es −782 con `espera_mas_larga` y −1,116 con
  `cola_mas_larga`.
- **En las 20 semillas** solo existe la demora por intersección (7.7): en B0, +128 (p = 0.01) con
  `espera_mas_larga` y +117 (p = 0.02) con `cola_mas_larga`, sin una diferencia clara. La
  recompensa por intersección de `espera_mas_larga` en las 20 semillas no está disponible sin
  simular de nuevo.

No se corrió nada más que este episodio.

### 7.10 Causa del vehículo detenido con verde en B0 (misma re-simulación puntual, registro por segundo)

El autor pidió determinar si la detención de 7.9 es un problema de la simulación o de las
políticas. Se re-simuló **el mismo episodio**: it5, semilla 11013, offset 29, `cola_mas_larga`.
Script y resultado: `junction_check_seed11013.py` y `.json`. Entre t = 150 y t = 210 s se
registró, segundo a segundo:

- los carriles internos de B0, con cada vehículo, su posición y su velocidad;
- los cuatro accesos;
- el estado del semáforo;
- para el vehículo de cabeza: el estado del semáforo para su movimiento (`getNextTLS`), su líder
  y sus conflictos en la intersección (`getJunctionFoes`);
- la tabla de movimientos de cada acceso (`lane.getLinks`).

**Control:** la recompensa total reproduce la del validador (−3,107).

**El semáforo es coherente con sumo-rl.** B0 está en verde para la arterial
(`rrrrGGggrrrrGGgg`) hasta t = 200 s, en amarillo en t = 201–202 s y en verde para la
transversal (`GGggrrrrGGggrrrr`) desde t = 203 s. No hay ninguna discrepancia entre la fase que
registra sumo-rl y la que aplica SUMO.

**Causa: un giro a la izquierda permisivo que cede el paso dentro de la intersección y bloquea
el carril compartido.**

1. El vehículo detenido (`east_through_flow_s0.3`) sigue **recto** hacia el Oeste; su movimiento
   tiene `G` durante todo el evento. Está parado a 3.5 m de la línea de detención porque su
   **líder** está delante, dentro de la intersección.
2. El líder (`east_to_B0_south_flow_s0.1`) **gira a la izquierda**, desde el acceso de C0 hacia
   el sur. Entra al carril interno `:B0_6_0` en t = 158 s y queda **detenido en el punto de
   espera interno del giro desde t = 163 hasta t = 192 s (30 s)**.
3. Ese movimiento tiene **`g` minúscula** en la fase de la arterial: es un verde permisivo y debe
   ceder el paso. Lo mismo vale para el giro a la izquierda del acceso opuesto. Los rectos y los
   giros a la derecha tienen `G` (prioridad).
4. Cede el paso al tráfico que viene de frente, hacia el Este desde A0: rectos que cruzan hacia
   las 170, 172 y 191 s, y giros a la derecha hacia el sur, que van al **mismo carril de salida**
   que el giro a la izquierda, hacia las 167 y 188 s.
5. Cada acceso tiene **un solo carril** que lleva los cuatro movimientos: derecha, recto,
   izquierda y retorno, según `lane.getLinks`. No hay carril de giro ni carril vecino. Mientras el
   que gira espera dentro del cruce, nadie detrás de él puede avanzar, aunque tenga verde. En
   t = 193 s el giro termina y el vehículo recto arranca.

**Clasificación: (a), comportamiento esperado de SUMO,** que no se había considerado al diseñar
la recompensa ni las políticas. **No es (b), un error de configuración.**

- Las fases son las que construye sumo-rl a partir del programa de `netgenerate`, y SUMO las
  aplica tal cual.
- El giro a la izquierda permisivo en un carril compartido es una decisión de diseño heredada de
  la v1 y anotada en la sección 1: un carril por brazo, sin carril de giro y sin fase protegida
  para la izquierda.
- Que el que gira bloquee a los que siguen recto en un solo carril es el comportamiento normal de
  esa geometría, también en la realidad.

**Lo que no queda explicado.** Entre t ≈ 172 y t ≈ 188 s, el siguiente vehículo de frente se
acercaba desde 184 m hasta 15 m a unos 11 m/s: un hueco aparente de unos 16 s. El que giraba no
lo aprovechó. Que lo rechace depende del modelo de aceptación de huecos de SUMO para un vehículo
detenido dentro de la intersección (cómo registra a los vehículos que se acercan y qué margen
exige). Con lo registrado no se puede determinar qué regla concreta lo llevó a esperar, y no se
corrió nada más para averiguarlo.

**Consecuencia para las políticas (hipótesis consistente con la traza, no probada).** Un giro a
la izquierda permisivo que espera dentro de la intersección se libera cuando la arterial pasa a
rojo, porque el tráfico de frente se detiene. Una política que cambia de fase con frecuencia
vacía esos giros a menudo y evita que bloqueen el carril. Una regla que mantiene el verde mientras
cuenta vehículos detenidos en la arterial (`cola_mas_larga`, `espera_mas_larga`) **prolonga el
bloqueo**: los vehículos que cuenta están detenidos detrás del que gira, y el verde que les da no
los hace avanzar. Esto explicaría la ventaja de `min_verde_y_cambiar` con la recompensa basada en
espera, y es plausible que también actúe en C0, donde el tráfico de frente es más denso en las dos
mitades. Para C0 no se verificó: no se re-simuló ningún episodio de C0.

**Confiabilidad de los resultados con B0.** SUMO está haciendo lo que su modelo indica; no hay
indicios de un error de simulación. Pero los resultados de B0, y probablemente de C0, dependen de
un fenómeno estructural de la red (el bloqueo del carril compartido por giros a la izquierda
permisivos) que favorece a las políticas que cambian con frecuencia. Si eso es aceptable o la red
necesita giros protegidos o carriles de giro es una decisión de diseño de la Fase 0, pendiente del
autor.

### 7.11 Cierre de la calibración: se adopta it5 con el criterio 3 desagregado

**Decisión del autor (2 de octubre de 2026):** la red **no se cambia**. Los carriles compartidos
con giro a la izquierda sin protección son comunes en el tráfico real, y cambiarlos invalidaría
toda la calibración. Se adopta **it5, con offset aleatorio del pulso (7.1), como la demanda
definitiva de la v2**, con el criterio 3 evaluado por intersección.

**Esto se aparta del pre-registro de 7.5, y se deja escrito.** 7.5 fijó que, si it5 no pasaba los
tres criterios, se detendría el trabajo sin más ajustes, y no los pasó. El criterio desagregado se
adopta **después de ver los datos** de 7.6–7.10, como decisión explícita del autor y no como
resultado de un criterio fijado de antemano. Cualquier cita de esta validación debe decirlo así.

**Verificación informativa en C0** (`blocked_left_turn_check.py` y `.json`). Se re-simularon solo
los tres episodios en los que `cola_mas_larga` más pierde en C0: semillas 11013, 11011 y 11016,
con las dos políticas, seis corridas en total. Las seis reproducen exactamente la recompensa del
validador. Se midieron:

- los vehículos·s detenidos **dentro** de la intersección (giros esperando un hueco);
- los vehículos·s **parados en un acceso con verde** cuyo líder es un vehículo detenido dentro del
  cruce: el mecanismo de B0 de 7.10.

| Semilla | Política | Detenidos dentro de C0 | Parados con verde detrás de un giro en C0 | Espera interna más larga en C0 |
|---|---|---|---|---|
| 11013 | `cola_mas_larga` | 51 | **170** | 24 s |
| 11013 | `min_verde_y_cambiar` | 17 | 3 | 4 s |
| 11011 | `cola_mas_larga` | 61 | **144** | 26 s |
| 11011 | `min_verde_y_cambiar` | 15 | 5 | 6 s |
| 11016 | `cola_mas_larga` | 64 | **100** | 18 s |
| 11016 | `min_verde_y_cambiar` | 21 | 0 | 8 s |

En los tres peores episodios de C0 aparece el mismo fenómeno que en B0, y de forma marcada. Con
`cola_mas_larga` hay entre 100 y 170 vehículos·s parados con verde detrás de un giro que espera
dentro del cruce; con la trivial, entre 0 y 5. En 11013 aparece además en B0 (161 contra 5). No se
calculó qué fracción de la pérdida de recompensa en C0 explica: el término de espera crece más que
proporcionalmente con la duración de cada parada, así que no se reparte de forma lineal. Son 3
episodios elegidos por ser los peores, no una muestra representativa.

**Conclusión.**

1. **El mecanismo.** Un giro a la izquierda permisivo espera un hueco dentro de la intersección y
   bloquea el carril compartido. Los vehículos detrás tienen verde pero no avanzan. Una regla que
   decide con el conteo o la espera de los **carriles de entrada** ve una cola grande en la
   arterial y le mantiene el verde, que no la hace avanzar. Las cuatro reglas probadas
   (`cola_mas_larga`, `espera_mas_larga`, `max_presion` y la grilla de tiempo fijo) caen en esto o
   no lo pueden aprovechar. Cambiar de fase con frecuencia lo evita sin "saberlo": el rojo detiene
   el tráfico de frente y libera al que gira.
2. **Precisión importante.** La información para distinguir una cola que avanza de una bloqueada
   **sí existe en el estado instantáneo de SUMO** (un vehículo detenido en el carril interno del
   giro). Lo que no la tiene es **el estado de la v1**, que solo lee los carriles de entrada y es
   el que usa la v2 (`DISENO_ESTADO_ACCION_RECOMPENSA.md`). Con ese estado, la señal disponible es
   **temporal**: una cola que no baja mientras su fase tiene verde. Eso es lo que un controlador
   con memoria (o el World Model, que predice la dinámica) puede aprovechar y una regla miope no.
   Que lo aproveche de verdad es una **hipótesis** de la v2, no un resultado.
3. **La causa de que B0 y C0 no pasen.** Con la evidencia reunida (la traza completa de B0 y tres
   episodios de C0), este mecanismo es la causa principal identificada, no la intensidad de la
   demanda. Ya se había visto que subir la carga (it3) o el contraste del pulso (it6–it9) no lo
   arreglaba. No se demostró que sea la **única** causa.

**El criterio 3 desagregado** (7.7): `cola_mas_larga` frente a `min_verde_y_cambiar`, recompensa
al estilo de la v1 por intersección, 20 semillas.

| Intersección | Demanda transversal (N/S, veh/h) | Resultado | Lectura |
|---|---|---|---|
| A0 | 200/120 | ✅ +160 (p = 1e-4) | La referencia gana. **No** es por baja demanda transversal, que en A0 es la segunda más alta. En los tres episodios de C0 el fenómeno aparece poco en A0 (0–18 vehículos·s). |
| B0 | 40/30 | ❌ +64 (p = 0.33) | Aparece el fenómeno (7.10, 7.11). Gana en la mediana. |
| C0 | 350/250 de media, con pico | ❌ −327 (p = 4.5e-4) | Aparece de forma marcada (7.11). |
| D0 | 40/30 | ✅ +180 (p = 4e-4) | La referencia gana. |

La explicación propuesta de por qué A0 y D0 pasan (que el bloqueo aparece poco ahí) **no se
verificó con todas las semillas**. D0 tiene la misma demanda transversal que B0 y sí pasa. La
diferencia más probable es la arterial: B0 y C0 están en el interior del corredor y reciben tráfico
de frente en las dos direcciones desde intersecciones vecinas, pero esto no se midió.

**Lo que se adopta, exactamente.** El criterio 3 queda como **"cumplido en 2 de 4 intersecciones
(A0, D0); en B0 y C0 no se cumple por el bloqueo de giros a la izquierda permisivos en carril
compartido, documentado en 7.10–7.11"**. Los criterios 1 (sin cola invisible) y 2 (demora: +15.7%,
p = 7e-4) se cumplen en el agregado con offset aleatorio (7.2). B0 y C0 son las intersecciones
donde, según esta hipótesis, un controlador aprendido tiene margen frente a las reglas miopes. La
evaluación de la v2 debe reportar los resultados también por intersección, para poder contrastar
esa hipótesis.

**Cierre.** Con esto quedan cerradas la Fase 0 y los Pasos 0 y 1 de la Fase 1 (offset aleatorio).
`FINAL_CANDIDATE` sigue siendo `"it5"`. El archivo de rutas oficial (offset 0) sigue siendo la
demanda de la Fase 0, para reproducir; el entorno de la v2 (`CorridorTrafficEnvironment`) usa it5
con el offset sorteado por semilla.
