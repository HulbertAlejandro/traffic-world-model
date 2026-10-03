# v2 — Pre-registro del dataset del corredor de 4 intersecciones

Escrito y commiteado **antes de recolectar ningún episodio**, ni siquiera la muestra piloto. La
recolección la hace `scripts/v2/collect_dataset_v2.py`, cuyas constantes repiten lo que fija este
documento. Si el código y el documento no coinciden, vale el documento y el código se corrige
antes de recolectar.

## 1. Entorno

- `CorridorTrafficEnvironment` (`environments/corridor_environment.py`): estado de 104
  dimensiones, acción conjunta mantener/cambiar en 4 semáforos y recompensa al estilo de la v1
  sumada sobre los 4 (`docs/v2/DISENO_ESTADO_ACCION_RECOMPENSA.md`).
- Demanda it5 con el offset del pulso sorteado con la semilla de cada episodio
  (`docs/v2/DISENO_RED_4_INTERSECCIONES.md`, 7.1 y 7.11).
- **Duración del episodio: 300 s (60 pasos de 5 s), sin cambios.** Es la de la Fase 0 y la de la
  calibración. Además, 300 s es exactamente un período del patrón de it5: con cualquier offset,
  cada episodio recorre una vez cada mitad del pulso, así que todos los episodios tienen la misma
  demanda total esperada. No hay una razón de peso para cambiarla.

## 2. Semillas (sin solapes)

| Uso | Semillas | Episodios |
|---|---|---|
| Calibración de la Fase 0 (no se usan aquí) | 11000–11019 | — |
| Tests (no se usan aquí) | 7, 19900, 19901 (y 11000, que es de calibración) | — |
| Muestra piloto (se inspecciona y se descarta) | 19000–19017 | 18 |
| `train` | 20000–20111 | 112 |
| `validation` | 21000–21023 | 24 |
| `test` | 22000–22023 | 24 |
| `ood` (reservado) | 23000–23029 | 30 |

La semilla de cada episodio fija tres cosas: la semilla de SUMO, el offset del pulso y, en la
política aleatoria, su generador de acciones (sección 3). Una recolección nueva con el mismo
código da el mismo dataset. En la v1 no era así, porque las acciones aleatorias usaban el
generador global de numpy sin semilla.

**El conjunto `ood` está reservado desde ahora.** No se usa para ninguna decisión de diseño,
selección de modelo, ajuste de hiperparámetros ni normalización, hasta la evaluación final. Cumple
el papel que en la v1 cumplieron los escenarios 7000–7029, que allí se agregaron después; aquí se
reservan desde el principio. "Fuera de distribución" significa **semillas nunca vistas**, no otra
demanda: tiene la misma demanda, la misma mezcla de políticas y el mismo protocolo que los demás
splits.

## 3. Políticas de recolección

Son tres (la v1 usó una sola, acciones aleatorias). Todas están en `scripts/v2/corridor_policies.py`
con la codificación mantener/cambiar.

| Política | Qué hace | Para qué está |
|---|---|---|
| `aleatoria` | Mantener o cambiar con probabilidad 1/2 en cada semáforo y cada paso, con un generador sembrado con la semilla del episodio | Cobertura de acciones: todas las combinaciones de decisiones en todos los estados, como el dataset de la v1 |
| `fijo_2_3` | Tiempo fijo, 2 pasos en la fase transversal y 3 en la arterial, el mismo programa en los 4 semáforos | El mejor tiempo fijo uniforme de la Fase 0 (it5, offset 0). Trayectorias de un controlador estructurado y en lazo abierto |
| `cola_mas_larga` | La referencia de la Fase 0: fase verde con más vehículos detenidos en sus carriles de entrada | Trayectorias de un controlador que mira el estado, con verdes largos en la arterial y los episodios de bloqueo por giros a la izquierda de 7.10–7.11, el régimen donde se espera margen |

`fijo_2_3` y `cola_mas_larga` llaman a las mismas funciones del validador de la Fase 0 y traducen
su salida a mantener/cambiar. Un test verifica que, con offset 0, reproducen exactamente el
retorno medido en la Fase 0.

Una nota sobre `fijo_2_3`: con offset aleatorio, el mejor tiempo fijo uniforme por demora pasa a
ser `fijo_2_2`, que equivale exactamente a `min_verde_y_cambiar` (7.2). Se mantiene `fijo_2_3`
porque es el que fijó el autor ("el mejor ciclo encontrado en la Fase 0"), y porque el régimen de
cambio frecuente ya lo cubre la política aleatoria, que pide un cambio en la mitad de los pasos.

**Asignación de políticas a episodios.** Dentro de cada split se reparten en orden según la
posición de la semilla: `(semilla − inicio del split) mod 3` → `aleatoria`, `fijo_2_3`,
`cola_mas_larga`.

| Split | `aleatoria` | `fijo_2_3` | `cola_mas_larga` | Total |
|---|---|---|---|---|
| `train` | 38 | 37 | 37 | 112 |
| `validation` | 8 | 8 | 8 | 24 |
| `test` | 8 | 8 | 8 | 24 |
| `ood` | 10 | 10 | 10 | 30 |

**Criterio:** la misma mezcla en todos los splits, para que `validation`, `test` y `ood` midan la
misma distribución con la que se entrena, y partes iguales porque no hay una razón previa para
favorecer a una política. La asignación depende solo de la posición de la semilla, no de nada
observado.

## 4. Formato

- Un archivo por episodio: `datasets/v2/raw/<split>/episode_<semilla>.npz`, con **las mismas 8
  claves de la v1**:

| Clave | Forma | Tipo | Nota |
|---|---|---|---|
| `states` | (60, 104) | float32 | estado antes de la acción |
| `actions` | (60, 4) | int64 | mantener (0) / cambiar (1) por semáforo, A0–D0 |
| `rewards` | (60,) | float32 | recompensa de la v2 |
| `next_states` | (60, 104) | float32 | estado después de la acción |
| `terminated` | (60,) | bool | siempre False |
| `truncated` | (60,) | bool | True solo en el último paso |
| `episode_id` | (60,) | int64 | la semilla del episodio, única entre splits |
| `time_step` | (60,) | int64 | 0–59 |

- Se reutilizan sin modificar `scripts/merge_dataset.merge_files`, que une los episodios de cada
  split, y `scripts/normalize_dataset.normalize_dataset`, cuyo escalador se ajusta **solo con
  `train`**. El split ya viene dado por la carpeta, así que no se usa el reparto aleatorio de
  `scripts/split_dataset.py`. Las únicas diferencias con la v1 son las formas de `states` y
  `actions`, que esas funciones no suponen.
- **Los `.npz` no se commitean** (CLAUDE.md); `.gitignore` excluye `datasets/v2/`. Se versiona un
  manifiesto, `docs/results/v2/dataset/manifest.json`, con la semilla, el split, la política, el
  offset, el retorno, el tiempo medido y el SHA-256 de cada archivo. Con él, el dataset se puede
  verificar y regenerar.

## 5. Muestra piloto (antes de la recolección completa)

Semillas 19000–19017, 6 por política, en `datasets/v2/pilot/`, con su propio manifiesto. Se
recolecta, se inspecciona y no se usa para nada más. **Si falla cualquiera de estos chequeos, no
se recolecta el dataset completo y se reporta:**

1. Los 18 episodios tienen exactamente 60 transiciones; `truncated` es True solo en la última y
   `terminated` es siempre False.
2. Ningún NaN ni Inf en `states`, `next_states` ni `rewards`.
3. Las formas y los tipos son los de la tabla de la sección 4, y las acciones están en {0, 1}.
4. Todas las recompensas son ≤ 0.
5. `next_states[t]` es igual a `states[t+1]` en todos los pasos.
6. **Escala de la recompensa coherente con la Fase 0.** El retorno medio de `fijo_2_3` y de
   `cola_mas_larga` en la piloto está dentro de ±50% de su media en la validación de it5 con offset
   aleatorio (`it5_offset_300s.json`: −1,715 y −1,317). Son semillas distintas, así que se espera
   cierta diferencia; ±50% solo detecta errores de escala o de signo.
7. **Sin cola invisible** con `cola_mas_larga`: como máximo 5 vehículos pendientes de inserción
   después de los 30 s. Se registra por episodio el máximo de pendientes de todo el episodio.

**Columnas constantes.** Con la piloto se listan las columnas del estado que son constantes en los
18 episodios. Se espera que lo sean las 2 posiciones amarillas de `phase_one_hot` de cada semáforo
(8 columnas), como en la v1. **Se mantienen las 104 columnas, como en la v1:** `normalize_dataset`
ya trata las de desviación cero, y quitarlas cambiaría el formato. Quedan documentadas como posible
redundancia, para revisarla después.

## 6. Costo

La Fase 0 estimó 1.71 s por episodio de 300 s con el estado de la v1 leído en los 4 semáforos, sin
contar la creación del entorno. El costo real se mide por episodio, incluida la generación del
archivo de rutas de cada semilla. **Si la media de la recolección completa supera 2.6 s por
episodio (1.5 veces la estimación), se reporta como sustancialmente mayor.**

## 7. Lo que este pre-registro prohíbe

- Cambiar semillas, splits, políticas, asignación o duración después de ver los datos.
- Usar `ood` para cualquier decisión antes de la evaluación final.
- Descartar episodios del dataset completo. Si alguno falla un chequeo, se reporta y se decide
  aparte, no en silencio.
