# Pre-registro: el sueño corregido (v2.1-0)

**Escrito el 5 de octubre de 2026, antes de entrenar ningún controlador con el sueño corregido y
antes de evaluar nada en 24000–24023.** Es un experimento **descriptivo**: no decide la arquitectura
oficial ni cambia ningún resultado publicado de la Fase 3 (`ADDENDUM_CONTROL.md`).

## 1. Pregunta

`CorridorDreamEnvironment` tenía un desfase en la ventana de acciones (`ADDENDUM_PLANIFICACION.md`,
sección 11). Los 20 PPO del sueño de la Fase 3 se entrenaron con él (`window_alignment="legacy"`).
¿Cambia su comportamiento en SUMO real si se entrenan en el sueño corregido
(`window_alignment="aligned"`), sobre todo el bloqueo y los catastróficos de la LSTM?

**Solo cambia un factor: la alineación de la ventana.**

## 2. Qué se entrena

- **20 PPO del sueño:** LSTM y Transformer, semillas 0–9, con
  `CorridorDreamEnvironment(window_alignment="aligned")`.
- **Todo lo demás, idéntico a la Fase 3:**
  - `ControllerConfig()` y la misma función `ppo_config`;
  - 50,000 pasos pedidos (50,176 hechos);
  - `dream_max_steps` = 7 y el recorte [−345.43, 0.0];
  - el modelo del mundo de la misma semilla y arquitectura (`arch_comparison/<arq>_raw_s<i>`);
  - las ventanas semilla del train;
  - la semilla de PPO *i*.
- **Selección del checkpoint en SUMO real con 21000–21004**, las mismas de la Fase 3, no
  24000–24004, para que la selección no sea un segundo factor.
- **Salidas:** `models/checkpoints/v2/control_aligned/dream_<arq>_s<i>/`. Los 40 controladores de
  `models/checkpoints/v2/control/` no se tocan: el script de entrenamiento se niega a escribir en
  esa carpeta sin `--overwrite-official`, y los md5 se comparan antes y después.
- `training/train_controller_v2.py --window-alignment aligned`; el valor queda en `run_info.json`.
- **4 procesos**, uno por corrida, reanudable. Solo con el equipo enchufado y más de 2 GB
  disponibles.

## 3. Qué se evalúa

- **Los 40 controladores del sueño** (los 20 antiguos de la Fase 3 y los 20 corregidos), **todos
  en `validation_v21` (24000–24023)**, una sola evaluación por controlador, con acuerdo con las
  referencias en los mismos estados (`--reference-agreement`).
- No se usan 25000–25047, 23000–23029 ni ningún otro split.

## 4. Métricas (fijadas ahora)

Por brazo (arquitectura × antiguo/corregido):

- **Retorno:** media y mediana del total, y media por intersección (A0, B0, C0, D0).
- **Catastróficos:** episodios con retorno < **−3,600** (el umbral de la Fase 3).
- **Bloqueo**, con las reglas de la Fase 3:
  - controlador "bloqueado": media de menos de 3 cambios de fase por episodio en algún semáforo;
  - episodio "bloqueado": menos de 3 cambios en algún semáforo.
- **Cambios de fase** medios por semáforo.
- **Acuerdo con cada referencia** por semáforo, en los mismos estados.
- **Comparación por arquitectura, corregido − antiguo:** Welch sobre las 10 medias por semilla, con
  IC al 95%. Una sola comparación por arquitectura y **sin corrección**: es descriptiva.

## 5. Cómo se leerá (escrito antes de ver el resultado)

Para cada arquitectura, el cambio se considera **"importante"** si, en validation_v21, se cumple
**al menos uno** de estos criterios:

1. **Episodios bloqueados:** los corregidos tienen **menos de la mitad** que los antiguos, siempre
   que los antiguos tengan al menos 10 (con menos, la mitad es ruido).
2. **Catastróficos:** los corregidos tienen **menos de la mitad** que los antiguos, con el mismo
   mínimo de 10.
3. **Retorno:** el IC al 95% de Welch de (corregido − antiguo) **excluye 0 y es positivo**.

Si no se cumple ninguno, el cambio **no** es importante. Eso incluye que los corregidos empeoren;
en ese caso se reporta el empeoramiento, pero no cuenta como cambio importante para la regla de
`plan_ppo` (`ADDENDUM_PLANIFICACION.md`, sección 13).

**Lectura prevista:**

- **Importante en la LSTM:** el desfase contribuía al bloqueo, y los PPO de la Fase 3 no
  representan lo que el sueño correcto produce.
- **No importante:** el bloqueo no viene del desfase. Sería consistente con el diagnóstico 12.1:
  el punto ciego persiste con el sueño alineado.
- **En ningún caso** se reescriben los resultados de la Fase 3, que quedan como "entrenados con el
  sueño con defecto".

## 6. Interacciones reales

Cada PPO corregido consume 3,000 pasos reales de selección, como en la Fase 3, más el dataset
compartido. La evaluación en validation_v21 (24 × 60 = 1,440 pasos por controlador) es
diagnóstica y no forma parte de la construcción de ningún controlador.
