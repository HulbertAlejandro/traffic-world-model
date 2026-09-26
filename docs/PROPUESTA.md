# Propuesta de Proyecto de Grado

*Documento reescrito para reflejar el proyecto tal como fue ejecutado (estado del repositorio hasta el commit `42e5eb8`, incluida la auditoría técnica del 26 de septiembre), no solo como fue planteado originalmente.*

---

PROPUESTA DE PROYECTO DE GRADO

Control inteligente de semáforos mediante World Models para la reducción de congestión vehicular en entornos de simulación SUMO

Ingeniería de Sistemas

Universidad del Quindío

*Documento guía de formulación y desarrollo — Versión final, post-ejecución*

2026

> **Nota sobre esta versión.** Este documento fue reescrito por completo después de que el núcleo del proyecto quedara implementado, entrenado y evaluado. Se conserva la estructura de secciones de las versiones anteriores (incluida la auditoría tecnológica curricular de la Sección 30), pero el contenido de cada sección describe lo que **realmente se construyó** y los **resultados realmente obtenidos**, no solo lo que se planeaba al inicio. Donde el proyecto terminó divergiendo de lo planteado originalmente, se explica el porqué en el lugar correspondiente, con la misma honestidad con la que se documentó cada hallazgo durante el desarrollo.

# 1. Propuesta de título

Control inteligente de semáforos mediante World Models para la reducción de congestión vehicular en entornos de simulación SUMO

# 2. Resumen

Este proyecto diseñó e implementó un sistema basado en World Models para aprender la dinámica de una intersección de tráfico urbano simulada en SUMO, y utilizó el modelo aprendido como un entorno interno (el "Dream Environment") para entrenar un controlador semafórico **sin que este interactuara con SUMO durante su entrenamiento**. Se comparó este enfoque contra un controlador entrenado directamente contra el simulador (RL directo), y contra dos baselines clásicos: control de tiempo fijo y una regla determinista simple.

El sistema completo se implementó con los siguientes componentes, todos funcionales y verificados: una intersección propia en SUMO con demanda de tráfico asimétrica (diseñada específicamente para que ninguna solución trivial fuera óptima), una representación vectorial del estado (26 dimensiones), un Autoencoder determinista que comprime esa representación a 16 dimensiones, un modelo temporal (LSTM) que predice tanto el siguiente estado latente como la recompensa, un Dream Environment que permite entrenar un controlador PPO sin tocar el simulador, y el mismo controlador entrenado también de forma directa contra SUMO para la comparación.

**Resultado principal, con evidencia estadística**: frente a un RL directo entrenado con 10,000 pasos (13,240 interacciones reales por semilla), el controlador entrenado dentro del World Model controla mejor en SUMO real, tanto en los 30 escenarios de evaluación usados durante todo el proyecto (-326.79 frente a -454.51, media de 3 y 4 semillas de entrenamiento) como en 30 escenarios nuevos (-293.81 frente a -439.73). Lo hace con entre 2.9 y 8.5 veces menos interacciones reales con el simulador, según cómo se contabilice (1.7x a 5.0x si el dataset no se amortiza entre semillas). Frente a un RL directo con el triple de presupuesto (39,208 interacciones por semilla), **no se detectó una diferencia consistente**: el World Model es mejor en los escenarios de evaluación (+75.34; p = 0.029 a nivel de semilla y 0.020 pareado por escenario) pero no en los nuevos (pareado p = 0.54). La ventaja demostrada del World Model es, por tanto, de **eficiencia**: iguala o supera al RL directo con una fracción del contacto con el simulador. Además, es el método con menos episodios catastróficos. Estos resultados corresponden al RL directo reentrenado tras corregir un bug de integración encontrado en una auditoría técnica del repositorio (Sección 20.2).

Como extensión académica opcional, se contempló el reemplazo del modelo temporal recurrente por un Transformer y por un TSMixer. Ambas extensiones se pospusieron mientras se resolvían varios problemas críticos del núcleo (detallados en la Sección 20), cuya resolución era condición necesaria para que cualquier resultado del proyecto fuera confiable, y se ejecutaron después, una vez estabilizado el núcleo (Experimento 3, Sección 20.1). **Resultado: se mantiene la LSTM**, que predice la recompensa mejor que ambas alternativas en los 10 horizontes evaluados.

Cada componente del sistema se implementó en su forma más simple suficiente para el problema, priorizando las técnicas cubiertas en el curso sobre extensiones del paper original de World Models no vistas en clase (ver Sección 30).

# 3. Planteamiento del problema

El control de semáforos es un problema de decisión secuencial en el que las acciones afectan la formación de colas, los tiempos de espera, la velocidad y el flujo de vehículos. Los métodos tradicionales de Reinforcement Learning pueden requerir numerosas interacciones con el simulador para aprender una política útil. Un World Model ofrece una alternativa: aprender primero una aproximación de la dinámica del entorno y utilizarla posteriormente para entrenar un controlador sin consultar continuamente el entorno real.

El reto central, confirmado durante el desarrollo, no fue solo si un modelo aprendido podía generar predicciones útiles — sí pudo, con evidencia (Experimento 1) — sino **construir correctamente toda la cadena de piezas que traducen esas predicciones en control real**. La mayor parte del esfuerzo del proyecto terminó dedicado a encontrar y corregir puntos donde esa cadena se rompía silenciosamente (una lectura de estado desconectada de la realidad, un puente de normalización incompleto, un criterio de selección de modelo sin relación con el desempeño real), no al diseño conceptual del World Model en sí, que resultó relativamente directo de implementar.

# 4. Pregunta de investigación

¿Puede un World Model aprender de forma suficientemente precisa la dinámica de una intersección de tráfico simulada en SUMO y utilizar sus predicciones para entrenar un controlador semafórico, reduciendo las interacciones necesarias con el simulador sin deteriorar significativamente el desempeño del control?

**Respuesta obtenida**: sí. Con un presupuesto de RL directo de 13,240 interacciones reales por semilla, el controlador entrenado dentro del World Model lo supera en SUMO real usando entre 2.9 y 8.5 veces menos interacciones (según la contabilidad del dataset), tanto en los escenarios de evaluación del proyecto como en escenarios nuevos. Con el triple de presupuesto, el RL directo deja de distinguirse de forma consistente del World Model: hay brecha a favor del World Model en unos escenarios y no en otros. La respuesta precisa es que el World Model **reduce las interacciones necesarias sin deteriorar el desempeño**, tal como plantea la pregunta, sin que eso implique mejor control cuando el RL directo dispone de presupuesto de sobra.

# 5. Objetivo general

Diseñar e implementar un sistema basado en World Models que aprenda la dinámica del tráfico urbano y entrene un agente de control de semáforos dentro de un modelo del entorno, evaluando su desempeño frente al entrenamiento tradicional en SUMO.

**Cumplido en su totalidad.**

# 6. Objetivos específicos

- Construir y configurar una intersección de tráfico en SUMO que permita generar episodios reproducibles y controlar sus fases semafóricas mediante TraCI. **Cumplido.**
- Definir una representación del estado del tráfico, un espacio de acciones y una función de recompensa orientados a reducir congestión y tiempos de espera. **Cumplido** — estado de 26 dimensiones, recompensa `R_t = -α·espera - β·cola + γ·flujo - δ·cambio_de_fase`.
- Recolectar trayectorias de interacción con SUMO y construir un conjunto de datos de transiciones para el aprendizaje de la dinámica. **Cumplido** — 80 episodios, con semilla de tráfico distinta por episodio.
- Implementar y evaluar un Autoencoder que permita obtener una representación latente compacta del estado, cuando esta compresión resulte justificada mediante una comparación explícita contra el vector de estado crudo. **Cumplido** — Experimento 0: con el mismo protocolo en las dos ramas, 5 semillas por rama y entrenamiento hasta convergencia, el Autoencoder mejora la predicción de la recompensa de forma moderada y mayoritaria (37 de 50 pares semilla × horizonte, mejora mediana de 10.9%), y se mantuvo en el sistema final (Sección 20.3).
- Implementar un modelo temporal de World Model que prediga el siguiente estado latente y la recompensa a partir del estado, las acciones y el historial temporal, usando una arquitectura determinística. **Cumplido** — LSTM con capa densa de salida, implementado con `torch.nn.LSTM` estándar de PyTorch (no con ecuaciones de compuertas escritas manualmente, como se planteaba originalmente — una simplificación de implementación que no cambia el comportamiento matemático del modelo).
- Evaluar la capacidad predictiva del World Model a uno y varios pasos, cuantificando el error acumulado. **Cumplido** — Experimento 1, superando al baseline persistente en los 10 horizontes evaluados.
- Construir un mecanismo de imaginación o un Dream Environment simplificado para evaluar consecuencias hipotéticas de acciones antes de ejecutarlas en SUMO. **Cumplido, con un diseño distinto al inicialmente conceptualizado**: en vez de un mecanismo de planificación por acciones candidatas evaluadas antes de cada paso real, se implementó como un entorno completo de entrenamiento (compatible con Gymnasium) sobre el cual se entrena un controlador PPO de principio a fin, sin tocar SUMO. Cumple el mismo propósito de fondo (aprender sin interacción real) con una arquitectura más simple de integrar con Stable-Baselines3.
- Implementar un controlador de Reinforcement Learning mediante Stable-Baselines3 y compararlo con un controlador que interactúe directamente con SUMO. **Cumplido**, con 3 semillas de entrenamiento para el controlador del World Model y 4 para el RL directo, evaluadas en los escenarios de evaluación del proyecto y en escenarios nuevos.
- Analizar si el enfoque basado en World Models reduce el número de interacciones o el tiempo de entrenamiento manteniendo un desempeño competitivo. **Cumplido** — ver Sección 2 para el resultado.
- Como extensión opcional, comparar el modelo temporal recurrente (LSTM) con un Transformer y con un TSMixer. **Cumplido** — Experimento 3, ejecutado después de estabilizar el núcleo: la LSTM se mantiene, porque gana en `reward_mse` en los 10 horizontes evaluados frente a ambas alternativas, con una reducción mediana de 38.1% frente al Transformer y de 56.1% frente a TSMixer (ver Sección 20.1 y PROJECT_STATUS.md, sección "Experimento 3").

# 7. Justificación

- Combina Inteligencia Artificial, Deep Learning y Reinforcement Learning en un problema aplicado.
- Aborda un problema relevante: la reducción de congestión, tiempos de espera y formación de colas.
- SUMO permite trabajar con un entorno de simulación reproducible y controlable — condición que resultó esencial durante el proyecto, ya que la reproducibilidad exacta (semillas fijas, hiperparámetros persistidos) fue la herramienta principal para diagnosticar cada uno de los problemas encontrados en el camino.
- Las métricas de evaluación son cuantificables y permiten comparaciones objetivas, incluyendo pruebas de significancia estadística formal sobre los resultados finales.
- El enfoque World Model permitió estudiar una cuestión académica concreta y obtener una respuesta con evidencia: cuánto puede sustituirse la interacción directa con el simulador por predicciones de un modelo aprendido.
- Cada bloque conceptual del proyecto (redes neuronales, modelos generativos, espacio latente, Autoencoder, RNN/LSTM) corresponde a un tema efectivamente cubierto en el curso (ver Sección 30.1).

# 8. Fundamento conceptual: adaptación de World Models

En la formulación clásica de World Models, el agente aprende una representación interna del entorno, aprende su dinámica temporal y utiliza esa representación para tomar decisiones. En este proyecto se conservó esa filosofía, adaptando la representación a un entorno numérico (no visual).

Arquitectura conceptual, tal como quedó implementada:

*SUMO → Estado del tráfico (26 dims) → Autoencoder → z (16 dims) → LSTM → (ẑ_{t+1}, r̂_{t+1}) → Dream Environment → Controlador PPO → Acción → SUMO*

La representación latente `z` es una versión comprimida y determinista del estado (no un VAE con muestreo probabilístico, como se contemplaba inicialmente — ver Sección 30.4 para el criterio que decidió esto). El modelo temporal recibe una ventana de historia de `z` y acciones, y predice tanto el siguiente `z` como la recompensa asociada. El Dream Environment usa esas predicciones para entrenar el controlador sin ejecutar SUMO.

# 9. Arquitectura implementada

| Componente | Entrada | Salida | Propósito | Estado |
| --- | --- | --- | --- | --- |
| SUMO + TraCI | Red + flujo + acción | Estado del tráfico | Simular la intersección y ejecutar acciones | ✅ Implementado |
| Autoencoder | Estado vectorial (26 dims) | z (16 dims) | Comprimir el estado; se mantiene tras confirmarse su utilidad (Experimento 0) | ✅ Implementado |
| Modelo temporal (LSTM) | Secuencia de (z, acción) | (ẑ siguiente, r̂ siguiente) | Aprender la dinámica del tráfico en el espacio latente | ✅ Implementado. Transformer/TSMixer: implementados y evaluados, no reemplazan a la LSTM (Experimento 3) |
| Dream Environment | Episodios codificados + acciones del agente | Transiciones imaginadas | Entrenar el controlador sin consultar SUMO en cada paso | ✅ Implementado como entorno completo de entrenamiento (Gymnasium) |
| Controller (PPO) | z (del Dream Environment o de SUMO real vía el Encoder) | Acción semafórica | Seleccionar la acción de control | ✅ Implementado, con verificación de 3 semillas |

# 10. SUMO y TraCI

SUMO (Simulation of Urban MObility) es el simulador utilizado. TraCI controla la simulación desde Python, consulta variables del tráfico y modifica el estado del semáforo. La implementación se mantuvo, como se planteó, limitada a una única intersección — decisión de alcance que se sostuvo durante todo el proyecto (ver Sección 24) y que resultó acertada dado el tiempo real que exigió investigar y corregir los problemas descritos en la Sección 20.

# 11. Estado del tráfico

El estado implementado es un vector numérico de **26 dimensiones**: para cada uno de los 4 carriles de entrada (número de vehículos, longitud de cola, tiempo de espera acumulado, velocidad media, ocupación — 20 valores en total), más la fase actual del semáforo codificada como *one-hot* (4 valores, uno por fase posible) y el tiempo transcurrido/restante de esa fase (2 valores).

Esta cifra difiere de la estimación inicial de la propuesta ("11 a 20 dimensiones aproximadamente") porque, durante el desarrollo, se decidió codificar la fase del semáforo como *one-hot* en vez de como un único valor escalar — una decisión de diseño que evita que el modelo interprete un orden numérico inexistente entre fases, a costa de más dimensiones. El criterio de la Sección 30.4 (comparar contra ~262,000 dimensiones de una imagen para justificar la necesidad de compresión) sigue siendo válido con 26 dimensiones exactamente igual que con 11-20: la compresión no se asume necesaria por el tamaño del vector, se decide empíricamente (Experimento 0).

# 12. Espacio de acciones

Implementado como un espacio discreto de 2 acciones, tal como se planteó. Sin embargo, la semántica real difiere de la descripción original: la propuesta definía `0 = mantener la fase actual` y `1 = cambiar a la siguiente fase`, pero la librería de integración con SUMO (`sumo-rl`) interpreta el número recibido como **el índice de la fase verde de destino**, no como un interruptor de mantener/cambiar. Con exactamente 2 fases posibles en esta intersección, el efecto práctico coincide con la descripción original solo cuando la fase actual es la 0 (acción 1 = cambiar, acción 0 = mantener); cuando la fase actual es la 1 se invierte (acción 0 = cambiar, acción 1 = mantener). Como cada fase ocupa aproximadamente el 50% del tiempo en las trayectorias medidas (PROJECT_STATUS.md, "Notas técnicas" de la sección del controlador PPO contra SUMO real), la coincidencia con la descripción original ("acción 1 = cambiar") se da en aproximadamente la mitad de los pasos, no en la mayoría — una diferencia que quedó documentada como deuda técnica menor, sin corregir, porque todo el pipeline (recolección de datos, entrenamiento, evaluación) usa la convención de forma consistente entre sí.

Las restricciones de operación recomendadas (tiempo mínimo de verde, tiempo mínimo entre cambios) se cumplen automáticamente por la configuración de `sumo-rl`, sin necesidad de código adicional del proyecto.

# 13. Función de recompensa

Implementada exactamente como se planteó:

*R_t = − α·WaitingTime_t − β·QueueLength_t + γ·Throughput_t − δ·SignalChanges_t*

con coeficientes α = β = γ = 1.0, δ = 0.1. La intención del término de flujo (`Throughput_t`) era contar los vehículos que **completaron** su recorrido en cada paso, no los presentes en el carril (que premiaría la congestión). La auditoría técnica encontró que la implementación no lo lograba: `getArrivedNumber()` de TraCI solo cuenta las llegadas del último segundo simulado, y un paso de control simula 5, así que el término medido equivale a ~13% de las llegadas reales. Con γ = 1 su peso es de ~0.2 por paso frente a una recompensa media de -43, de modo que no altera ninguna comparación de recompensa. El término se mantuvo sin cambios, porque todos los datos y modelos se entrenaron con él. Para medir el flujo vehicular se agregó una métrica correcta (llegadas del intervalo completo), que no forma parte de la recompensa.

# 14. Dataset y generación de datos

El proyecto no dependió de un dataset de imágenes ni de una base de datos externa (ver Sección 28 para la nota sobre esto). La base de datos es un conjunto de trayectorias generadas directamente en SUMO: 80 episodios de 60 pasos cada uno, almacenados como `(s_t, a_t, r_t, s_{t+1})` y, tras la compresión, como `(z_t, a_t, r_t, z_{t+1})`.

Los splits de entrenamiento, validación y prueba se separan **por episodio completo**, nunca por transición suelta dentro del mismo episodio, con verificación automatizada de que no hay fuga de datos entre conjuntos. Cada episodio de recolección usa una semilla de SUMO distinta (una corrección aplicada durante el desarrollo: en una primera versión, todos los episodios de recolección compartían la misma semilla de tráfico, y la única variedad venía de las acciones aleatorias, no del tráfico en sí).

# 15. World Model

El modelo temporal se implementó como una LSTM con una capa densa de salida que predice directamente `ẑ_{t+1}` y `r̂_{t+1}` (predicción determinística, pérdida MSE), siguiendo el patrón LSTM → Dense → ŷ del material del curso. A diferencia de lo planteado originalmente, la implementación usa el módulo `torch.nn.LSTM` estándar de PyTorch en vez de replicar manualmente las ecuaciones de compuertas (olvido, entrada, salida) — una decisión práctica de implementación que no cambia el comportamiento matemático del modelo (las ecuaciones internas de `torch.nn.LSTM` son exactamente las mismas compuertas vistas en el curso), solo la forma en que se escribió el código.

Conceptualmente:

*(z_t, a_t, historial) → (ẑ_{t+1}, r̂_{t+1})*

Como alternativas al LSTM se implementaron y evaluaron un Transformer y un TSMixer (Experimento 3, Sección 20.1), ambos detrás de la misma interfaz común (`TemporalModel`, un contrato `Protocol` de Python), de modo que cualquiera de los dos podía sustituir al LSTM sin modificar el resto del sistema. **El LSTM se mantuvo**: tuvo menor error de predicción de recompensa (`reward_mse`) que ambas alternativas en los 10 horizontes evaluados. El Transformer dejó además un hallazgo: predice mejor el estado latente que el LSTM en los 10 horizontes (menor `latent_mse`), pero peor la recompensa, que es lo que el Dream Environment entrega al controlador. Una métrica mejor en entrenamiento no se tradujo en una mejor predicción de lo que realmente importa (ver PROJECT_STATUS.md, sección "Experimento 3").

**Resultado del Experimento 1** (evaluación predictiva, Sección 20): el LSTM superó a un baseline persistente ("nada cambia") en los 10 horizontes evaluados (1 a 10 pasos), con un error de predicción de recompensa a un paso equivalente al 2.7% del error de ese baseline.

# 16. Dream Environment e imaginación

El Dream Environment se implementó como una clase compatible con la interfaz estándar de Gymnasium (`DreamEnvironment`), que usa el LSTM ya entrenado para imaginar transiciones sin ejecutar SUMO. El diseño real difiere del concepto original de "evaluar varias acciones candidatas y ejecutar solo la seleccionada en SUMO": en cambio, se construyó como un entorno de entrenamiento completo sobre el cual un controlador PPO (Stable-Baselines3) aprende de punta a punta, íntegramente dentro de la imaginación del modelo. Ambos diseños cumplen el mismo propósito de fondo — reducir la dependencia de interacciones reales — pero el segundo es más simple de integrar directamente con una librería estándar de RL.

Cada episodio imaginado se siembra con una ventana real de contexto (los 16 pasos que el LSTM necesita como historia), tomada de un episodio real ya codificado — nunca empieza "de la nada". Un hallazgo importante durante el desarrollo: el modelo extrapola con menor confiabilidad cuando se le pide imaginar secuencias de la misma acción sostenidas por más pasos de los que vio durante su propio entrenamiento. Esto llevó a limitar la duración de cada episodio imaginado (`max_dream_steps=7`, calibrado empíricamente) y a recortar la recompensa imaginada a un rango estadísticamente razonable, calculado de los datos reales — dos mitigaciones que reducen, sin eliminar del todo, ese riesgo de extrapolación.

# 17. Controlador

Implementado con PPO (Proximal Policy Optimization) a través de Stable-Baselines3, tal como se planteó, descartando CMA-ES por las razones ya documentadas en la Sección 30.3. Se entrenaron y verificaron **dos controladores independientes**:

1. **PPO del sueño**: entrenado enteramente dentro del `DreamEnvironment`, sin ninguna interacción real con SUMO durante el aprendizaje. Su mejor checkpoint se selecciona evaluando periódicamente contra SUMO real (a través de un puente que traduce el estado real al espacio latente), no contra la recompensa imaginada — un cambio de diseño motivado por un hallazgo real durante el desarrollo (ver Sección 20).
2. **PPO directo**: entrenado directamente contra SUMO real, sin Autoencoder ni Dream Environment — el baseline de RL directo (Sección 18).

La auditoría técnica (Sección 20.2) encontró que el entrenamiento del PPO directo, que abre dos simulaciones en el mismo proceso (entrenamiento y evaluación periódica), leía parte de sus estados y recompensas de la simulación equivocada. Se corrigió y el PPO directo se reentrenó con 4 semillas de entrenamiento; el PPO del sueño, que usa una sola simulación, no estaba afectado. En cada método, el checkpoint "oficial" es la semilla con mejor resultado en los escenarios de evaluación (semilla 2 en el sueño y semilla 0 en el RL directo), pero los resultados siempre se reportan como media de todas las semillas.

Ambos controladores usan `VecNormalize` (una utilidad estándar de Stable-Baselines3) para normalizar la recompensa durante el entrenamiento; el controlador directo además normaliza sus observaciones, dado que el estado crudo tiene una escala mucho más desigual entre variables que el espacio latente `z`.

# 18. Baselines

Los tres baselines planteados originalmente se implementaron y evaluaron:

- **Control semafórico de tiempo fijo**: ciclo fijo de cambio de fase.
- **Agente de Reinforcement Learning que interactúa directamente con SUMO**: el PPO directo descrito en la Sección 17.
- **Método propuesto: World Model + controlador entrenado en el sueño.**

Se agregó, además, un cuarto punto de comparación no contemplado originalmente: una **regla determinista simple** ("pedir siempre la fase contraria a la actual"), que resultó ser la política óptima bajo la demanda de tráfico simétrica usada en una primera fase del proyecto — su inclusión fue clave para detectar que, bajo esa demanda, ningún método aprendido superaba a una heurística trivial, lo que motivó el rediseño del escenario de demanda descrito en la Sección 20.

# 19. Métricas de evaluación

## Para el tráfico (implementadas):

- Tiempo promedio de espera.
- Longitud promedio de las colas.
- Vehículos atendidos (llegadas a destino por intervalo de control). La métrica de "throughput" reportada inicialmente no medía llegadas (Sección 13); se reemplazó por una medición correcta, verificada contra un conteo independiente segundo a segundo.

## Para el tráfico (planteadas originalmente, no implementadas):

- Tiempo promedio de viaje, consumo de combustible y emisiones de CO₂ no se midieron — SUMO puede reportarlos, pero no se consideraron necesarios para responder la pregunta de investigación central, y priorizar su implementación no era condición necesaria frente a los problemas críticos descritos en la Sección 20.

## Para el aprendizaje (implementadas):

- Recompensa acumulada, con pruebas de significancia estadística sobre los resultados finales: t de Welch sobre las medias por semilla de entrenamiento (la prueba que dice algo sobre el método, con pocas semillas y por tanto baja potencia) y t pareada por escenario (los escenarios de evaluación son compartidos por todas las políticas). Las pruebas a nivel de episodio se reportan solo como referencia: 90 episodios de 3 políticas entrenadas no son 90 observaciones independientes del método.
- Número de interacciones reales con SUMO por método.
- Error de predicción a uno y varios pasos, y de la recompensa.
- `explained_variance` de la función de valor de PPO — una métrica no contemplada originalmente que resultó central en el diagnóstico de por qué los primeros controladores entrenados no funcionaban bien (Sección 20).
- Número de parámetros del modelo temporal, reportado para las tres arquitecturas comparadas en el Experimento 3 (LSTM 77,969; Transformer 269,585; TSMixer 10,519).

## Para el aprendizaje (planteadas originalmente, no implementadas):

- Tiempo de entrenamiento e inferencia del modelo temporal. En el Experimento 3 no se reportaron como métrica: la decisión entre arquitecturas se tomó solo por la predicción de recompensa, y ninguna alternativa la mejoró, así que su costo computacional no llegó a ser un factor de decisión.

# 20. Diseño experimental y cronología real del proyecto

**Experimento 0 — Necesidad del Autoencoder.** Se comparó el modelo temporal (LSTM) usando (a) el vector de estado normalizado crudo, y (b) el vector latente `z`. **Resultado final: (b) mejora la predicción de la recompensa de forma moderada y mayoritaria** (37 de 50 pares semilla × horizonte, mejora mediana de 10.9%, 4 de 5 semillas), y el Autoencoder se mantuvo en el sistema final. El resultado publicado antes (9 de 10 horizontes, 14.7%) venía de una sola semilla y de un protocolo de entrenamiento distinto entre las dos ramas; la Sección 20.3 explica cómo se corrigió.

**Experimento 1 — Capacidad predictiva.** El LSTM se evaluó sobre episodios no vistos, comparando predicción a uno y varios pasos contra un baseline persistente. **Resultado: el LSTM supera al baseline en los 10 horizontes evaluados.**

**Experimento 2 — Control por imaginación.** Se comparó el controlador entrenado sin World Model (RL directo) frente al que usa el modelo aprendido (entrenado en el sueño), ambos medidos en SUMO real. **Resultado: frente al RL directo con 10,000 pasos, el controlador del World Model es mejor en los escenarios de evaluación y en escenarios nuevos, con entre 2.9 y 8.5 veces menos interacciones reales. Frente al RL directo con el triple de presupuesto, no se detectó una diferencia consistente (Sección 20.2).**

**Experimento 3 — Comparación de arquitecturas temporales (Transformer, TSMixer).** **Pospuesto, y ejecutado después de estabilizar el núcleo (Sección 20.1).** No se ejecutó en su momento. La razón no fue falta de tiempo en abstracto, sino que, en el camino hacia el Experimento 2, aparecieron cuatro problemas críticos que exigían resolución antes de que cualquier resultado fuera confiable:

1. **Un bug en la lectura de la fase del semáforo**: el estado leía la fase desde una fuente de TraCI que la librería de control del semáforo nunca actualiza, dejando esa variable congelada durante toda la recolección de datos inicial. Se corrigió y se regeneró el dataset.
2. **Un bug de normalización en el puente entre SUMO y el controlador del sueño**: el estado real no se normalizaba con las mismas estadísticas que usó el Autoencoder durante su entrenamiento, invalidando silenciosamente las primeras evaluaciones contra SUMO real.
3. **Un criterio de selección de "mejor modelo" sin relación con el desempeño real**: se descubrió, con una investigación dedicada, que elegir el mejor checkpoint del controlador del sueño por su recompensa imaginada no predecía en absoluto su desempeño real (correlación de Pearson ≈ 0.08) — se cambió el criterio a evaluación periódica contra SUMO real.
4. **La función de valor de PPO no aprendía nada** (`explained_variance` ≈ 0) en ninguno de los dos controladores, por falta de normalización de la recompensa — corregido con `VecNormalize`.

Priorizar el diagnóstico y la corrección de estos cuatro problemas, cada uno con evidencia real antes y después del arreglo, se consideró más valioso académicamente que ejecutar el Experimento 3 con un núcleo potencialmente poco confiable. La interfaz para ejecutarlo más adelante (`TemporalModel`) quedó preparada, y es la que se usó cuando el experimento se retomó (Sección 20.1).

**Un quinto hallazgo no anticipado**: bajo la demanda de tráfico simétrica usada originalmente, tanto el controlador del sueño como el RL directo convergieron a la misma regla trivial ("cambiar de fase tan rápido como el reglamento de tiempos lo permite"), que resultó ser la política óptima bajo esa demanda — impidiendo cualquier comparación real de calidad de control entre métodos. Esto motivó diseñar una demanda de tráfico **asimétrica** (500 veh/h en la vía principal, 150 veh/h en la secundaria), calibrada específicamente para que esa regla trivial dejara de ser óptima, permitiendo así la comparación real reportada en la Sección 2.

## 20.1 Experimento 3, retomado y cerrado

Una vez estabilizado el núcleo (los cuatro problemas anteriores corregidos y el resultado del Experimento 2 verificado con 3 semillas por método), se retomó el Experimento 3.

**Qué se implementó.** Dos clases nuevas que implementan la interfaz `TemporalModel` y predicen las mismas dos salidas que la LSTM, `(ẑ_{t+1}, r̂_{t+1})`:

- `LatentDynamicsTransformer`: proyección lineal de `(z, a)`, codificación posicional sinusoidal y 2 capas de encoder Transformer estándar de PyTorch (269,585 parámetros).
- `LatentDynamicsTSMixer`: 2 bloques que alternan mezcla temporal y mezcla de variables, solo con capas densas, normalización y conexiones residuales, sin recurrencia ni atención (10,519 parámetros).

Un selector (`build_world_model`) construye la arquitectura correcta a partir de los hiperparámetros guardados junto a cada checkpoint, así que la evaluación y el Dream Environment no dependen de qué arquitectura se entrenó. Las dos alternativas se entrenaron con exactamente el mismo protocolo que la LSTM (mismo dataset latente, semilla, optimizador, épocas, early stopping y normalización de recompensa) y se evaluaron sobre el mismo conjunto de prueba y los mismos 10 horizontes. **La LSTM (77,969 parámetros) no se reentrenó**: se usó el checkpoint existente, el mismo que usa el controlador del sueño, verificado sin cambios antes y después.

**Criterio de decisión, fijado antes de ver resultados.** Un candidato reemplaza a la LSTM solo si (a) tiene el menor `reward_mse` en la mayoría de los horizontes, y (b) su reducción mediana de `reward_mse` frente a la LSTM es de al menos 7.3%, la mitad del 14.7% de reducción mediana que justificó mantener el Autoencoder en el Experimento 0. Se usa `reward_mse` porque la recompensa es lo que el Dream Environment entrega al controlador. *(Nota posterior: el 14.7% del Experimento 0 dejó de sostenerse al rehacerlo, y el valor corregido es 10.9% (Sección 20.3). La conclusión del Experimento 3 no cambia: la LSTM gana con reducciones del 38–56%, muy por encima de la mitad de cualquiera de los dos valores.)*

**Resultado.** La LSTM tiene el menor `reward_mse` en **los 10 horizontes**, con una reducción mediana de **38.1% frente al Transformer** y de **56.1% frente a TSMixer**. Ninguna alternativa se acerca al criterio: **se mantiene la LSTM**, porque predice mejor, y además porque ya está integrada y validada en todo el pipeline (Dream Environment, selección de checkpoint en SUMO real, controlador verificado con 3 semillas).

**Verificación de convergencia de TSMixer.** Con el protocolo compartido, la mejor época de TSMixer fue la penúltima de 100, lo que hacía sospechar que estaba subentrenado. Se repitió su entrenamiento con 300 épocas completas, sin early stopping: su pérdida de validación se estanca desde la época ~100 mientras la de entrenamiento sigue bajando (empieza a sobreajustar). Su mejor checkpoint de esa corrida mejora algo, pero la LSTM sigue ganando en los 10 horizontes (reducción mediana de 44.5%). **El resultado se mantiene.**

**Hallazgo: una métrica de entrenamiento que no predice la que importa.** Frente a la LSTM, el Transformer tiene **menor error de predicción del estado latente (`latent_mse`) en los 10 horizontes** y **menor pérdida de validación**, que es la métrica con la que se elige el mejor checkpoint. Sin embargo, su **`reward_mse` es peor en los 10 horizontes**: casi empata a un paso (+4%), la diferencia crece hasta duplicar el error de la LSTM en el horizonte 6, y luego se estrecha sin llegar a invertirse. Es el mismo patrón de fondo que el problema crítico 3 de esta sección, donde la recompensa imaginada del controlador del sueño tenía una correlación de apenas 0.08 con su recompensa real en SUMO. En los dos casos, **la métrica que se optimiza o con la que se selecciona durante el entrenamiento no predice la métrica que realmente importa**. Consecuencia práctica: si las arquitecturas se hubieran comparado por pérdida de validación, se habría elegido al Transformer, la opción equivocada. La comparación tiene que hacerse sobre la predicción de recompensa en rollouts de varios pasos, como se hizo.

## 20.2 RL directo: presupuesto, auditoría técnica y resultado final

**Verificación de presupuesto.** La comparación del Experimento 2 dejó una limitación declarada: el RL directo se entrenó con un presupuesto modesto (10,000 pasos reales de entrenamiento), y no se podía descartar que más presupuesto cerrara la brecha. Para responderlo, sus semillas se reentrenaron con **30,000 pasos**, cambiando solo el presupuesto.

**Auditoría técnica y bug de integración.** Una auditoría posterior del repositorio completo, ejecutando el código, encontró que el constructor del estado leía los carriles a través de la conexión global de TraCI, que apunta a la última simulación iniciada en el proceso, y no a la del propio entorno. El único entrenamiento que abre dos simulaciones a la vez es el del RL directo (entrenamiento y evaluación periódica). Después de cada evaluación periódica, su entrenamiento observaba, y recibía como recompensa, la simulación de evaluación, congelada, hasta el siguiente reinicio: un 3.7% de las transiciones con 10,000 pasos y un 4.0% con 30,000. Se corrigió con una verificación bit a bit: con una sola simulación, el código corregido reproduce exactamente los episodios del dataset. El RL directo se reentrenó con **4 semillas** para cada presupuesto. El dataset, el Autoencoder, el LSTM, el Dream Environment y el PPO del sueño no estaban afectados.

**Resultado final**, medido en SUMO real con un script de evaluación versionado que guarda cada episodio. Se evaluó en los 30 escenarios usados durante todo el proyecto y en 30 escenarios nuevos, nunca usados durante el desarrollo, porque los primeros sirvieron para tomar decisiones de diseño y no son un conjunto de prueba virgen:

| | Escenarios de evaluación | Escenarios nuevos | Episodios catastróficos (< -600) | Interacciones reales por semilla |
| --- | --- | --- | --- | --- |
| World Model (PPO del sueño, 3 semillas) | **-326.79** | **-293.81** | 5/90 y 1/90 | ~4,600 (7,800 sin amortizar el dataset) |
| RL directo, 10,000 pasos (4 semillas) | -454.51 | -439.73 | 25/120 y 29/120 | 13,240 |
| RL directo, 30,000 pasos (4 semillas) | -402.13 | -316.59 | 24/120 y 9/120 | 39,208 |
| Control de tiempo fijo | -411.27 | -391.70 | 0/30 y 0/30 | — |

**Pruebas estadísticas.** Frente al RL directo con 10,000 pasos, el World Model es mejor en los dos conjuntos: pareado por escenario, p = 0.0002 y p = 4.5e-9; a nivel de semilla, p = 0.003 en los escenarios de evaluación y p = 0.077 en los nuevos. Frente al de 30,000 pasos, hay brecha en los escenarios de evaluación (+75.34; semilla p = 0.029, pareado p = 0.020) y ninguna en los nuevos (+22.78; pareado p = 0.54). Con tan pocas semillas y tan poca dispersión entre ellas, la prueba a nivel de semilla no es confiable en ese último caso.

**Contabilidad de interacciones.** El World Model consume ~4,600 interacciones por semilla: el dataset de 4,800 transiciones, recolectado una sola vez y compartido por las 3 semillas, más 3,000 de la selección del checkpoint en SUMO real. Sin amortizar el dataset, son 7,800. El RL directo consume 13,240 con 10,000 pasos y 39,208 con 30,000, contando la evaluación periódica que elige su mejor checkpoint. Las razones son 2.9x y 8.5x (1.7x y 5.0x sin amortizar).

**Interpretación.** La ventaja demostrada del World Model es de **eficiencia en interacciones**: con una fracción del contacto con el simulador, iguala o supera al RL directo, que es lo que plantea la pregunta de investigación (Sección 4). No es una ventaja de control cuando el RL directo dispone de presupuesto de sobra. Tampoco se midió la curva completa de desempeño frente a interacciones del RL directo (hay dos puntos), así que el punto exacto en que deja de distinguirse del World Model queda sin determinar. Dos salvedades adicionales. Con 30,000 pasos, el RL directo elige su mejor checkpoint entre 30 evaluaciones periódicas en vez de 10, sobre los mismos 5 escenarios de selección, así que parte de su mejora puede venir de esa búsqueda más amplia y no solo de entrenar más. Y con 4 semillas, los resultados del RL directo todavía cambian de forma apreciable entre reentrenamientos; harían falta al menos 5 semillas por método para comparaciones más firmes.

## 20.3 Experimento 0 rehecho

La auditoría técnica encontró que las dos ramas del Experimento 0 no se entrenaban igual: el script de la rama cruda copiaba el bucle de entrenamiento y se había quedado sin la regularización (weight decay) ni el early stopping que se agregaron después a la rama `z`. Se corrigió haciendo que las dos ramas importen el mismo protocolo, y se repitió el experimento en tres pasos:

1. Con el protocolo compartido y la misma semilla que el resultado publicado, la conclusión se invirtió: el estado crudo ganó los 10 horizontes.
2. Con 5 semillas por rama (comparación pareada por semilla), no hubo diferencia consistente: la semilla publicada resultó ser la más extrema de las cinco.
3. Se comprobó además que el tope de 100 épocas cortaba el entrenamiento de la rama `z` antes de converger, pero no el de la cruda. Con un tope de 300 épocas, dejando que el early stopping decidiera el corte (entre las épocas 105 y 167 en las 10 corridas), **el Autoencoder mejora la predicción de la recompensa en 37 de 50 pares semilla × horizonte, con una mejora mediana de 10.9%**. Gana en 4 de las 5 semillas, más en horizontes cortos y medios (+11% a +21%) que en los largos (+4% a +6%).

La conclusión original (el Autoencoder aporta) se mantiene en su dirección, pero con otra evidencia y otra magnitud. Esto confirma la lección de la Sección 33: el problema no estaba en ningún componente por separado, sino en que dos piezas que debían compartir el mismo protocolo no lo compartían.

# 21. Hipótesis — evaluadas contra los resultados obtenidos

- **H1.** Un World Model entrenado con trayectorias de SUMO puede aprender una aproximación útil de la dinámica de una intersección de tráfico. **Confirmada** (Experimento 1).
- **H2.** El error de predicción aumentará al proyectar más pasos de manera autorregresiva debido al *compounding error*. **Confirmada** — el error latente acumulado crece de forma medible entre el horizonte 1 y el 10, aunque el modelo sigue superando al baseline persistente en todos los horizontes evaluados.
- **H3.** El uso de un World Model puede reducir la cantidad de interacciones necesarias con SUMO para obtener un controlador competitivo respecto al entrenamiento directo. **Confirmada**, en el sentido preciso de la hipótesis: frente a un RL directo con 2.9 veces sus interacciones (1.7 veces sin amortizar el dataset), el World Model es mejor en SUMO real, en los escenarios de evaluación y en escenarios nuevos. Frente a uno con 8.5 veces sus interacciones (5.0 sin amortizar), no se detectó una diferencia consistente (Sección 20.2). La confirmación es de eficiencia en interacciones, no de mejor control cuando el RL directo dispone de presupuesto de sobra.
- **H4.** Un Transformer temporal puede presentar un comportamiento predictivo diferente al de una LSTM. **Confirmada parcialmente**: el Transformer sí se comporta de forma distinta a la LSTM, pero en la dirección contraria a una mejora. Predice mejor el estado latente, pero peor la recompensa, y pierde en la métrica de decisión (`reward_mse`) en los 10 horizontes evaluados (Experimento 3, Sección 20.1).
- **H5.** Un Autoencoder no necesariamente mejorará la predicción del modelo temporal frente al vector crudo normalizado; su inclusión debe depender del Experimento 0. **Refutada, con matices**: con el mismo protocolo en las dos ramas, 5 semillas y entrenamiento hasta convergencia, el Autoencoder mejora la predicción de la recompensa de forma moderada y mayoritaria (4 de 5 semillas, mejora mediana de 10.9%), no universal (Sección 20.3). Se mantuvo en el sistema final por el mecanismo de decisión que esta misma hipótesis proponía. La versión anterior de este experimento, con una sola semilla y un protocolo desigual, mostraba una ventaja más clara que no resistió la corrección.
- **H6.** Una arquitectura sin recurrencia ni atención (TSMixer) podría igualar el error de predicción de la LSTM y del Transformer. **Rechazada**: TSMixer fue la peor de las tres arquitecturas en todos los horizontes evaluados, incluso después de verificar que no le faltaba convergencia (300 épocas sin early stopping, mismo resultado; Sección 20.1).

# 22. Plan de trabajo — ejecución real

El plan original de 6 semanas se usó como guía de orden de desarrollo (simulador → datos → representación → World Model → imaginación/control → comparación), pero el tiempo real se extendió más allá de lo planeado, principalmente por las cuatro correcciones críticas descritas en la Sección 20 — cada una exigió su propio ciclo de diagnóstico con evidencia, corrección, y re-verificación completa de los resultados posteriores. El orden conceptual del plan original se mantuvo válido de principio a fin; lo que se subestimó fue el tiempo necesario para que cada etapa fuera *confiable*, no solo *funcional*.

# 23. Riesgos y mitigación — evaluados retrospectivamente

| Riesgo | Impacto previsto | Qué ocurrió realmente |
| --- | --- | --- |
| Complejidad de SUMO/TraCI | Alto | Se materializó parcialmente: no en la complejidad de uso básico, sino en una desconexión sutil entre cómo `sumo-rl` controla el semáforo y cómo se leía su estado — el bug crítico #1 de la Sección 20. |
| Modelo temporal inestable | Alto | No se materializó de forma severa; el LSTM entrenó de forma estable en todas las corridas. |
| *Compounding error* | Medio/alto | Confirmado (H2), pero acotado — el modelo siguió siendo útil en todos los horizontes evaluados. |
| Alcance excesivo | Alto | Mitigado exitosamente: Transformer y TSMixer se mantuvieron fuera del núcleo mientras se hacían las correcciones críticas, tal como se planeó, y se ejecutaron después, una vez estabilizado el núcleo (Experimento 3). |
| Resultados poco concluyentes | Medio | Se materializó parcialmente: con 3 y 4 semillas por método, los resultados del RL directo cambian de forma apreciable entre reentrenamientos. Mitigado con pruebas a nivel de semilla y pareadas por escenario, y con una evaluación adicional en escenarios nuevos. La conclusión de eficiencia es firme; la comparación con el RL directo de mayor presupuesto no es concluyente. |
| Tecnología no justificable ante la profesora | Medio | La auditoría de la Sección 30 se mantuvo vigente durante todo el proyecto. |

# 24. Alcance y criterios de corte

El núcleo obligatorio (una intersección SUMO, dataset de trayectorias, representación vectorial, World Model temporal, evaluación predictiva, y comparación de control con y sin uso del modelo) **se completó en su totalidad**. El Autoencoder se mantuvo en el sistema final porque el Experimento 0 demostró su utilidad.

El escenario de demanda asimétrica (Sección 20) fue una adición **dentro del alcance del núcleo**, no una extensión — es una variación necesaria del mismo escenario de una sola intersección, requerida para que la comparación de control tuviera sentido, no una ampliación de la arquitectura del sistema.

Las extensiones Transformer y TSMixer, con la misma prioridad entre sí, **se ejecutaron en una fase posterior del proyecto, una vez estabilizado el núcleo**: exactamente en el orden que esta misma sección anticipaba ("se implementarán únicamente después de estabilizar el núcleo"). Se mantuvieron fuera del núcleo mientras este demandó más tiempo de estabilización del previsto, y se retomaron cuando dejó de demandarlo. Es una confirmación de que el orden de prioridades funcionó, no una corrección. Resultado: ninguna de las dos reemplaza a la LSTM (Sección 20.1). La visión mediante imágenes, la implementación completa de Dreamer/STORM, y MDN-RNN + CMA-ES se mantuvieron fuera del alcance, como se planteó desde el inicio.

# 25. Resultados obtenidos

- Una intersección SUMO reproducible y controlable desde Python, con una demanda de tráfico calibrada específicamente para que el problema de control tuviera una respuesta no trivial.
- Un conjunto de 80 episodios de trayectorias de tráfico, con variedad real (semilla de SUMO distinta por episodio) y correctamente estructurado (sin fuga de datos entre splits).
- Un World Model con error de predicción medible y superior a un baseline persistente en los 10 horizontes evaluados.
- Un análisis del *compounding error*, confirmando su existencia sin que invalide la utilidad del modelo.
- Un Dream Environment funcional, usado para entrenar un controlador PPO de principio a fin sin ninguna interacción real con SUMO.
- Dos controladores evaluados en el entorno real de SUMO (World Model y RL directo), con 3 y 4 semillas de entrenamiento independientes, en los escenarios de evaluación del proyecto y en escenarios nuevos, con los resultados de cada episodio versionados en el repositorio.
- Una comparación objetiva y estadísticamente evaluada frente a control fijo, una regla determinista, y RL directo.
- Una conclusión experimental sobre el valor de usar un World Model para reducir interacciones con el simulador: **positiva**. El World Model iguala o supera al RL directo con entre 2.9 y 8.5 veces menos interacciones reales, y es el método con menos episodios catastróficos. Frente a un RL directo con el triple de presupuesto no se detectó una diferencia consistente (Sección 20.2). Los matices estadísticos se declaran explícitamente, no se ocultan.
- Una comparación de tres arquitecturas temporales (LSTM, Transformer, TSMixer) bajo el mismo protocolo, con un criterio de decisión fijado antes de ver los resultados: la LSTM se mantiene (Experimento 3).
- Una justificación explícita, técnica por tecnología, de por qué cada herramienta usada era necesaria (Sección 30), y de por qué las que no quedaron en el sistema final se descartaron: el VAE probabilístico por no ser necesario, y el Transformer y TSMixer porque, una vez evaluados, no mejoraron a la LSTM (Sección 20).

# 26. Limitaciones

- El proyecto se limitó, como se planeó, a una única intersección.
- La simulación no representa toda la complejidad del tráfico urbano real.
- Las conclusiones están condicionadas al escenario de demanda usado (asimétrico, constante en el tiempo) — una demanda variable en el tiempo o una red de varias intersecciones podría cambiar el resultado relativo entre métodos.
- El World Model acumula error en predicciones largas (*compounding error*, confirmado), mitigado pero no eliminado mediante límites de horizonte calibrados empíricamente.
- Persiste, en ambos métodos, un número reducido de episodios donde el desempeño es mucho peor que el promedio ("episodios catastróficos": 5 de 90 en el World Model en los escenarios de evaluación y 1 de 90 en los nuevos, frente a 25 de 120 y 29 de 120 en el RL directo con 10,000 pasos), cuya causa completa no se identificó pese a una investigación dedicada.
- Con 3 semillas de entrenamiento en el World Model y 4 en el RL directo, las pruebas a nivel de semilla tienen poca potencia, y los resultados del RL directo cambian de forma apreciable entre reentrenamientos. La comparación con el RL directo de mayor presupuesto no es concluyente (Sección 20.2).
- El modelo temporal (LSTM) del sistema se entrenó hasta el tope de 100 épocas, no hasta converger (con early stopping, su mejor época habría sido la 152). No se reentrenó porque todo lo construido encima (Dream Environment, controladores y su evaluación) depende de él.
- El término de flujo de la recompensa no medía las llegadas reales (Sección 13). Su peso es despreciable, así que no altera las comparaciones, pero la recompensa no premia el flujo vehicular como se planteó.
- La recolección del dataset fija la semilla del tráfico de cada episodio, pero no la de las acciones aleatorias: una recolección nueva produciría otro dataset. El dataset usado está respaldado.
- Los escenarios de evaluación usados durante todo el proyecto también sirvieron para tomar decisiones de diseño. Por eso los resultados finales se validaron además en 30 escenarios nuevos, nunca usados antes (Sección 20.2).
- No se implementó Dreamer, STORM, ni el sistema original de Ha y Schmidhuber con MDN-RNN y CMA-ES, tal como se planteó desde el inicio.

# 27. Antecedentes y referencias

## Referencias académicas

Ha, D., & Schmidhuber, J. (2018). World Models. Fundamenta la idea de aprender una representación interna y una dinámica del entorno para utilizarla en el control — la base conceptual de todo el proyecto.

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. Referencia conceptual para el uso de trayectorias imaginadas en el espacio latente, el principio detrás del Dream Environment implementado.

Zhang, W., Wang, G., Sun, J., Yuan, Y., & Huang, G. (2023). STORM: Efficient Stochastic Transformer-based World Models for Reinforcement Learning. Referencia para la variante del modelo temporal basada en Transformer, evaluada en el Experimento 3 (no reemplazó a la LSTM).

Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. Ö., & Pfister, T. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting. Sustenta la variante del modelo temporal TSMixer, evaluada en el Experimento 3 (no reemplazó a la LSTM).

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? Justificación metodológica para incluir una arquitectura simple sin atención (TSMixer) como término de comparación en el Experimento 3.

Dai et al. (2022). Image-based traffic signal control via world models. Antecedente que conecta World Models con el control de señales de tráfico.

## Herramientas de software (infraestructura, no técnicas de modelado)

Alegre, L. N. SUMO-RL — wrapper tipo Gymnasium para el control de semáforos en SUMO. https://github.com/LucasAlegre/sumo-rl. Usado como capa de integración entre `TrafficEnvironment` y SUMO/TraCI.

Stable-Baselines3 — implementación estándar de PPO y de la utilidad `VecNormalize`, central en la resolución del problema de la función de valor descrito en la Sección 20.

# 28. Dataset — nota sobre la decisión final

La propuesta original contemplaba, como referencia, el uso de un benchmark externo (Traffic Signal Control Benchmark basado en CityFlow). En la práctica, el proyecto se desarrolló íntegramente sobre una red y una demanda de tráfico propias, construidas y calibradas específicamente para el objetivo experimental del proyecto (incluyendo el rediseño a demanda asimétrica descrito en la Sección 20) — no se usó ningún dataset ni benchmark externo. Esta decisión simplificó la reproducibilidad del proyecto (control total sobre la demanda y la red) a cambio de no poder comparar directamente contra resultados de la literatura sobre CityFlow u otros benchmarks estándar.

# 29. Contribución académica

La contribución se planteó, y se mantuvo, como una adaptación y evaluación experimental de World Models para control semafórico, no como la creación de una nueva familia de modelos. El aporte concreto obtenido: una representación y dinámica aprendidas del tráfico, un mecanismo de entrenamiento sin interacción real (Dream Environment), y una comparación rigurosa (con pruebas estadísticas formales) frente a un agente entrenado directamente en SUMO — con una respuesta afirmativa a la pregunta de investigación, declarada con sus límites de evidencia explícitos.

# 30. Auditoría tecnológica y alineación curricular

Esta sección responde a si cada tecnología usada en el proyecto es necesaria y corresponde a un tema efectivamente cubierto por la profesora. El material del curso revisado incluye: repaso de redes neuronales y backpropagation, overfitting, convolución (CNN), modelos discriminativos vs. generativos, espacio latente, Autoencoder y VAE, embeddings, redes recurrentes (RNN) y LSTM, y la arquitectura Transformer completa.

## 30.1 Técnicas de modelado — alineadas con el curso

| Técnica | Uso en el proyecto | Tema del curso que la respalda | Estado final |
| --- | --- | --- | --- |
| Red neuronal densa (backprop, descenso de gradiente) | Base de todas las demás piezas: encoder/decoder, capa de salida de la LSTM | Repaso de redes neuronales, backpropagation | Implementada, núcleo del sistema |
| Autoencoder (determinista) | Compresión del vector de estado en z | Bloque de modelos generativos, espacio latente | Implementado; el Experimento 0 confirmó una mejora moderada y mayoritaria de la predicción de la recompensa (5 semillas por rama, mejora mediana de 10.9%) |
| LSTM | Modelo temporal que predice (z_{t+1}, r_{t+1}) a partir del historial | Bloque de RNN/LSTM | Implementada (núcleo), con `torch.nn.LSTM` estándar |
| Transformer | Extensión opcional para reemplazar la LSTM | Documento de la profesora sobre Transformers | Ejecutada, no reemplaza a la LSTM (ver Sección 20) |
| TSMixer | Extensión opcional para reemplazar la LSTM, misma prioridad que Transformer | Primitivas cubiertas: capas densas, normalización, residuales | Ejecutada, no reemplaza a la LSTM (ver Sección 20) |
| Embeddings de tokens discretos | No usado | Bloque de Word Embeddings | No aplica: el estado es un vector numérico continuo |
| Redes convolucionales (CNN) | No usado en el núcleo | Repaso de convolución | No aplica: el estado es un vector, no una imagen |

## 30.2 Infraestructura necesaria — no cubierta en el curso, pero indispensable

| Herramienta | Rol en el proyecto | ¿Por qué es necesaria de todos modos? |
| --- | --- | --- |
| SUMO + TraCI | Simulador de tráfico y su protocolo de control | Es el entorno mismo que se está modelando; no existe alternativa dentro del alcance. |
| sumo-rl | Wrapper tipo Gymnasium sobre TraCI | Evita reescribir manualmente decenas de llamadas TraCI ya resueltas por la comunidad. |
| Gymnasium | Interfaz estándar de entornos de RL | Requerida por sumo-rl, `DreamEnvironment` y Stable-Baselines3. |
| Stable-Baselines3 (PPO, VecNormalize) | Algoritmo de control y utilidades de normalización | Evita implementar RL desde cero; `VecNormalize` resultó indispensable para resolver un problema real (Sección 20). |
| PyTorch | Framework para el Autoencoder y la LSTM | Framework estándar de Deep Learning usado en todo el proyecto. |

## 30.3 Decisiones de exclusión explícita

- **CNN / convolución**: el estado es un vector numérico, no una imagen. Se mantuvo excluida durante todo el proyecto.
- **Embeddings de tokens**: el estado ya es un vector continuo por naturaleza. Se mantuvo excluida.
- **Mixture Density Network (MDN-RNN)**: reemplazada por una LSTM con salida densa determinística, tal como se planteó. Se mantuvo esa decisión — la LSTM implementada predice de forma determinística, sin componente probabilístico.
- **CMA-ES**: reemplazado por PPO vía Stable-Baselines3, tal como se planteó.
- **GANs, distancia de Wasserstein**: no fueron necesarias; el Autoencoder (determinista, sin siquiera el componente probabilístico de un VAE) cubrió la necesidad de representación del sistema.
- **VAE con reparametrización y divergencia KL**: aunque la propuesta contemplaba un Autoencoder *o* VAE, se implementó la versión determinista más simple. El Experimento 0 no exigía el componente probabilístico para responder su pregunta (¿ayuda comprimir?), y añadirlo habría sido complejidad adicional sin necesidad demostrada.

## 30.4 Criterio de decisión para el Autoencoder — aplicado y resuelto

Con el vector de estado en 26 dimensiones (la cifra final, ajustada desde la estimación inicial de 11-20 al decidir codificar la fase del semáforo como *one-hot* — ver Sección 11), el criterio de esta sección se mantiene exactamente igual de válido: 26 dimensiones sigue estando muy lejos de las ~262,000 de una imagen, así que la compresión latente no se asumió necesaria de antemano. El Experimento 0 resolvió esto empíricamente: con el mismo protocolo en las dos ramas, 5 semillas y entrenamiento hasta convergencia, el Autoencoder mejora la predicción de la recompensa de forma moderada y mayoritaria (37 de 50 pares semilla × horizonte, mejora mediana de 10.9%). Se mantuvo en el sistema final por este criterio, no por asunción (Sección 20.3).

## 30.5 Resumen de la auditoría

De las tecnologías contempladas, ninguna resultó completamente innecesaria. Los ajustes de esta versión final son, igual que en la revisión anterior, de alcance y de resultado empírico, no de eliminación arbitraria: el Autoencoder se mantuvo porque el Experimento 0 lo confirmó; el modelo temporal se simplificó a una LSTM determinística con implementación estándar de PyTorch; CMA-ES se reemplazó por PPO. Las dos extensiones opcionales del modelo temporal (Transformer y TSMixer) se ejecutaron finalmente, una vez estabilizado y verificado el núcleo, que era la prioridad que esta misma sección establecía desde el principio. Resultado: se mantiene la LSTM (Sección 20.1). Esto confirma en la práctica el principio de parsimonia que esta sección defendía: la LSTM, la opción ya integrada y validada en todo el sistema, se mantuvo no por defecto sino con evidencia, y la alternativa más compleja (el Transformer, con 3.5 veces sus parámetros) no aportó ninguna mejora en la predicción de recompensa.

# 31. Arquitectura final implementada

```
SUMO (demanda asimétrica: 500 veh/h vía principal, 150 veh/h vía secundaria)
    ↓
Estado del tráfico (vector numérico, 26 dimensiones)
    ↓
Autoencoder (determinista — confirmado útil por el Experimento 0)
    ↓
Representación z (16 dimensiones)
    ↓
Modelo temporal: LSTM determinística (torch.nn.LSTM, Dense de salida, sin MDN)
    [Transformer y TSMixer: implementados con la misma interfaz (TemporalModel) y evaluados; no reemplazan a la LSTM]
    ↓
Predicción de (ẑ_{t+1}, r̂_{t+1})
    ↓
Dream Environment (entorno completo de entrenamiento, compatible con Gymnasium)
    ↓
Controller (PPO vía Stable-Baselines3, con VecNormalize)
    ↓
Acción semafórica
    ↓
SUMO (evaluación real; también existe un PPO entrenado directamente aquí, para comparación)
```

# 32. Criterios de éxito — evaluados

- El simulador ejecuta episodios de forma estable y reproducible. **Cumplido.**
- El modelo predictivo supera un baseline simple de predicción. **Cumplido** (Experimento 1, 10/10 horizontes).
- El error a uno y varios pasos se encuentra cuantificado. **Cumplido.**
- La imaginación produce decisiones que pueden evaluarse nuevamente en SUMO. **Cumplido** — el controlador entrenado en el sueño se evaluó exhaustivamente en SUMO real.
- Los resultados se reportan mediante métricas de tráfico y de aprendizaje. **Cumplido**, con la salvedad de que algunas métricas planteadas originalmente (tiempo de viaje, combustible, CO₂) no se implementaron por no ser necesarias para la pregunta de investigación central.
- La comparación con los baselines permite formular una conclusión sustentada por datos. **Cumplido**, incluyendo pruebas de significancia estadística formal, algo no explícitamente exigido en la propuesta original pero incorporado por rigor.
- Cada tecnología usada puede justificarse explícitamente contra el temario del curso o como infraestructura indispensable. **Cumplido** (Sección 30).

# 33. Nota metodológica final

La estrategia seguida fue, tal como se planteó, construir primero una versión mínima funcional y medirla antes de agregar componentes. El orden real de desarrollo (simulador → datos → representación con su experimento de necesidad → World Model → evaluación predictiva → imaginación/control → comparación) coincidió con el plan original. Lo que la experiencia real del proyecto añade a esta nota metodológica, para cualquier trabajo futuro que continúe esta línea: **la etapa más costosa en tiempo no fue construir cada componente, sino verificar que la conexión entre componentes fuera correcta** — cada uno de los cuatro problemas críticos descritos en la Sección 20 era, en esencia, una desconexión silenciosa entre dos piezas que individualmente funcionaban bien. Un plan de trabajo futuro sobre esta base debería reservar tiempo explícito para esa verificación de integración, no solo para la construcción de cada pieza por separado.

*Propuesta de Proyecto de Grado — World Models + SUMO (versión final, post-ejecución, estado del repositorio hasta el commit `42e5eb8`)*
