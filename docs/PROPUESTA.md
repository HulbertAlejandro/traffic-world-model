# Propuesta de Proyecto de Grado

*Documento convertido desde PropuestaWorldModelsSUMO.docx para consulta directa por Claude Code.*

---

PROPUESTA DE PROYECTO DE GRADO

Control inteligente de semáforos mediante World Models para la reducción de congestión vehicular en entornos de simulación SUMO

Ingeniería de Sistemas

Universidad del Quindío

*Documento guía de formulación y desarrollo — Versión revisada*

2026

> **Nota sobre esta versión.** Este documento incorpora una auditoría tecnológica (Sección 30) que revisa cada herramienta y técnica del proyecto contra el material efectivamente cubierto por la profesora (redes neuronales, CNN, modelos generativos, espacio latente/Autoencoder/VAE, embeddings, RNN/LSTM, Transformers), con el fin de justificar su necesidad y ajustar el alcance donde correspondía. Los cambios respecto a la versión anterior están señalados explícitamente en las secciones 15, 17, 21, 27 y 31, y desarrollados en la Sección 30. Esta versión incorpora además TSMixer como segundo experimento opcional de modelo temporal, con exactamente la misma prioridad que el Transformer y sin alterar el resto del pipeline; los ajustes correspondientes aparecen en las secciones 2, 6, 9, 15, 16, 19, 20, 21, 22, 23, 24, 27, 30 y 31.

# 1. Propuesta de título

Control inteligente de semáforos mediante World Models para la reducción de congestión vehicular en entornos de simulación SUMO

# 2. Resumen

Este proyecto propone diseñar e implementar un sistema basado en World Models para aprender la dinámica de una intersección de tráfico urbano simulada en SUMO y utilizar el modelo aprendido como un entorno interno para apoyar el entrenamiento y/o selección de acciones de un controlador semafórico. La propuesta adapta el concepto de World Models a un dominio donde el estado puede representarse inicialmente mediante variables numéricas del tráfico, evitando depender obligatoriamente de imágenes.

El núcleo del proyecto contempla una intersección, una representación vectorial del estado, un modelo de representación latente mediante Autoencoder o VAE, un modelo temporal de dinámica y un controlador de aprendizaje por refuerzo. Se plantea comparar el entrenamiento tradicional, en el que el agente interactúa directamente con SUMO, frente a un enfoque asistido por World Model, con especial interés en determinar si se pueden reducir las interacciones necesarias con el simulador manteniendo o mejorando el desempeño.

Como extensión académica, se podrá estudiar el reemplazo del modelo temporal recurrente por un Transformer y, con la misma prioridad, por un TSMixer (arquitectura basada únicamente en capas densas que alterna mezcla temporal y mezcla de características). Ambas son extensiones opcionales del mismo proyecto: comparten el pipeline completo, los datos, la representación y las métricas del modelo principal, y solo sustituyen el bloque de modelo temporal, manteniendo el resto de condiciones controladas para comparar las tres arquitecturas.

Cada componente del sistema (Autoencoder/VAE, modelo temporal, controlador) se implementa deliberadamente en su forma más simple que sea suficiente, priorizando las técnicas cubiertas en el curso sobre extensiones del paper original de World Models que no fueron vistas en clase (ver Sección 30).

# 3. Planteamiento del problema

El control de semáforos es un problema de decisión secuencial en el que las acciones afectan la formación de colas, los tiempos de espera, la velocidad y el flujo de vehículos. Los métodos tradicionales de Reinforcement Learning pueden requerir numerosas interacciones con el simulador para aprender una política útil. Un World Model ofrece una alternativa: aprender primero una aproximación de la dinámica del entorno y utilizarla posteriormente para predecir las consecuencias de acciones sin consultar continuamente el entorno real.

El reto central consiste en determinar si un modelo aprendido de la dinámica del tráfico puede generar predicciones suficientemente útiles para apoyar el control semafórico. También debe estudiarse el error acumulado cuando las predicciones se proyectan durante varios pasos y si dicho error limita el uso del modelo como entorno de entrenamiento o imaginación.

# 4. Pregunta de investigación

¿Puede un World Model aprender de forma suficientemente precisa la dinámica de una intersección de tráfico simulada en SUMO y utilizar sus predicciones para apoyar el entrenamiento o la selección de acciones de un controlador semafórico, reduciendo las interacciones necesarias con el simulador sin deteriorar significativamente el desempeño del control?

# 5. Objetivo general

Diseñar e implementar un sistema basado en World Models que aprenda la dinámica del tráfico urbano y entrene un agente de control de semáforos dentro de un modelo del entorno, evaluando su desempeño frente al entrenamiento tradicional en SUMO.

# 6. Objetivos específicos

- Construir y configurar una intersección de tráfico en SUMO que permita generar episodios reproducibles y controlar sus fases semafóricas mediante TraCI.

- Definir una representación del estado del tráfico, un espacio de acciones y una función de recompensa orientados a reducir congestión y tiempos de espera.

- Recolectar trayectorias de interacción con SUMO y construir un conjunto de datos de transiciones para el aprendizaje de la dinámica.

- Implementar y evaluar un Autoencoder o VAE que permita obtener una representación latente compacta del estado, cuando esta compresión resulte justificada mediante una comparación explícita contra el vector de estado crudo (ver Sección 30.4).

- Implementar un modelo temporal de World Model que prediga el siguiente estado latente y la recompensa a partir del estado, las acciones y el historial temporal, usando una arquitectura determinística (LSTM con salida densa) antes de considerar extensiones probabilísticas no cubiertas en el curso.

- Evaluar la capacidad predictiva del World Model a uno y varios pasos, cuantificando el error acumulado de las predicciones.

- Construir un mecanismo de imaginación o un Dream Environment simplificado para evaluar consecuencias hipotéticas de acciones antes de ejecutarlas en SUMO.

- Implementar un controlador de Reinforcement Learning mediante una librería estándar (Stable-Baselines3) y compararlo con un controlador que interactúe directamente con SUMO.

- Analizar si el enfoque basado en World Models reduce el número de interacciones o el tiempo de entrenamiento manteniendo un desempeño competitivo.

- Como extensión opcional, comparar el modelo temporal recurrente (LSTM, modelo principal) con un Transformer y con un TSMixer bajo condiciones experimentales equivalentes, sustituyendo únicamente el bloque temporal del pipeline.

# 7. Justificación

- Combina Inteligencia Artificial, Deep Learning y Reinforcement Learning en un problema aplicado.

- Aborda un problema relevante: la reducción de congestión, tiempos de espera y formación de colas.

- SUMO permite trabajar con un entorno de simulación reproducible y controlable.

- Las métricas de evaluación son cuantificables y permiten comparaciones objetivas.

- El enfoque World Model permite estudiar una cuestión académica concreta: cuánto puede sustituirse la interacción directa con el simulador por predicciones de un modelo aprendido.

- La arquitectura puede ampliarse posteriormente hacia múltiples intersecciones, datos reales, representaciones visuales o modelos temporales basados en atención.

- Cada bloque conceptual del proyecto (redes neuronales, CNN, modelos generativos, espacio latente, VAE, embeddings, LSTM, Transformers) corresponde a un tema efectivamente cubierto en el curso, lo que permite defender técnicamente cada decisión de diseño ante la profesora (ver Sección 30.1).

# 8. Fundamento conceptual: adaptación de World Models

En la formulación clásica de World Models, el agente aprende una representación interna del entorno, aprende su dinámica temporal y utiliza esa representación para tomar decisiones. En el proyecto de tráfico se conserva esta filosofía, pero se adapta la representación a un entorno numérico.

Arquitectura conceptual inicial:

*SUMO → Estado del tráfico → Autoencoder/VAE → z → Modelo temporal → Predicción → Controller → Acción → SUMO*

La representación latente z constituye una versión comprimida del estado. El modelo temporal recibe información del estado y de las acciones anteriores y aprende a predecir la evolución futura. El controlador utiliza las predicciones para seleccionar acciones.

# 9. Arquitectura propuesta

| Componente | Entrada | Salida | Propósito |
| --- | --- | --- | --- |
| SUMO + TraCI | Red + flujo + acción | Estado del tráfico | Simular la intersección y ejecutar acciones. |
| Representación | Estado vectorial | z | Comprimir el estado si resulta necesario. |
| Modelo temporal | z, acción, recompensa e historial | z siguiente + recompensa | Aprender la dinámica del tráfico. Bloque intercambiable: LSTM (principal), Transformer o TSMixer (opcionales). |
| Dream Environment | z + acción | Predicción futura | Permitir imaginación sin consultar SUMO en cada paso. |
| Controller | Estado/predicciones | Acción semafórica | Seleccionar la acción de control. |

# 10. SUMO y TraCI

SUMO (Simulation of Urban MObility) será el simulador principal propuesto. TraCI permitirá controlar la simulación desde Python, consultar variables del tráfico y modificar el estado del semáforo. La primera implementación debe limitarse a una única intersección para reducir el riesgo técnico y facilitar la interpretación experimental.

Antes de desarrollar el World Model se debe verificar una línea base funcional: la simulación debe ejecutar episodios completos, el estado debe registrarse correctamente y las acciones semafóricas deben producir cambios observables en el tráfico.

# 11. Estado del tráfico

El estado inicial debe ser un vector numérico compacto. Las variables candidatas son:

- Número de vehículos por carril

- Número de vehículos esperando por carril

- Longitud de cola

- Velocidad promedio

- Ocupación

- Tiempo de espera

- Flujo de entrada y/o salida

- Fase actual del semáforo

- Tiempo transcurrido de la fase

La selección final debe evitar variables redundantes. El estado puede formalizarse como s_t = [feature_1, feature_2, …, feature_n]. Si el Autoencoder/VAE no aporta una mejora clara, se podrá utilizar directamente una representación normalizada del vector de estado y mantener el World Model sobre dicha representación.

# 12. Espacio de acciones

Se recomienda iniciar con un espacio discreto y pequeño:

- 0 = mantener la fase actual

- 1 = cambiar a la siguiente fase

Como extensión se puede estudiar una tercera acción para extender la fase. Deben incorporarse restricciones de operación, como un tiempo mínimo de verde y un tiempo mínimo entre cambios, para evitar conmutaciones irreales o excesivamente frecuentes.

# 13. Función de recompensa

La recompensa debe relacionarse directamente con los objetivos de control. Una formulación inicial puede ser:

*R_t = − α·WaitingTime_t − β·QueueLength_t + γ·Throughput_t − δ·SignalChanges_t*

Los coeficientes α, β, γ y δ no deben asumirse como óptimos; deben definirse mediante pruebas preliminares y mantenerse controlados entre experimentos. La recompensa debe evitar incentivar cambios excesivamente frecuentes del semáforo.

# 14. Dataset y generación de datos

El proyecto no depende obligatoriamente de un dataset de imágenes. La base de datos principal será un conjunto de trayectorias generadas en SUMO. Cada transición puede almacenarse como (s_t, a_t, r_t, s_t+1) y, después de obtener la representación latente, como (z_t, a_t, r_t, z_t+1).

Se deben separar entrenamiento, validación y prueba por episodios o escenarios, evitando que fragmentos de la misma trayectoria aparezcan simultáneamente en entrenamiento y prueba. Se deben guardar semillas, configuración de la simulación y versiones de los modelos para garantizar reproducibilidad.

# 15. World Model  [REVISADO]

| Cambio respecto a la versión anterior Se elimina la referencia implícita a una Mixture Density Network (MDN-RNN), propia de la implementación original de Ha & Schmidhuber (2018) pero no cubierta en el curso. El modelo temporal se implementa como una LSTM con una capa densa de salida que predice directamente el vector ẑ_{t+1} y la recompensa r̂_{t+1} (predicción determinística, pérdida MSE), siguiendo exactamente el mismo patrón LSTM → Dense → ŷ mostrado en el material de la profesora. Una extensión probabilística (MDN) queda documentada como trabajo futuro opcional, no como parte del núcleo. |
| --- |

El modelo debe aprender una aproximación de la dinámica sin utilizar las ecuaciones internas de SUMO. Conceptualmente:

*(z_t, a_t, r_t, historial) → (ẑ_t+1, r̂_t+1)*

La primera implementación recomendada utiliza un modelo recurrente LSTM, implementado en PyTorch replicando explícitamente las ecuaciones de compuertas (olvido, entrada, salida) vistas en el material didáctico del curso, por ser sencillo de estabilizar y de justificar matemáticamente en la sustentación. La LSTM es el modelo temporal principal del proyecto y no se reemplaza. Posteriormente, si el núcleo funciona, se puede comparar con un Transformer temporal, siguiendo la misma arquitectura encoder vista en el documento de la profesora sobre Transformers, y con la misma prioridad, con un TSMixer: una arquitectura puramente MLP que alterna bloques de mezcla temporal (a lo largo de la ventana de historia) y de mezcla de características (a lo largo de las variables del estado latente y la acción), con normalización y conexiones residuales, sin recurrencia ni atención.

Las tres arquitecturas (LSTM, Transformer y TSMixer) se implementan detrás de la misma interfaz: reciben una ventana de historia de longitud L con los pares (z, acción) y devuelven (ẑ_{t+1}, r̂_{t+1}) con la misma pérdida MSE, el mismo dataset, los mismos splits por episodio, la misma normalización calculada solo con entrenamiento y las mismas semillas. La única diferencia es el módulo interno que transforma la ventana: compuertas recurrentes en la LSTM, self-attention en el Transformer y capas densas de mezcla temporal y de características en TSMixer. Esto permite que sustituir el modelo temporal sea un cambio de una sola clase en el código, y no una reestructuración del proyecto.

# 16. Dream Environment e imaginación

Una vez entrenado el World Model, se debe crear una dinámica interna que permita proyectar consecuencias hipotéticas de acciones. En lugar de ejecutar cada acción candidata en SUMO, el sistema utiliza el modelo aprendido para estimar los estados y recompensas futuros.

El procedimiento puede consistir en: tomar el estado actual, simular varias acciones candidatas dentro del World Model, proyectar un horizonte corto, sumar las recompensas predichas y ejecutar en SUMO únicamente la acción seleccionada. Esto constituye una versión simplificada de imaginación o planificación en el modelo, no una implementación completa de Dreamer.

El Dream Environment se diseña de modo que sea independiente de la arquitectura temporal utilizada. Como la LSTM mantiene un estado oculto entre pasos mientras que el Transformer y TSMixer operan sobre una ventana fija de historia, el entorno imaginado conserva un búfer con las últimas L transiciones y expone siempre el mismo método de avance de un paso; el modelo temporal decide internamente si consume el búfer completo o su propio estado recurrente. Con ello, el mismo Dream Environment, el mismo controlador PPO y el mismo protocolo de evaluación se reutilizan sin cambios para las tres arquitecturas.

# 17. Controlador  [REVISADO]

| Cambio respecto a la versión anterior Se descarta explícitamente CMA-ES (estrategia evolutiva usada en el paper original) como tecnología del núcleo del proyecto: no se cubre en el curso, añade una familia algorítmica completa (optimización sin gradiente) ajena al contenido de Deep Learning visto, y no aporta valor adicional frente a PPO para este alcance. El controlador se implementa con PPO (Proximal Policy Optimization) a través de Stable-Baselines3, una librería estándar de RL que opera sobre el mismo backend de PyTorch usado en el resto del proyecto, evitando introducir un paradigma de optimización adicional no justificado. |
| --- |

El controlador principal se implementa con PPO sobre Stable-Baselines3, algoritmo adecuado para acciones discretas y ampliamente documentado. La comparación principal será entre un controlador que interactúa directamente con SUMO y un controlador que utiliza el World Model para generar trayectorias imaginadas o para apoyar su entrenamiento.

# 18. Baselines

- Control semafórico de tiempo fijo.

- Agente de Reinforcement Learning que interactúa directamente con SUMO.

- Método propuesto: World Model + imaginación/control.

El baseline fijo permite determinar si los métodos aprendidos realmente mejoran una estrategia convencional. El baseline RL directo permite evaluar el valor de aprender y utilizar una dinámica interna.

# 19. Métricas de evaluación

## Para el tráfico:

- Tiempo promedio de viaje

- Tiempo promedio de espera

- Longitud promedio de las colas

- Vehículos atendidos por minuto / throughput

- Consumo de combustible, si la configuración de SUMO lo permite

- Emisiones de CO₂, si la configuración de SUMO lo permite

## Para el aprendizaje:

- Recompensa acumulada

- Tiempo de entrenamiento

- Número de interacciones con SUMO

- Error de predicción a uno y varios pasos

- Error de predicción de la recompensa

- Número de parámetros, tiempo de entrenamiento e inferencia del modelo temporal (necesarios para comparar arquitecturas a costo comparable)

# 20. Diseño experimental  [AMPLIADO]

Experimento 0 — Necesidad del Autoencoder/VAE (nuevo). 

Antes de invertir tiempo entrenando un VAE, se compara el desempeño del modelo temporal (LSTM) usando (a) el vector de estado normalizado crudo, y (b) el vector latente z producido por un Autoencoder/VAE. El VAE se mantiene en el núcleo del proyecto únicamente si (b) mejora medible y consistentemente el error de predicción o la estabilidad del entrenamiento frente a (a). Este experimento formaliza el criterio ya mencionado en las Secciones 11 y 24, evitando usar el VAE "porque el curso lo cubrió" sin justificación empírica.

Experimento 1 — Capacidad predictiva. 

Entrenar el World Model y medir su error sobre episodios no vistos. Comparar predicción a un paso frente a predicción autorregresiva a varios pasos.

Experimento 2 — Control por imaginación. 

Comparar un agente sin World Model frente a un agente que utiliza el modelo aprendido para evaluar acciones. Medir desempeño en SUMO real.

Experimento 3 — Comparación de arquitecturas temporales (opcional). 

Comparar LSTM (modelo principal), Transformer (opcional) y TSMixer (opcional) usando exactamente la misma representación, los mismos splits por episodio, la misma normalización, la misma ventana de historia L, el mismo horizonte de evaluación autorregresiva, la misma función de pérdida, las mismas semillas y las mismas métricas. Solo se sustituye el bloque de modelo temporal: SUMO, el dataset, el Autoencoder/VAE, el Dream Environment y el controlador permanecen idénticos. Se reportará además el número de parámetros y el tiempo de entrenamiento e inferencia de cada arquitectura, de modo que la comparación no favorezca simplemente al modelo con mayor capacidad. Las dos extensiones tienen la misma prioridad; si el tiempo solo alcanza para una, se documentará cuál se ejecutó y por qué.

# 21. Hipótesis  [AMPLIADA]

- H1. Un World Model entrenado con trayectorias de SUMO puede aprender una aproximación útil de la dinámica de una intersección de tráfico.

- H2. El error de predicción aumentará al proyectar más pasos de manera autorregresiva debido al compounding error.

- H3. El uso de un World Model puede reducir la cantidad de interacciones necesarias con SUMO para obtener un controlador competitivo respecto al entrenamiento directo, aunque esta mejora debe comprobarse experimentalmente.

- H4. Un Transformer temporal puede presentar un comportamiento predictivo diferente al de una LSTM sobre la misma representación y tarea; no se debe asumir de antemano cuál será superior.

- H5 (nueva). Dado que el estado del tráfico es un vector numérico de baja dimensión (no una imagen), un Autoencoder/VAE no necesariamente mejorará la predicción del modelo temporal frente al vector crudo normalizado; su inclusión en el sistema final debe depender del resultado del Experimento 0, no asumirse de antemano.

- H6 (nueva). Dado que la dinámica se aprende sobre una ventana corta de un vector de baja dimensión, una arquitectura sin recurrencia ni atención (TSMixer, construida solo con capas densas de mezcla temporal y de características) podría igualar el error de predicción de la LSTM y del Transformer. Si esto ocurre, el resultado es informativo en sí mismo, porque indicaría que la complejidad adicional de la recurrencia o la atención no está justificada para este problema; no se asume de antemano cuál de las tres arquitecturas será superior.

# 22. Plan de trabajo de 6 semanas

| Semana | Actividades |
| --- | --- |
| 1 | Instalar y aprender SUMO/TraCI. Construir una intersección simple. Verificar fases, flujo y extracción de variables. |
| 2 | Definir estado, acciones y recompensa. Implementar entorno RL. Crear baseline de tiempo fijo y recolectar primeras trayectorias. |
| 3 | Generar dataset de transiciones. Normalizar datos. Ejecutar Experimento 0 (necesidad del Autoencoder/VAE) y entrenarlo solo si se justifica. |
| 4 | Implementar y entrenar el modelo temporal (LSTM determinístico) del World Model. Evaluar predicciones a uno y varios pasos. |
| 5 | Construir Dream Environment/imaginación. Integrar controlador PPO (Stable-Baselines3). Comparar con entrenamiento directo en SUMO. |
| 6 | Ejecutar experimentos finales, múltiples episodios/semillas, métricas, gráficas, análisis, documentación y preparación de sustentación. Si queda tiempo disponible, ejecutar el Experimento 3 (Transformer y/o TSMixer) reutilizando el mismo pipeline. |

# 23. Riesgos y mitigación

| Riesgo | Impacto | Mitigación |
| --- | --- | --- |
| Complejidad de SUMO/TraCI | Alto | Comenzar con una sola intersección y validar primero un entorno mínimo. |
| Modelo temporal inestable | Alto | Usar secuencias cortas, normalización, checkpoints y baseline predictivo. |
| Compounding error | Medio/alto | Evaluar horizontes cortos y cuantificar el crecimiento del error. |
| Alcance excesivo | Alto | Mantener visión, Transformer, TSMixer, MDN y CMA-ES como extensiones opcionales fuera del núcleo (ver Sección 30). |
| Resultados poco concluyentes | Medio | Usar múltiples episodios, semillas y baselines comparables. |
| Tecnología no justificable ante la profesora | Medio | Mantener la auditoría de la Sección 30 actualizada cada vez que se añada una librería o técnica nueva. |

# 24. Alcance y criterios de corte

El núcleo obligatorio debe ser: una intersección SUMO, dataset de trayectorias, representación vectorial, World Model temporal, evaluación predictiva y comparación de control con y sin uso del modelo. El Autoencoder/VAE debe mantenerse solo si demuestra utilidad para la representación (Experimento 0).

Las extensiones Transformer y TSMixer tienen la misma prioridad entre sí y se implementarán únicamente después de estabilizar el núcleo; ninguna de las dos modifica el pipeline, solo sustituye el bloque de modelo temporal, y si el tiempo disponible alcanza para una sola se documentará explícitamente cuál se ejecutó. La visión mediante imágenes no es necesaria para que el proyecto sea defendible y debe considerarse secundaria. La implementación completa de Dreamer, STORM, MDN-RNN + CMA-ES y arquitecturas excesivamente complejas queda fuera del alcance inicial, tanto por riesgo técnico como por no estar cubiertas en el curso (ver Sección 30).

# 25. Resultados esperados

- Una intersección SUMO reproducible y controlable desde Python.

- Un conjunto de trayectorias de tráfico correctamente estructurado.

- Un World Model con error de predicción medible.

- Un análisis del compounding error.

- Un mecanismo funcional de imaginación o Dream Environment.

- Un controlador evaluado en el entorno real de SUMO.

- Una comparación objetiva frente a control fijo y RL directo.

- Una conclusión experimental sobre el valor de utilizar un World Model para reducir interacciones con el simulador.

- Una justificación explícita, técnica by tecnología, de por qué cada herramienta usada era necesaria (Sección 30).

# 26. Limitaciones

- El proyecto se limita inicialmente a una única intersección.

- La simulación no representa toda la complejidad del tráfico urbano real.

- Las conclusiones estarán condicionadas al escenario y a la distribución de tráfico utilizados.

- Un World Model puede acumular errores durante predicciones largas.

- No se pretende implementar íntegramente Dreamer, STORM ni reproducir el sistema original de Ha y Schmidhuber, incluyendo su uso de MDN-RNN y CMA-ES.

# 27. Antecedentes y referencias  [AMPLIADO]

## Referencias académicas

Ha, D., & Schmidhuber, J. (2018). World Models. Este trabajo fundamenta la idea de aprender una representación interna y una dinámica del entorno para utilizarla en el control.

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. Sirve como referencia conceptual para el uso de trayectorias imaginadas en el espacio latente.

Zhang, W., Wang, G., Sun, J., Yuan, Y., & Huang, G. (2023). STORM: Efficient Stochastic Transformer-based World Models for Reinforcement Learning. Sirve como referencia para una extensión basada en Transformer.

Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. Ö., & Pfister, T. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting. Sustenta la variante opcional de modelo temporal construida únicamente con capas densas que alternan mezcla temporal y mezcla de características.

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? Justifica metodológicamente incluir una arquitectura simple sin atención como término de comparación frente a LSTM y Transformer en tareas de predicción temporal.

Dai et al. (2022). Image-based traffic signal control via world models. Este antecedente conecta directamente World Models con el control de señales de tráfico y sirve como referencia específica del dominio.

## Herramientas de software (infraestructura, no técnicas de modelado)

Alegre, L. N. SUMO-RL — wrapper tipo Gymnasium para el control de semáforos en SUMO. https://github.com/LucasAlegre/sumo-rl

Tallec, C., et al. Reimplementación de World Models en PyTorch (referencia de organización del pipeline VAE → MDN-RNN → Controller). https://github.com/ctallec/world-models

Shekhar, A. World Model reproduction on CarRacing-v3 (referencia de estructura de resultados y métricas a reportar). https://github.com/Abhi183/world-model

# 28. Artículo y dataset propuestos para el desarrollo

Artículo principal del dominio: “Image-based traffic signal control via world models” (2022). Debe utilizarse como antecedente científico para justificar la aplicación de World Models al control semafórico. El proyecto propuesto no pretende copiarlo; busca adaptar la idea a una intersección SUMO y estudiar una arquitectura controlada experimentalmente.

Dataset/benchmark recomendado: Traffic Signal Control Benchmark basado en CityFlow. Como referencia inicial se puede utilizar un escenario de una sola intersección, y posteriormente evaluar escenarios sintéticos si el núcleo funciona. Para SUMO, si se requiere una base de datos externa, se deberá seleccionar un conjunto compatible con la red y los flujos configurados en el simulador, evitando mezclar formatos sin una etapa de conversión y validación.

# 29. Contribución académica esperada

La contribución se plantea como una adaptación y evaluación experimental de World Models para control semafórico, no como la creación de una nueva familia de modelos. El aporte consiste en construir una representación y dinámica aprendidas del tráfico, integrar un mecanismo de imaginación y comparar su eficiencia frente a un agente que interactúa directamente con SUMO.

La pregunta de mayor interés experimental es si el modelo aprendido puede reducir el costo de interacción con el simulador sin producir una degradación significativa en las métricas de tráfico.

# 30. Auditoría tecnológica y alineación curricular  [SECCIÓN NUEVA]

Esta sección responde directamente a la pregunta de si cada tecnología usada en el proyecto es necesaria y si corresponde a un tema efectivamente cubierto por la profesora. El material del curso revisado incluye: repaso de redes neuronales y backpropagation, overfitting, convolución (CNN), modelos discriminativos vs. generativos, espacio latente, Autoencoder y VAE (incluyendo el truco de reparametrización), embeddings, redes recurrentes (RNN) y LSTM (incluyendo el ejemplo numérico paso a paso de las compuertas), y la arquitectura Transformer completa (self-attention, multi-head attention, cross-attention, Add & Norm).

## 30.1 Técnicas de modelado — alineadas con el curso

| Técnica | Uso en el proyecto | Tema del curso que la respalda | Veredicto |
| --- | --- | --- | --- |
| Red neuronal densa (backprop, descenso de gradiente) | Base de todas las demás piezas: encoder/decoder del VAE, capa de salida de la LSTM | Repaso de redes neuronales, backpropagation, regla de la cadena | Necesaria y central |
| Autoencoder / VAE (incl. reparametrización, divergencia KL) | Compresión opcional del vector de estado en z | Bloque de modelos generativos, espacio latente, VAE (reparametrización explicada paso a paso) | Condicional: solo si el Experimento 0 lo justifica |
| LSTM (compuertas de olvido/entrada/salida) | Modelo temporal que predice z_{t+1} a partir del historial | Bloque de RNN/LSTM con ejemplo numérico idéntico en estructura (f_t, i_t, Ḍi_t, o_t) | Necesaria (núcleo) |
| Transformer (self-attention, Add & Norm) | Extensión opcional para reemplazar la LSTM | Documento completo de la profesora sobre la arquitectura Transformer | Opcional, ya contemplada como extensión (Sección 21/H4) |
| TSMixer (mezcla temporal + mezcla de características, solo capas densas) | Extensión opcional para reemplazar la LSTM, con la misma prioridad que el Transformer | No aparece con ese nombre en el curso, pero se construye íntegramente con primitivas cubiertas: capas densas, backpropagation, normalización, conexiones residuales y regularización | Opcional, contemplada como extensión (Sección 21/H6) |
| Embeddings de tokens discretos | No se usa un módulo de embedding dedicado | Bloque de Word Embeddings del curso | No aplica: el estado es un vector numérico continuo, no símbolos discretos (ver 30.3) |
| Redes convolucionales (CNN) | No se usa en el núcleo del proyecto | Repaso de convolución, kernels, mapas de activación | No aplica: el estado es un vector, no una imagen (ver 30.3) |

## 30.2 Infraestructura necesaria — no cubierta en el curso, pero indispensable

Estas herramientas no corresponden a contenido de Deep Learning visto en clase, pero son indispensables porque el proyecto necesita un entorno de simulación y un mecanismo de control con el que interactuar; sin ellas no existiría un problema sobre el cual aplicar las técnicas de la Sección 30.1.

| Herramienta | Rol en el proyecto | ¿Por qué es necesaria de todos modos? |
| --- | --- | --- |
| SUMO + TraCI | Simulador de tráfico y su protocolo de control desde Python | No existe alternativa dentro del alcance del proyecto: es el entorno mismo que se está modelando. |
| sumo-rl | Wrapper tipo Gymnasium sobre TraCI (observación/acción/recompensa) | Evita reescribir manualmente decenas de llamadas TraCI ya resueltas y probadas por la comunidad; es infraestructura, no una técnica de IA que compita con el temario. |
| Gymnasium | Interfaz estándar de entornos de RL | Requerida por sumo-rl y por Stable-Baselines3; es el "idioma común" entre el entorno y el controlador. |
| Stable-Baselines3 (PPO) | Algoritmo de control (Reinforcement Learning) | El proyecto necesita un controlador entrenable; usar una librería estándar evita introducir una implementación propia de RL desde cero, que no es el foco del curso ni de la propuesta (ver 17). |
| PyTorch | Framework para implementar el VAE y la LSTM/Transformer | Framework estándar de Deep Learning; permite implementar manualmente las ecuaciones vistas en clase (compuertas de la LSTM, reparametrización del VAE) en vez de usar una caja negra. |

## 30.3 Decisiones de exclusión explícita

Tan importante como justificar lo que se usa es documentar lo que deliberadamente NO se usa, y por qué, para que ninguna ausencia se lea como un descuido:

- CNN / convolución: el estado del tráfico es un vector numérico (densidad, cola, fase, etc.), no una imagen. Introducir convoluciones exigiría una representación visual (como en Dai et al., 2022) que el proyecto explícitamente evita por complejidad innecesaria (Sección 24). El tema se domina conceptualmente por el curso, pero no aplica a esta implementación.

- Embeddings de tokens: los embeddings de palabras del curso resuelven cómo convertir símbolos discretos (texto) en vectores continuos. El estado del tráfico ya es un vector continuo por naturaleza; forzar una tabla de embeddings sobre variables numéricas no tiene justificación matemática y se descarta.

- Mixture Density Network (MDN-RNN): parte del World Model original, pero no cubierta en el curso. Se reemplaza por una LSTM con salida densa determinística (Sección 15), que sigue exactamente el patrón LSTM → Dense → ŷ enseñado en clase.

- CMA-ES: estrategia evolutiva del paper original para entrenar el controlador. Se reemplaza por PPO vía Stable-Baselines3 (Sección 17), evitando introducir una familia algorítmica de optimización sin gradiente ajena al contenido de Deep Learning del curso.

- GANs, distancia de Wasserstein: mencionadas en el curso como parte del panorama de modelos generativos, pero no son necesarias aquí porque el VAE ya cubre la necesidad de generación/imaginación del sistema; añadir una GAN duplicaría funcionalidad sin aportar valor al problema de control semafórico.

## 30.4 Criterio de decisión para el Autoencoder/VAE

Dado que el vector de estado tiene solo entre 11 y 20 dimensiones aproximadamente (muy lejos de las ~262,000 dimensiones de una imagen que sí justifican comprimir con un VAE, según el propio material del curso), no se asume de antemano que la compresión latente aporte valor. El Experimento 0 (Sección 20) decide esto empíricamente: el VAE se mantiene en el sistema final solo si mejora medible y consistentemente la predicción del modelo temporal frente al vector de estado crudo normalizado. Esta es una aplicación directa del criterio ya presente en las Secciones 11 y 24, formalizada aquí como un experimento explícito en vez de una decisión de diseño no verificada.

## 30.5 Resumen de la auditoría

De las tecnologías originalmente contempladas, ninguna resultó completamente innecesaria; todas cumplen un rol claro. Los ajustes de esta revisión son de alcance, no de eliminación: se pospone/condiciona el VAE (30.4), se simplifica el modelo temporal evitando MDN, y se reemplaza CMA-ES por una librería estándar de RL. El efecto neto es un proyecto más fácil de defender técnicamente, porque cada línea de código puede trazarse a un tema específico del curso o a una necesidad de infraestructura explícitamente justificada. La incorporación de TSMixer como segundo experimento opcional no altera esta conclusión: no introduce ninguna librería ni paradigma nuevo, se implementa en PyTorch con las mismas primitivas ya cubiertas (capas densas, normalización, conexiones residuales) y reutiliza el pipeline completo, de modo que su costo marginal es el de una clase adicional del modelo temporal.

# 31. Arquitectura final de referencia  [ACTUALIZADA]

SUMO

↓

Estado del tráfico (vector numérico)

↓

Autoencoder / VAE (solo si el Experimento 0 lo justifica)

↓

Representación (z, o el vector crudo normalizado)

↓

Modelo temporal (único bloque que cambia entre experimentos):

LSTM determinística (Dense de salida, sin MDN) — modelo principal

Transformer encoder temporal — experimento opcional

TSMixer (all-MLP: mezcla temporal + mezcla de características) — experimento opcional

↓

Predicción de estado + recompensa

↓

Dream Environment / Imaginación

↓

Controller (PPO vía Stable-Baselines3)

↓

Acción semafórica

↓

SUMO

Extensiones opcionales: sustituir la LSTM por un Transformer o por un TSMixer para realizar una comparación arquitectónica manteniendo el resto del diseño controlado. Las tres alternativas comparten exactamente el mismo pipeline (SUMO → dataset → normalización → Autoencoder/VAE → espacio latente → modelo temporal → Dream Environment → entrenamiento del controlador → comparación de resultados) y el mismo protocolo experimental; el único bloque que cambia es el modelo temporal. Por eso ninguna de las dos extensiones constituye un proyecto aparte: son experimentos adicionales dentro de la misma infraestructura.

# 32. Criterios para considerar el proyecto exitoso

- El simulador ejecuta episodios de forma estable y reproducible.

- El modelo predictivo supera un baseline simple de predicción.

- El error a uno y varios pasos se encuentra cuantificado.

- La imaginación produce decisiones que pueden evaluarse nuevamente en SUMO.

- Los resultados se reportan mediante métricas de tráfico y de aprendizaje.

- La comparación con los baselines permite formular una conclusión sustentada por datos.

- Cada tecnología usada puede justificarse explícitamente contra el temario del curso o como infraestructura indispensable (Sección 30).

# 33. Nota metodológica final

La estrategia recomendada es construir primero una versión mínima que funcione y medirla antes de agregar componentes. El orden de desarrollo debe ser: simulador → datos → baseline → representación (con su experimento de necesidad) → World Model → evaluación predictiva → imaginación/control → comparación → extensiones opcionales de modelo temporal (Transformer y TSMixer, con la misma prioridad). Esta secuencia permite que el proyecto conserve un núcleo defendible incluso si alguna extensión no alcanza estabilidad dentro del tiempo disponible, y que cada decisión tecnológica quede respaldada tanto académica como técnicamente.

*Propuesta de Proyecto de Grado — World Models + SUMO (versión revisada con auditoría tecnológica)*