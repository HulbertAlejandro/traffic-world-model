# Addendum (escrito y commiteado antes de entrenar): refuerzo de `latent_dim` = 8

Pedido del autor después de ver el resultado de `latent_dim` = 12. El MANIFEST no cambia.

- **Procedimiento:** idéntico al del ganador. `run_reinforce.sh 8` entrena z8 con las semillas
  3–9. La rama cruda de 24 con semillas 3–9 ya existe (Paso 4 de k = 12) y se reutiliza sin
  reentrenar: es la misma comparación, y el entrenamiento es determinista.
- **Selección:** 8 no ganó el barrido; se refuerza como segundo mejor, elegido por el autor
  después de ver los resultados de 12. Sus semillas 0–2 llevan sesgo de selección, así que la
  verificación sin sesgo son las semillas 3–9 solas.
- **Pruebas por rama** (las mismas que para 12): `stats.py` sobre las 10 semillas y sobre las
  semillas 3–9, y `posthoc_unpaired.py` ampliado con z8.
- **Criterio fijado para "8 y 12 se distinguen":** prueba de permutación exacta bilateral, sin
  emparejar, entre las 10 corridas de z8 y las 10 de z12, sobre la media geométrica del
  `reward_mse` en h = 1..10 (la métrica del control *post hoc*), con α = 0.05. Si p ≥ 0.05, se
  reportan como no distinguibles con estos datos. Eso no demuestra que sean iguales: se informan
  también la diferencia y el rango de las reducciones por semilla de cada uno.
