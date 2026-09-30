# DESCARTABLE: solución puntual a un bloqueo del entorno (2026-09-29)

**No forma parte del pipeline, ni del oficial ni del exploratorio.** Existe solo porque, en este
equipo, Smart App Control / Control de aplicaciones de Windows bloquea la DLL de `kiwisolver`
("Una directiva de Control de aplicaciones bloqueó este archivo"). Sin esa DLL no carga
matplotlib, y `grid_arch.py` no puede importar la evaluación oficial, que importa matplotlib a nivel
de módulo aunque el cálculo no dibuje nada. Reinstalar `kiwisolver` 1.5.1 no lo resolvió. No se
tocó ninguna configuración de Windows ni ningún archivo oficial.

`grid_arch_standalone.py` copia (no importa) la evaluación de `evaluation/world_model_evaluation.py`
y `evaluate()` de `scripts/compare_experiment_0_multiseed.py`, además de `anova2` de `grid5.py` y
el análisis de `grid_arch.py`. Esas fuentes siguen siendo la única referencia.

**Verificación antes de usarlo** (`--verify`): se recalcularon desde los checkpoints las 35 + 35
corridas de LSTM y Transformer, sin usar la caché. El resultado reproduce **exactamente** (igualdad
de floats, 0 celdas distintas) `gridarch_lstm.json` y `gridarch_transformer.json`, que escribió
`grid_arch.py` con la evaluación importada, y también el análisis completo
`grid_arch_lstm_transformer.json`. Solo después se usó para calcular `gridarch_tsmixer.json`.

Con el bloqueo resuelto, `grid_arch.py lstm transformer tsmixer` lee las tres cachés y no necesita
este archivo. Esta carpeta puede borrarse entonces. Se conserva solo para dejar rastro de cómo se
calculó `gridarch_tsmixer.json`.
