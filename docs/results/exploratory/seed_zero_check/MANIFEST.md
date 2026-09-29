# Exploratorio: verificación de la semilla 0 (Experimento 0)

**No es parte del Experimento 0 oficial** (docs/results/experiment_0_multiseed_300ep.json, 5 semillas),
no lo reemplaza y no se cita en la documentación principal.

Semillas nuevas fijadas ANTES de ejecutar la primera corrida (registrado 2026-09-28T19:42:28-05:00):
10, 20, 30, 40, 50 — todas se reportan, sea cual sea el resultado.

Protocolo: idéntico al oficial (training/train_world_model.py y train_world_model_raw.py sin cambios,
--epochs 300, early stopping de paciencia 15, mismo split y escalador). Pesos en
models/checkpoints/exploratory_seed_zero_check/{z,raw}/seedN (no versionados).

## Resultado (exploratorio; no reemplaza el oficial de 5 semillas)

Reducción por semilla = mediana sobre h=1..10 de (crudo − z)/crudo (misma definición que la tabla oficial).

| Semilla | z gana horizontes | Reducción mediana |
|---|---|---|
| 0 (oficial) | 4/10 | -3.4% |
| 1 (oficial) | 6/10 | +7.4% |
| 2 (oficial) | 8/10 | +10.0% |
| 3 (oficial) | 9/10 | +29.4% |
| 4 (oficial) | 10/10 | +26.5% |
| 10 | 3/10 | -6.6% |
| 20 | 1/10 | -3.9% |
| 30 | 10/10 | +22.1% |
| 40 | 0/10 | -25.4% |
| 50 | 9/10 | +9.9% |

10 semillas: z gana 60/100 pares; mediana de las medianas por horizonte +8.4%; mediana sobre los 100 pares
+5.4%; 6/10 semillas favorecen a z (signo p = 0.75; Wilcoxon exacto sobre las medianas por semilla p = 0.19).
Semillas nuevas solas: media -0.8% (originales +14.0%; permutación p = 0.15).
Archivos: experiment_0_extra_seeds.json (solo 10–50), experiment_0_10seeds_combined.json (las 10; las 5
originales reevaluadas coinciden exactamente con docs/results/experiment_0_multiseed_300ep.json).
