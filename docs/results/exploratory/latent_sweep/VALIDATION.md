# Validación de los ejecutores exploratorios (antes del barrido)

Los ejecutores de `sweep.py` importan el protocolo oficial y solo cambian las rutas. Para comprobar
que reproducen el pipeline oficial, se corrieron con los datos oficiales:

| Prueba | Resultado |
|---|---|
| `train-ae --latent 16` sobre `datasets/processed` (26 dims) frente a `models/checkpoints/autoencoder_best.pt` | todos los tensores idénticos (mejor época 98) |
| `train_latent.npz` codificado por ese Autoencoder frente al oficial | idéntico |
| `train-lstm --seed 0` sobre `train_raw_seq.npz` frente a `exp0_multiseed_300ep/raw/seed0` | todos los tensores idénticos (corte en la época 105) |
| `train-lstm --seed 0` sobre `train_latent.npz` frente a `exp0_multiseed_300ep/z/seed0` | todos los tensores idénticos (corte en la época 167) |
| Datos de 24 dims (`make-data`) | `states`/`next_states` = oficiales sin las columnas 22 y 23; el resto de los campos, idéntico |

Se comparan los tensores y no el md5 del archivo, porque `torch.save` guarda el nombre del archivo
dentro del zip.

## Entrenadores de Transformer y TSMixer sobre estado crudo (Fase 2, Paso 1)

`training/train_world_model_transformer_raw.py` y `training/train_world_model_tsmixer_raw.py`
exponen `train(train_path, validation_path, seed, epochs, output_dir)`. Importan el protocolo de
`train_world_model.py` y los hiperparámetros, la clase y el escalador compartido de su script
oficial. Las dos ramas del experimento (latente y cruda) usan esa misma función.

| Prueba | Resultado |
|---|---|
| Transformer `train()` con `train_latent.npz` oficial, semilla 0, 100 épocas, frente a `models/checkpoints/world_model_transformer_best.pt` | todos los tensores idénticos (corte en la época 74, mejor 59; 328 s de pared, 2 en paralelo) |
| TSMixer, lo mismo, frente a `world_model_tsmixer_best.pt` | todos los tensores idénticos (llegó al tope de 100, mejor época 99; 277 s) |
| CLI sobre el estado de 24 dims (2 épocas) | entrena con `input_dim` = 24 |
| CLI con `--output-dir models/checkpoints` | se niega (guardia) |
| `tests/test_transformer_tsmixer_raw_protocol.py` | 6/6. Suite completa: 46 pasan; 8 archivos no se pueden cargar porque una directiva de Control de aplicaciones de Windows bloquea una DLL de pandas (entorno, no código; son los tests de SUMO/SB3, que estos cambios no tocan) |
