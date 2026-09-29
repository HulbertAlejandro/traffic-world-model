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
