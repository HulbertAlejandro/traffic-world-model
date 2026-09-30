#!/bin/bash
# Paso 3 comparisons: each latent_dim (seeds 0-2) against the shared raw-24 branch (seeds 0-2).
cd "$(dirname "$0")/../../../.."
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
for k in 4 8 12 16 20; do
  echo "== latent_dim $k =="
  $PY $D/sweep.py compare --z-dirs $W/z$k/seed{0,1,2} --raw-dirs $W/raw24/seed{0,1,2} \
      --test-z $W/ae$k/test_latent.npz --test-raw $W/data/test_raw_seq.npz --output $D/sweep_z$k.json
done
$PY $D/select_winner.py
