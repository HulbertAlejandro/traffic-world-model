#!/bin/bash
# Paso 3 (MANIFEST.md): 5 autoencoders, 15 z runs (seeds 0-2), 3 raw-24 runs (seeds 0-2). 6 at a time.
cd "$(dirname "$0")/../../../.."
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
start=$(date +%s)
for k in 4 8 12 16 20; do $PY $D/sweep.py train-ae --latent $k --out $W/ae$k > $D/logs/ae$k.log 2>&1; echo "ae$k exit=$?"; done
jobs_run() { while [ $(jobs -rp | wc -l) -ge 6 ]; do sleep 5; done; }
for s in 0 1 2; do
  jobs_run; ($PY $D/sweep.py train-lstm --train $W/data/train_raw_seq.npz --val $W/data/validation_raw_seq.npz --seed $s --out $W/raw24/seed$s > $D/logs/raw24_seed$s.log 2>&1; echo "raw24 seed$s exit=$?") &
done
for k in 4 8 12 16 20; do for s in 0 1 2; do
  jobs_run; ($PY $D/sweep.py train-lstm --train $W/ae$k/train_latent.npz --val $W/ae$k/validation_latent.npz --seed $s --out $W/z$k/seed$s > $D/logs/z${k}_seed$s.log 2>&1; echo "z$k seed$s exit=$?") &
done; done
wait
echo "paso3 wall seconds: $(( $(date +%s) - start ))"
