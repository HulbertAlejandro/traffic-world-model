#!/bin/bash
# ADDENDUM_grid5.md: AE seeds 3, 4 (latent_dim=12) and the LSTM runs missing from the 5x5 grid.
cd "$(dirname "$0")/../../../.."
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
MAX_PARALLEL=3
start=$(date +%s)
for a in 3 4; do
  if [ -f $W/ae12_seed$a/ae_summary.json ]; then echo "ae12_seed$a already done, skipped"; continue; fi
  $PY $D/sweep.py train-ae --latent 12 --seed $a --out $W/ae12_seed$a > $D/logs/ae12_seed$a.log 2>&1; echo "ae12_seed$a exit=$?"
done
wait_slot() { while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do sleep 5; done; }
for a in 1 2 3 4; do for s in 0 1 2 3 4; do
  if [ -f $W/z12_ae$a/seed$s/lstm_summary.json ]; then echo "z12_ae$a seed$s already done, skipped"; continue; fi
  wait_slot
  ($PY $D/sweep.py train-lstm --train $W/ae12_seed$a/train_latent.npz --val $W/ae12_seed$a/validation_latent.npz \
       --seed $s --out $W/z12_ae$a/seed$s > $D/logs/z12_ae${a}_seed$s.log 2>&1; echo "z12_ae$a seed$s exit=$?") &
done; done
wait
echo "wall seconds: $(( $(date +%s) - start ))"
