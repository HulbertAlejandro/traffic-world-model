#!/bin/bash
# ADDENDUM_ae_seeds.md: latent_dim=12 autoencoders with seeds 1 and 2 (seed 0 = ae12/, reused), each
# with LSTM seeds 0-2. First checks that train-ae --seed 0 still reproduces ae12/ (scratch folder).
cd "$(dirname "$0")/../../../.."
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
MAX_PARALLEL=3
start=$(date +%s)
$PY $D/sweep.py train-ae --latent 12 --seed 0 --out $W/check_ae12_seed0 > $D/logs/check_ae12_seed0.log 2>&1
$PY -c "
import torch
a=torch.load('$W/check_ae12_seed0/autoencoder_best.pt');b=torch.load('$W/ae12/autoencoder_best.pt')
ok=all(torch.equal(a[k],b[k]) for k in b); print('train-ae --seed 0 reproduces ae12/:', ok); raise SystemExit(0 if ok else 1)
" || { echo "ABORT: seed-0 check failed"; exit 1; }
for a in 1 2; do
  if [ -f $W/ae12_seed$a/ae_summary.json ]; then echo "ae12_seed$a already done, skipped"; continue; fi
  $PY $D/sweep.py train-ae --latent 12 --seed $a --out $W/ae12_seed$a > $D/logs/ae12_seed$a.log 2>&1; echo "ae12_seed$a exit=$?"
done
wait_slot() { while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do sleep 5; done; }
for a in 1 2; do for s in 0 1 2; do
  if [ -f $W/z12_ae$a/seed$s/lstm_summary.json ]; then echo "z12_ae$a seed$s already done, skipped"; continue; fi
  wait_slot
  ($PY $D/sweep.py train-lstm --train $W/ae12_seed$a/train_latent.npz --val $W/ae12_seed$a/validation_latent.npz \
       --seed $s --out $W/z12_ae$a/seed$s > $D/logs/z12_ae${a}_seed$s.log 2>&1; echo "z12_ae$a seed$s exit=$?") &
done; done
wait
echo "wall seconds: $(( $(date +%s) - start ))"
