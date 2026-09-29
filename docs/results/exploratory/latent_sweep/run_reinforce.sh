#!/bin/bash
# Paso 4 (MANIFEST.md): seeds 3-9 for the winning latent_dim (z) and for raw-24; 3 at a time.
# Usage: run_reinforce.sh K
cd "$(dirname "$0")/../../../.."
K=$1; [ -n "$K" ] || { echo "usage: $0 K"; exit 2; }
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
MAX_PARALLEL=3
start=$(date +%s)
wait_slot() { while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do sleep 5; done; }
lstm() {  # name seed train val
  if [ -f $W/$1/seed$2/lstm_summary.json ]; then echo "$1 seed$2 already done, skipped"; return; fi
  wait_slot
  ($PY $D/sweep.py train-lstm --train $3 --val $4 --seed $2 --out $W/$1/seed$2 > $D/logs/$1_seed$2.log 2>&1; echo "$1 seed$2 exit=$?") &
}
for s in 3 4 5 6 7 8 9; do
  lstm z$K $s $W/ae$K/train_latent.npz $W/ae$K/validation_latent.npz
  lstm raw24 $s $W/data/train_raw_seq.npz $W/data/validation_raw_seq.npz
done
wait
echo "wall seconds: $(( $(date +%s) - start ))"
$PY $D/sweep.py compare --z-dirs $W/z$K/seed{0..9} --raw-dirs $W/raw24/seed{0..9} \
    --test-z $W/ae$K/test_latent.npz --test-raw $W/data/test_raw_seq.npz --output $D/winner_z${K}_10seeds.json
