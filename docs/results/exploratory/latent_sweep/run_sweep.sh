#!/bin/bash
# Paso 3 (MANIFEST.md): 5 autoencoders, 15 z runs (seeds 0-2), 3 raw-24 runs (seeds 0-2).
# 3 runs at a time (the first launch, with 6, ran the machine out of memory). A run that already
# finished (its *_summary.json exists) is skipped: training is deterministic, so rerunning it
# would reproduce the same weights. Interrupted runs have no summary and restart from scratch.
cd "$(dirname "$0")/../../../.."
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
MAX_PARALLEL=3
start=$(date +%s)
for k in 4 8 12 16 20; do
  if [ -f $W/ae$k/ae_summary.json ]; then echo "ae$k already done, skipped"; continue; fi
  $PY $D/sweep.py train-ae --latent $k --out $W/ae$k > $D/logs/ae$k.log 2>&1; echo "ae$k exit=$?"
done
wait_slot() { while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do sleep 5; done; }
lstm() {  # name seed train val
  if [ -f $W/$1/seed$2/lstm_summary.json ]; then echo "$1 seed$2 already done, skipped"; return; fi
  wait_slot
  ($PY $D/sweep.py train-lstm --train $3 --val $4 --seed $2 --out $W/$1/seed$2 > $D/logs/$1_seed$2.log 2>&1; echo "$1 seed$2 exit=$?") &
}
for s in 0 1 2; do lstm raw24 $s $W/data/train_raw_seq.npz $W/data/validation_raw_seq.npz; done
for k in 4 8 12 16 20; do for s in 0 1 2; do lstm z$k $s $W/ae$k/train_latent.npz $W/ae$k/validation_latent.npz; done; done
wait
echo "wall seconds this launch: $(( $(date +%s) - start ))"
