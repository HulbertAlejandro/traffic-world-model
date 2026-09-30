#!/bin/bash
# ADDENDUM_transformer_tsmixer.md. Usage: run_multiarch.sh A   (AEs 16 seeds 1-4, LSTM@16 grid, Transformer)
#                                         run_multiarch.sh B   (TSMixer)
# 300-epoch cap and patience 15 for every run; 3 at a time; finished runs (summary present) are skipped.
cd "$(dirname "$0")/../../../.."
PART=$1; [ "$PART" = A ] || [ "$PART" = B ] || { echo "usage: $0 A|B"; exit 2; }
D=docs/results/exploratory/latent_sweep; W=models/checkpoints/exploratory_latent_sweep; PY=.venv/Scripts/python.exe
MAX_PARALLEL=3
start=$(date +%s)
aedir() { [ $1 = 0 ] && echo $W/ae16 || echo $W/ae16_seed$1; }
wait_slot() { while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do sleep 5; done; }
run() {  # out log cmd...
  local out=$1 log=$2; shift 2
  if [ -f $out/lstm_summary.json ] || [ -f $out/train_summary.json ]; then echo "$out already done, skipped"; return; fi
  wait_slot
  ("$@" > $D/logs/$log.log 2>&1; echo "$log exit=$?") &
}
if [ $PART = A ]; then
  for a in 1 2 3 4; do
    if [ -f $W/ae16_seed$a/ae_summary.json ]; then echo "ae16_seed$a already done, skipped"; continue; fi
    $PY $D/sweep.py train-ae --latent 16 --seed $a --out $W/ae16_seed$a > $D/logs/ae16_seed$a.log 2>&1; echo "ae16_seed$a exit=$?"
  done
  for a in 0 1 2 3 4; do for s in 0 1 2 3 4; do
    z=$([ $a = 0 ] && echo z16 || echo z16_ae$a)
    run $W/$z/seed$s ${z}_seed$s $PY $D/sweep.py train-lstm --train $(aedir $a)/train_latent.npz --val $(aedir $a)/validation_latent.npz --seed $s --out $W/$z/seed$s
  done; done
  ARCH=transformer; P=tf
else
  ARCH=tsmixer; P=ts
fi
for a in 0 1 2 3 4; do for s in 0 1 2 3 4; do
  run $W/${P}16_ae$a/seed$s ${P}16_ae${a}_seed$s $PY $D/sweep.py train-arch --arch $ARCH --train $(aedir $a)/train_latent.npz --val $(aedir $a)/validation_latent.npz --seed $s --epochs 300 --out $W/${P}16_ae$a/seed$s
done; done
for s in 0 1 2 3 4 5 6 7 8 9; do
  run $W/${P}_raw24/seed$s ${P}_raw24_seed$s $PY $D/sweep.py train-arch --arch $ARCH --train $W/data/train_raw_seq.npz --val $W/data/validation_raw_seq.npz --seed $s --epochs 300 --out $W/${P}_raw24/seed$s
done
wait
echo "part $PART wall seconds: $(( $(date +%s) - start ))"
