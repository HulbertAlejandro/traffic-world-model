#!/bin/bash
# Trains the 10 pre-registered runs (5 seeds x 2 branches), 5 at a time.
cd "$(dirname "$0")/../../../.."
C=models/checkpoints/exploratory_seed_zero_check
L=docs/results/exploratory/seed_zero_check/logs
mkdir -p $L
run() { .venv/Scripts/python.exe training/$1 --seed $2 --epochs 300 --output-dir $C/$3/seed$2 > $L/$3_seed$2.log 2>&1; echo "$3 seed$2 exit=$?"; }
for s in 10 20 30 40 50; do run train_world_model.py $s z & done; wait
for s in 10 20 30 40 50; do run train_world_model_raw.py $s raw & done; wait
