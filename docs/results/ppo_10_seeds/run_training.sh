#!/usr/bin/env bash
# The 19 new trainings fixed in ADDENDUM.md, at most $PARALLEL at a time (default 3, as for the existing seeds).
# Usage: PARALLEL=2 bash run_training.sh [KIND:SEED ...]   (no arguments = all 19 jobs)
# History: a first attempt with PARALLEL=3 failed for lack of system memory. direct30k seeds 4-6 died at
# 14,000 of 30,000 steps (SUMO "Could not connect" on reset, all three at the same second), and seeds 7-9 had
# just started when the attempt was stopped. Its partial outputs were discarded (kept outside the repository
# as evidence), and all 19 jobs were re-run with the same seeds and PARALLEL=2 (ADDENDUM.md: technical
# failures repeat the same seed).
# Each job: log in logs/ (not versioned), one line in runs.tsv (job, seconds, exit code).
# A job whose output folder already holds files is refused by its script, so a re-run skips nothing silently.
set -u
cd "$(dirname "$0")/../../.."
HERE=docs/results/ppo_10_seeds
mkdir -p "$HERE/logs"
PY=.venv/Scripts/python.exe

jobs() {
  if [ $# -gt 0 ]; then for j in "$@"; do echo "${j%%:*} ${j##*:}"; done; return; fi
  for s in 4 5 6 7 8 9; do echo "direct30k $s"; done
  for s in 4 5 6 7 8 9; do echo "direct10k $s"; done
  for s in 3 4 5 6 7 8 9; do echo "dream $s"; done
}

run_one() {
  kind=$1; seed=$2; start=$(date +%s)
  case $kind in
    direct30k) $PY training/train_controller_direct.py --seed "$seed" --total-timesteps 30000 \
                 --output-dir "models/checkpoints/controller_direct_30k_10seeds/seed$seed" ;;
    direct10k) $PY training/train_controller_direct.py --seed "$seed" --total-timesteps 10000 \
                 --output-dir "models/checkpoints/controller_direct_10seeds/seed$seed" ;;
    dream)     $PY "$HERE/train_dream_seed.py" --seed "$seed" \
                 --output-dir "models/checkpoints/controller_10seeds/seed$seed" ;;
  esac > "$HERE/logs/${kind}_seed${seed}.log" 2>&1
  code=$?
  printf '%s\t%s\t%s\t%s\n' "$kind" "$seed" "$(( $(date +%s) - start ))" "$code" >> "$HERE/runs.tsv"
  echo "$kind seed $seed -> exit $code"
}
export -f run_one
export HERE PY

[ -f "$HERE/runs.tsv" ] || printf 'kind\tseed\tseconds\texit\n' > "$HERE/runs.tsv"
jobs "$@" | xargs -P "${PARALLEL:-3}" -L 1 bash -c 'run_one "$0" "$1"'
echo "done: $(($(wc -l < "$HERE/runs.tsv") - 1)) jobs"
