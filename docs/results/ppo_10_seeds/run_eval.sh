#!/usr/bin/env bash
# ADDENDUM.md evaluation: 10 seeds per controller + fixed time + rule, with the unmodified
# scripts/evaluate_multiseed_statistical.py, on the official scenarios and on 7000-7029 (in parallel).
set -eu
cd "$(dirname "$0")/../../.."
HERE=docs/results/ppo_10_seeds
CK=models/checkpoints
join() { local IFS=,; echo "$*"; }

DREAM=$(join $CK/controller/best_model_seed0_worse.zip $CK/controller/best_model_seed1.zip $CK/controller/best_model.zip \
             $(for s in 3 4 5 6 7 8 9; do echo $CK/controller_10seeds/seed$s/best_model.zip; done))
D10K=$(join $CK/controller_direct/best_model.zip $CK/controller_direct/best_model_seed1.zip \
            $CK/controller_direct/best_model_seed2.zip $CK/controller_direct/best_model_seed3.zip \
            $(for s in 4 5 6 7 8 9; do echo $CK/controller_direct_10seeds/seed$s/best_model.zip; done))
D30K=$(join $CK/controller_direct_30k/best_model_seed0.zip $CK/controller_direct_30k/best_model.zip \
            $CK/controller_direct_30k/best_model_seed2.zip $CK/controller_direct_30k/best_model_seed3.zip \
            $(for s in 4 5 6 7 8 9; do echo $CK/controller_direct_30k_10seeds/seed$s/best_model.zip; done))
COMMON=(--policy "sueno=dream:$DREAM" --policy "directo_10k=direct:$D10K" --policy "directo_30k=direct:$D30K"
        --policy tiempo_fijo=fixed --policy regla=rule
        --compare sueno:directo_10k --compare sueno:directo_30k --compare sueno:tiempo_fijo)

.venv/Scripts/python.exe scripts/evaluate_multiseed_statistical.py "${COMMON[@]}" \
  --seed-bases 3000 5000 --episodes-per-base 15 --output "$HERE/eval_official_scenarios" > "$HERE/logs/eval_official.log" 2>&1 &
.venv/Scripts/python.exe scripts/evaluate_multiseed_statistical.py "${COMMON[@]}" \
  --seed-bases 7000 --episodes-per-base 30 --output "$HERE/eval_fresh_scenarios_7000" > "$HERE/logs/eval_fresh.log" 2>&1 &
wait
echo "evaluations done"
