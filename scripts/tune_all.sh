#!/bin/bash
# Sequentially tune every (model, dataset) cell of the paper's full benchmark
# table. Resumes from any existing Optuna journal logs in results/ (each
# tune.py invocation calls load_if_exists=True on the study).
#
# Usage:
#   bash scripts/tune_all.sh
#
# Or to tune one model across all datasets:
#   bash scripts/tune_all.sh XGB_bb

set -e
cd "$(dirname "$0")/.."

# 27 CTR23 datasets used in the paper (sorted ascending by n*p, up to and
# including kings_county, excluding Moneyball).
TASK_IDS=(
  361618  # forest_fires
  361619  # student_performance_por
  361617  # energy_efficiency
  361622  # cars
  361621  # QSAR_fish_toxicity
  361237  # concrete_compressive_strength
  361264  # socmob
  361244  # solar_flare
  361243  # geographical_origin_of_music
  361235  # airfoil_self_noise
  361236  # auction_verification
  361623  # space_ga
  361234  # abalone
  361249  # white_wine
  361250  # red_wine
  361258  # kin8nm
  361256  # cpu_activity
  361259  # pumadyn32nh
  361267  # brazilian_houses
  361251  # grid_stability
  361260  # miami_housing
  361255  # california_housing
  361272  # fifa
  361247  # naval_propulsion_plant
  361261  # cps88wages
  361269  # health_insurance
  361266  # kings_county
)

# Note: TSL_1product (the TSLRegressorOneTensor ablation) is omitted; that
# Python package is not publicly released at this time. Its precomputed
# results are still in results/interpretable/tsl_1product/ and contribute to
# the regenerated summary table.
MODELS=(EBM LGBM_interp XGB_interp SepALS_r2 SepALS_r10 TSL_R2 TSL_R10 LGBM_bb RF_bb XGB_bb)

# Optional: filter to a single model passed as $1
if [ $# -ge 1 ]; then
  MODELS=("$1")
fi

for model in "${MODELS[@]}"; do
  for tid in "${TASK_IDS[@]}"; do
    echo "=== model=${model}  task_id=${tid} ==="
    python scripts/tune.py --model "$model" --task-id "$tid" || echo "  (failed, continuing)"
  done
done
