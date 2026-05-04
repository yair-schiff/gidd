#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 0 ]; then
  BUDGET_LIST=("$@")
else
  read -r -a BUDGET_LIST <<< "${BUDGETS:-64 128 256 512 1024}"
fi

BASE_PORT="${PORT:-29504}"

# REFERENCE_FEATURES_PATH="/lustre/fsw/portfolios/coreai/users/obelhasin/yair/human_reference_mauve_featurized.npy"
TEMPERATURE_LIST=(0.02 0.04 0.06 0.08)

JOB_IDX=0
for CORRECTION_TEMPERATURE in "${TEMPERATURE_LIST[@]}"; do
  for i in "${!BUDGET_LIST[@]}"; do
    BUDGET="${BUDGET_LIST[$i]}"
    JOB_PORT=$((BASE_PORT + JOB_IDX))
    echo "Submitting GIDD OWT MAUVE eval: budget=${BUDGET}, temp=${CORRECTION_TEMPERATURE}, port=${JOB_PORT}"
    submit_job \
      --partition batch \
      --nodes 1 \
      --gpu 8 \
      --time 0 \
      --autoresume_timer 150 \
      --workdir /lustre/fsw/portfolios/coreai/users/obelhasin/workspace/gidd \
      --name llmservice_deci_llm.gidd.eval.BUDGET-${BUDGET}.TEMP-${CORRECTION_TEMPERATURE} \
      --command "export BUDGET=${BUDGET}; export PORT=${JOB_PORT}; export CORRECTION_TEMPERATURE=${CORRECTION_TEMPERATURE}; bash ${SCRIPT_DIR}/eval_owt_mauve_gidd_hf.sh"
    JOB_IDX=$((JOB_IDX + 1))
  done
done
