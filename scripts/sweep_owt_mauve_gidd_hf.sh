#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 0 ]; then
  BUDGET_LIST=("$@")
else
  read -r -a BUDGET_LIST <<< "${BUDGETS:-64 128 256 512 1024}"
fi

BASE_PORT="${PORT:-29504}"

REFERENCE_FEATURES_PATH="/lustre/fsw/portfolios/coreai/users/obelhasin/yair/human_reference_mauve_featurized.npy"
CORRECTION_TEMPERATURE=0.5

for i in "${!BUDGET_LIST[@]}"; do
  BUDGET="${BUDGET_LIST[$i]}"
  JOB_PORT=$((BASE_PORT + i))
  echo "Submitting GIDD OWT MAUVE eval: budget=${BUDGET}, port=${JOB_PORT}"
  submit_job \
    --partition batch \
    --nodes 1 \
    --gpu 8 \
    --time 0 \
    --autoresume_timer 150 \
    --workdir /lustre/fsw/portfolios/coreai/users/obelhasin/workspace/gidd \
    --name llmservice_deci_llm.gidd.eval.BUDGET-${BUDGET} \
    --command "export BUDGET=${BUDGET}; export PORT=${JOB_PORT}; export REFERENCE_FEATURES_PATH=${REFERENCE_FEATURES_PATH}; export CORRECTION_TEMPERATURE=${CORRECTION_TEMPERATURE}; bash ${SCRIPT_DIR}/eval_owt_mauve_gidd_hf.sh"
done
