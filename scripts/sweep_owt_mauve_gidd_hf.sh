#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 0 ]; then
  BUDGET_LIST=("$@")
else
  read -r -a BUDGET_LIST <<< "${BUDGETS:-128 256 512 1024}"
fi

BASE_PORT="${PORT:-29504}"

for i in "${!BUDGET_LIST[@]}"; do
  BUDGET="${BUDGET_LIST[$i]}"
  JOB_PORT=$((BASE_PORT + i))
  echo "Submitting GIDD OWT MAUVE eval: budget=${BUDGET}, port=${JOB_PORT}"
  sbatch \
    --export=ALL,BUDGET="${BUDGET}",PORT="${JOB_PORT}" \
    "${SCRIPT_DIR}/eval_owt_mauve_gidd_hf.sh" \
    "${BUDGET}"
done
