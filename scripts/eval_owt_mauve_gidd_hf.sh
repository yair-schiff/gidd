#!/bin/bash
#SBATCH -J gidd_owt_mauve
#SBATCH -o %x_%A_%a.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=128000
#SBATCH -t 96:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --array=0-3
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ -f setup_env.sh ]; then
  source setup_env.sh
fi

export HYDRA_FULL_ERROR=1

BUDGETS=(128 256 512 1024)
if [ -z "${BUDGET:-}" ]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  BUDGET="${BUDGETS[$TASK_ID]}"
fi

MODEL_NAME="${MODEL_NAME:-dvruette/gidd-base-p_unif-0.2}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/owt_mauve_gidd}"
MAUVE_FEATURIZE_MODEL="${MAUVE_FEATURIZE_MODEL:-gpt2-large}"
MAUVE_MAX_TEXT_LENGTH="${MAUVE_MAX_TEXT_LENGTH:-1024}"
REFERENCE_FEATURES_PATH="${REFERENCE_FEATURES_PATH:-${OUTPUT_ROOT}/reference_cache/owt_refs_${MAUVE_FEATURIZE_MODEL}_num_samples-${NUM_SAMPLES}_max_len-${MAUVE_MAX_TEXT_LENGTH}.npy}"
PORT="${PORT:-$((29504 + ${SLURM_ARRAY_TASK_ID:-0}))}"

if [ -z "${NUM_DEVICES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_DEVICES=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_DEVICES=1
  fi
fi

torchrun --nproc_per_node "${NUM_DEVICES}" --master_port="${PORT}" gidd/eval/owt_mauve_generation.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  model_name="${MODEL_NAME}" \
  num_samples="${NUM_SAMPLES}" \
  batch_size="${BATCH_SIZE}" \
  seed="${SEED}" \
  budget="${BUDGET}" \
  output_root="${OUTPUT_ROOT}" \
  reference_features_path="${REFERENCE_FEATURES_PATH}" \
  mauve_featurize_model="${MAUVE_FEATURIZE_MODEL}" \
  mauve_max_text_length="${MAUVE_MAX_TEXT_LENGTH}"
