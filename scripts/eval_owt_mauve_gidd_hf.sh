#!/bin/bash
#SBATCH -J gidd_owt_mauve
#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=128000
#SBATCH -t 96:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ -f setup_env.sh ]; then
  source setup_env.sh
fi

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BUDGET="${1:-${BUDGET:-}}"
if [ -z "${BUDGET}" ]; then
  echo "Usage: sbatch $0 <budget>"
  echo "   or: BUDGET=<budget> sbatch $0"
  exit 2
fi

MODEL_NAME="${MODEL_NAME:-dvruette/gidd-small-p_unif-0.2}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/owt_mauve_gidd}"
MAUVE_FEATURIZE_MODEL="${MAUVE_FEATURIZE_MODEL:-gpt2-large}"
MAUVE_MAX_TEXT_LENGTH="${MAUVE_MAX_TEXT_LENGTH:-1024}"
SKIP_MAUVE="${SKIP_MAUVE:-false}"
GEN_PPL_MODEL_NAME="${GEN_PPL_MODEL_NAME:-gpt2-large}"
GEN_PPL_BATCH_SIZE="${GEN_PPL_BATCH_SIZE:-1}"
GEN_PPL_MAX_LENGTH="${GEN_PPL_MAX_LENGTH:-1024}"
SKIP_GEN_PPL="${SKIP_GEN_PPL:-false}"
SKIP_ENTROPY="${SKIP_ENTROPY:-false}"
SERIALIZED_MODEL_LOAD="${SERIALIZED_MODEL_LOAD:-true}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-gloo}"
ENABLE_AUTORESUME="${ENABLE_AUTORESUME:-true}"
OVERWRITE="${OVERWRITE:-false}"
SKIP_EXISTING_METRICS="${SKIP_EXISTING_METRICS:-true}"
REFERENCE_FEATURES_PATH="${REFERENCE_FEATURES_PATH:-/share/kuleshov/yzs2/nvidia-collab/human_reference_mauve_featurized.npy}" #${OUTPUT_ROOT}/reference_cache/owt_refs_${MAUVE_FEATURIZE_MODEL}_num_samples-${NUM_SAMPLES}_max_len-${MAUVE_MAX_TEXT_LENGTH}.npy}"
PORT="${PORT:-$((29504 + BUDGET % 1000))}"

if [ -z "${NUM_DEVICES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_DEVICES=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    NUM_DEVICES="${SLURM_GPUS_ON_NODE}"
  elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
    NUM_DEVICES=$(awk -F',' '{print NF}' <<< "${SLURM_JOB_GPUS}")
  else
    NUM_DEVICES=8
  fi
fi

echo "Launching budget=${BUDGET} on NUM_DEVICES=${NUM_DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

torchrun --nproc_per_node "${NUM_DEVICES}" --master_port="${PORT}" gidd/eval/owt_mauve_generation.py \
  hydra.output_subdir=null \
  hydra.run.dir="${PWD}" \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled \
  model_name="${MODEL_NAME}" \
  num_samples="${NUM_SAMPLES}" \
  batch_size="${BATCH_SIZE}" \
  seed="${SEED}" \
  distributed_backend="${DISTRIBUTED_BACKEND}" \
  serialized_model_load="${SERIALIZED_MODEL_LOAD}" \
  enable_autoresume="${ENABLE_AUTORESUME}" \
  budget="${BUDGET}" \
  output_root="${OUTPUT_ROOT}" \
  overwrite="${OVERWRITE}" \
  skip_existing_metrics="${SKIP_EXISTING_METRICS}" \
  reference_features_path="${REFERENCE_FEATURES_PATH}" \
  mauve_featurize_model="${MAUVE_FEATURIZE_MODEL}" \
  mauve_max_text_length="${MAUVE_MAX_TEXT_LENGTH}" \
  skip_mauve="${SKIP_MAUVE}" \
  gen_ppl_model_name_or_path="${GEN_PPL_MODEL_NAME}" \
  gen_ppl_batch_size="${GEN_PPL_BATCH_SIZE}" \
  gen_ppl_max_length="${GEN_PPL_MAX_LENGTH}" \
  skip_gen_ppl="${SKIP_GEN_PPL}" \
  skip_entropy="${SKIP_ENTROPY}"
