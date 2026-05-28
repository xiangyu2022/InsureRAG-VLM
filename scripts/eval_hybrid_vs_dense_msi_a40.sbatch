#!/bin/bash
#SBATCH --job-name=insurerag-hybrid-vs-dense
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

REPO_ROOT="/users/2/wan00232/insurance_RAG_VLM/InsureRAG-VLM"
SCRATCH_ROOT="${SCRATCH:-/scratch.global}"
VENV_DIR="${VENV_DIR:-${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/.venv_sft311}"

cd "${REPO_ROOT}"
mkdir -p logs/slurm reports/retrieval_eval

source "${VENV_DIR}/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/cache/pip}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/cache/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

srun python scripts/eval_hybrid_vs_dense.py "$@"
