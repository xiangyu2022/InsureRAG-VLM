#!/bin/bash
#SBATCH -J insurerag-external-hvd
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT=/users/2/wan00232/insurance_RAG_VLM/InsureRAG-VLM
VENV=/scratch.global/wan00232/insurance_RAG_VLM/.venv_sft311

cd "$REPO_ROOT"
mkdir -p logs/slurm reports/retrieval_eval

export HF_HOME=/scratch.global/wan00232/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch.global/wan00232/.cache/huggingface/transformers
export PIP_CACHE_DIR=/scratch.global/wan00232/.cache/pip
export TMPDIR=/scratch.global/wan00232/.tmp
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

source "$VENV/bin/activate"

PYTHONPATH="$REPO_ROOT" srun python scripts/eval_external_official_hybrid_vs_dense.py "$@"
