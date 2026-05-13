#!/bin/bash
#SBATCH --job-name=insurerag-dense-train
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#SBATCH --requeue

set -euo pipefail

REPO_ROOT="/users/2/wan00232/insurance_RAG_VLM/InsureRAG-VLM"
SCRATCH_ROOT="${SCRATCH:-/scratch.global}"
VENV_DIR="${VENV_DIR:-${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/.venv_sft311}"
DATASET_PATH="${DATASET_PATH:-reports/training_data_seed/retrieval_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/retrieval/bge-base-insurerag}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_LENGTH="${MAX_LENGTH:-384}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SEED="${SEED:-42}"
MARGIN="${MARGIN:-0.2}"
USE_FP16="${USE_FP16:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

cd "${REPO_ROOT}"

mkdir -p logs/slurm "$(dirname "${OUTPUT_DIR}")"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

CACHE_ROOT="${SCRATCH_ROOT}/${USER}/insurance_RAG_VLM/cache"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${PIP_CACHE_DIR}" "${TMPDIR}"

if python - <<'PY'
missing = []
for module_name in ("torch", "transformers", "numpy"):
    try:
        __import__(module_name)
    except Exception:
        missing.append(module_name)
if missing:
    raise SystemExit("MISSING:" + ",".join(missing))
print("Dense-retriever Python dependencies already available.")
PY
then
  echo "Dependencies already present."
else
  echo "Installing project dependencies into ${VENV_DIR}"
  python -m pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt
fi

ARGS=(
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --max-length "${MAX_LENGTH}"
  --learning-rate "${LEARNING_RATE}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --logging-steps "${LOGGING_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --seed "${SEED}"
  --margin "${MARGIN}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${USE_FP16}" == "1" ]]; then
  ARGS+=(--fp16)
fi
if [[ "${AUTO_RESUME}" == "1" ]]; then
  ARGS+=(--auto-resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

srun python main.py train-dense-retriever "${ARGS[@]}"
