#!/bin/bash
#SBATCH --job-name=insurerag-sft
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
cd "${REPO_ROOT}"

mkdir -p logs/slurm models

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
export PYTHONUNBUFFERED=1

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${PIP_CACHE_DIR}" "${TMPDIR}"

if python - <<'PY'
missing = []
for module_name in (
    "numpy",
    "pandas",
    "pyarrow",
    "PIL",
    "requests",
    "openai",
    "pytesseract",
    "fitz",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
):
    try:
        __import__(module_name)
    except Exception:
        missing.append(module_name)
if missing:
    raise SystemExit("MISSING:" + ",".join(missing))
print("GPU SFT Python dependencies already available.")
PY
then
  echo "Dependencies already present."
else
  echo "Installing project and GPU SFT dependencies into ${VENV_DIR}"
  python -m pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt
fi

srun python main.py sft-lora-qwen \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --output-dir models/qwen7b-insurerag-lora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 10 \
  --save-steps 50 \
  --save-total-limit 4 \
  --auto-resume \
  "$@"
