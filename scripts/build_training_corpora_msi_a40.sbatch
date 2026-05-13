#!/bin/bash
#SBATCH --job-name=insurerag-build-corpora
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
DATA_FOLDER="${DATA_FOLDER:-data/04_curated}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/training_data_seed}"
INDEX_DIR="${INDEX_DIR:-reports/training_data_seed/index}"
SFT_DATASET_PATH="${SFT_DATASET_PATH:-data/04_curated/sft_dataset.jsonl}"
QA_PATH="${QA_PATH:-}"
HARD_NEGATIVES_PATH="${HARD_NEGATIVES_PATH:-}"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-local-hashing}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid_multimodal}"
CORPUS_SOURCE="${CORPUS_SOURCE:-curated}"
TARGET_QA_COUNT="${TARGET_QA_COUNT:-300}"
UNSUPPORTED_COUNT="${UNSUPPORTED_COUNT:-50}"
TOP_K="${TOP_K:-5}"
MAX_NEGATIVES="${MAX_NEGATIVES:-4}"
MAX_SFT_PAGES="${MAX_SFT_PAGES:-3}"
DISABLE_IMAGE_SIGNAL="${DISABLE_IMAGE_SIGNAL:-1}"

cd "${REPO_ROOT}"

mkdir -p logs/slurm

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
):
    try:
        __import__(module_name)
    except Exception:
        missing.append(module_name)
if missing:
    raise SystemExit("MISSING:" + ",".join(missing))
print("Training-corpora Python dependencies already available.")
PY
then
  echo "Dependencies already present."
else
  echo "Installing project dependencies into ${VENV_DIR}"
  python -m pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt
fi

ARGS=(
  --data-folder "${DATA_FOLDER}"
  --output-dir "${OUTPUT_DIR}"
  --index-dir "${INDEX_DIR}"
  --sft-dataset-path "${SFT_DATASET_PATH}"
  --retrieval-model "${RETRIEVAL_MODEL}"
  --retrieval-mode "${RETRIEVAL_MODE}"
  --corpus-source "${CORPUS_SOURCE}"
  --target-qa-count "${TARGET_QA_COUNT}"
  --unsupported-count "${UNSUPPORTED_COUNT}"
  --top-k "${TOP_K}"
  --max-negatives "${MAX_NEGATIVES}"
  --max-sft-pages "${MAX_SFT_PAGES}"
)

if [[ -n "${QA_PATH}" ]]; then
  ARGS+=(--qa-path "${QA_PATH}")
fi
if [[ -n "${HARD_NEGATIVES_PATH}" ]]; then
  ARGS+=(--hard-negatives-path "${HARD_NEGATIVES_PATH}")
fi
if [[ "${DISABLE_IMAGE_SIGNAL}" == "1" ]]; then
  ARGS+=(--disable-image-signal)
fi

srun python main.py build-training-corpora "${ARGS[@]}"
