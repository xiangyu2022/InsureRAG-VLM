#!/usr/bin/env bash
# End-to-end smoke test with real public PDFs.
# Download them first with:
#   python main.py import-data --output-root data --datasets public_docs
set -euo pipefail

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-data/00_raw/external/public_docs}
SMOKE_DIR=$(mktemp -d)
trap "rm -rf $SMOKE_DIR" EXIT

echo "=== InsureRAG-VLM smoke test ==="
echo "Python: $($PYTHON --version)"
echo "Data: $DATA_DIR"
echo "Temp dir: $SMOKE_DIR"

if [ ! -d "$DATA_DIR" ] || ! find "$DATA_DIR" -name '*.pdf' -print -quit | grep -q .; then
  echo "FAIL: real public PDFs were not found in $DATA_DIR"
  echo "Run: $PYTHON main.py import-data --output-root data --datasets public_docs"
  exit 1
fi

echo "[1/5] Building text index..."
$PYTHON main.py build-index "$DATA_DIR" --index-dir "$SMOKE_DIR"

echo "[2/5] Generating QA pairs..."
$PYTHON main.py generate-qa "$DATA_DIR" --output-dir "$SMOKE_DIR" --target-count 50

echo "[3/5] Computing retrieval metrics..."
$PYTHON main.py retrieval-metrics "$DATA_DIR" "$SMOKE_DIR/qa_pairs.jsonl" \
  --index-dir "$SMOKE_DIR"

echo "[4/5] Running deterministic query..."
INSURERAG_USE_OLLAMA=0 $PYTHON main.py query \
  "$DATA_DIR" \
  "What coverage limits or deductibles are described in the documents?" \
  --index-dir "$SMOKE_DIR"

echo "[5/5] Validating outputs..."
for f in "$SMOKE_DIR/qa_pairs.jsonl" "$SMOKE_DIR/hard_negatives.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "FAIL: missing $f"
    exit 1
  fi
  lines=$(wc -l < "$f")
  echo "  $f: $lines lines"
done

echo "=== Smoke test PASSED ==="
