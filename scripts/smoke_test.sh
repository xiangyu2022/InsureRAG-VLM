#!/usr/bin/env bash
# End-to-end smoke test — no API keys required.
# Uses bundled synthetic policy PDFs so results are fully reproducible.
set -euo pipefail

PYTHON=${PYTHON:-python}
SMOKE_DIR=$(mktemp -d)
trap "rm -rf $SMOKE_DIR" EXIT

echo "=== InsureRAG-VLM smoke test ==="
echo "Python: $($PYTHON --version)"
echo "Temp dir: $SMOKE_DIR"

# 1. Build text index from synthetic policies
echo "[1/5] Building text index..."
$PYTHON main.py build-index data/00_raw/public --index-dir "$SMOKE_DIR"

# 2. Generate QA pairs from synthetic policies
echo "[2/5] Generating QA pairs..."
$PYTHON main.py generate-qa data/00_raw/public --output-dir "$SMOKE_DIR"

# 3. Run retrieval metrics (no LLM, no API keys)
echo "[3/5] Computing retrieval metrics..."
$PYTHON main.py retrieval-metrics data/00_raw/public "$SMOKE_DIR/qa_pairs.jsonl" \
  --index-dir "$SMOKE_DIR"

# 4. Run deterministic query (local-extractive, no LLM)
echo "[4/5] Running deterministic query..."
INSURERAG_USE_OLLAMA=0 $PYTHON main.py query \
  data/00_raw/public \
  "What is the liability coverage limit?" \
  --index-dir "$SMOKE_DIR"

# 5. Validate output files exist
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
