#!/usr/bin/env bash
# Prepare a local development environment for InsureRAG-VLM.
set -euo pipefail

PYTHON_BIN=${PYTHON:-}

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "ERROR: Python 3 is required but was not found on PATH."
    exit 1
  fi
fi

echo "=== InsureRAG-VLM local setup ==="
echo "Using Python: $($PYTHON_BIN --version)"

if [ ! -d ".venv" ]; then
  echo "[1/4] Creating .venv..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "[1/4] Reusing existing .venv..."
fi

if [ -x ".venv/bin/python" ]; then
  VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  echo "ERROR: Could not find the virtualenv Python executable."
  exit 1
fi

echo "[2/4] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "[3/4] Installing project dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "[4/4] Checking bundled sample data..."
if [ -f "data/00_raw/public/synthetic_auto_policy.pdf" ]; then
  echo "Found synthetic sample policy PDF."
else
  echo "WARNING: data/00_raw/public/synthetic_auto_policy.pdf was not found."
  echo "The app can still run, but the bundled smoke test needs sample data."
fi

cat <<'EOF'

Local setup complete.

Activate the environment:
  source .venv/bin/activate

Run the smoke test:
  make smoke-test

Start the browser demo:
  python main.py demo-web --port 7860

Then open:
  http://127.0.0.1:7860
EOF
