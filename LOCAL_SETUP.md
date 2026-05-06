# Local Setup

This guide moves day-to-day development out of GitHub Codespaces and onto your
own machine.

## 1. Clone The Repo

```bash
git clone <your-repo-url>
cd InsureRAG-VLM
```

If you already have the repo locally, pull the latest branch instead:

```bash
git pull
```

## 2. Create The Python Environment

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or use the helper script on macOS/Linux:

```bash
bash scripts/setup_local.sh
```

## 3. Verify The Project

Run the no-key smoke test:

```bash
make smoke-test
```

If `make` is not available, run:

```bash
bash scripts/smoke_test.sh
```

## 4. Start The Browser Demo

```bash
python main.py demo-web --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

For the deterministic local demo without Ollama:

```bash
INSURERAG_USE_OLLAMA=0 python main.py demo-web --port 7860
```

## 5. Debug In VS Code

Open this folder in VS Code and select the `.venv` interpreter.

The repo includes `.vscode/launch.json` with these debug targets:

- `Demo Web`: starts the browser app on port `7860`.
- `Query Synthetic Policy`: runs a deterministic local query.
- `Smoke Test`: runs the project smoke test script.

## 6. Move Data From Codespaces

Generated indexes, downloaded PDFs, and private PDFs are usually not committed to
Git. Copy any local-only data you care about from Codespaces before deleting it.

Useful folders:

```text
data/00_raw/internal/
data/00_raw/external/
data/01_interim/
data/02_processed/
data/03_index/
reports/
```

If you do not copy generated files, you can rebuild them locally:

```bash
python main.py build-index data/00_raw/public --index-dir data
python main.py preprocess-pages data/00_raw/public --output-root data --render-dpi 150
python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend local_image
```

## Optional GPU Setup

Only install the GPU stack on a CUDA-capable machine:

```bash
pip install -r requirements-gpu.txt
```

The normal CPU/local baseline does not require GPU dependencies or API keys.
