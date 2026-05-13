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

Download real public PDFs, then run the no-key smoke test:

```bash
python main.py import-data --output-root data --datasets public_docs
```

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
- `Query Public Docs`: runs a deterministic local query against downloaded public PDFs.
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
python main.py import-data --output-root data --datasets public_docs
python main.py build-index data/00_raw/external/public_docs --index-dir data --retrieval-mode hybrid_multimodal
python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 150
```

The default application path is now the hybrid multimodal retriever:
- dense text retrieval
- sparse text retrieval
- rule-based query understanding
- table-aware retrieval and graph expansion
- lightweight page-image auxiliary scoring
- insurance-logic reranking, page-level context packing, and cited answering

`build-visual-index` is still available, but it is now an optional research or comparison path rather than a prerequisite for normal local querying.

## Optional GPU Setup

Only install the GPU stack on a CUDA-capable machine:

```bash
pip install -r requirements-gpu.txt
```

The normal CPU/local baseline does not require GPU dependencies or API keys.

Windows PowerShell quick CUDA check:

```powershell
py -3 -m venv .venv-cu
.\.venv-cu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
python scripts\gpu_smoke_test.py
```

Expected result includes `cuda_available=True`, your NVIDIA GPU name, and
`gpu_smoke_test=PASSED`.
