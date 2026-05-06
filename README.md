# InsureRAG-VLM

Citation-grounded multimodal RAG for insurance policy and endorsement review. The project is designed around page-image retrieval first: PDFs are rendered into full-page visual units for ColQwen2/ColPali-style retrieval, while OCR and text layers are kept as auxiliary evidence for snippets, weak labels, baselines, and error analysis.

## Demo Walkthrough

![InsureRAG-VLM animated demo](assets/demo/insurerag_vlm_demo.gif)

The browser demo shows the core workflow:

- Upload or use a policy PDF.
- Render each PDF page into a page-image retrieval unit.
- Ask an insurance question, such as a deductible, limit, endorsement, exclusion, or glossary question.
- Retrieve ranked pages and show the retrieval trace.
- Return a structured answer with page-level citation, confidence, and highlighted evidence.
- Abstain when the retrieved evidence does not support the question.
- Compare policy versions and summarize deductible, limit, endorsement, and exclusion drift.

Run the interactive version locally with:

```bash
.venv/bin/python main.py demo-web --port 7860
```

Then open `http://127.0.0.1:7860`.

## What It Does

Input: a 20-200 page policy packet, endorsement, claim document, or internal training PDF plus a user question.

Output:
- An answer grounded in retrieved pages.
- Page-level citations and evidence snippets.
- Clause/version diff summaries for policy packet comparisons.
- Abstention when the retrieved evidence is insufficient.

## Why This Project

Insurance documents are visually rich. Coverage limits, deductibles, declarations, exclusions, endorsements, and version drift often depend on tables, layout, section hierarchy, and page-level context. OCR-only chunking can lose these signals. This project uses the page image as the primary retrieval object and keeps OCR/layout/table extraction as an explanation and evaluation layer.

## Architecture

Primary path:

```text
PDF / policy packet
  -> document registry + hashes
  -> render each page as a standardized image
  -> page-level metadata
  -> ColQwen2/ColPali-compatible page manifest
  -> visual page retrieval
  -> Qwen2.5-VL grounded answer with citations
```

Auxiliary path:

```text
text layer / lightweight OCR / layout blocks / tables
  -> weak QA labels
  -> citation snippets
  -> keyword filters
  -> OCR-text RAG baseline
  -> error analysis and policy diff support
```

## Data

Recommended public-data mix:
- CUAD for clause extraction and QA supervision.
- ACORD for clause retrieval and hard negatives.
- InsuranceQA for insurance question language style.
- FUNSD/CORD as form and structured-document proxies.
- Public policy PDFs for end-to-end page-image demos.

Private or internal policy files should stay outside Git. Use `data/00_raw/internal/` locally and keep only manifests, hashes, and processing scripts in the repository.

## Models

- Retriever: ColQwen2/ColPali-style page-image retrieval.
- Generator: Qwen2.5-VL-7B-Instruct with future LoRA/QLoRA fine-tuning.
- Baselines: OCR-text RAG, Florence-2, PaliGemma 2.
- Calibration: selective prediction / conformal abstention for unsupported questions.

## Current Status

Implemented:
- PDF page rendering and page-image preprocessing.
- Document registry with SHA-256 hashes.
- Page-level metadata with page IDs, dimensions, render DPI, source, version hints, and image hashes.
- Auxiliary text-layer/OCR metadata.
- ColQwen2-compatible `page_manifest.jsonl`.
- Local image-aware retrieval backend (`local_image`) that reads page images and preserves the future ColPali/ColQwen2 output schema.
- Weak query-page training pairs from section hints.
- Document-level train/valid/test splits.
- Public PDF data importer with 20 downloadable state insurance department PDFs.
- 200-500 scale real-PDF QA/evidence generation with QA-level document split labels.
- Deterministic insurance glossary with 250+ terms, acronyms, metrics, roles, and aliases.
- No-key local demo baseline with hashing retrieval and extractive cited answers.
- Synthetic insurance policy PDF and evaluation examples for immediate smoke tests.
- Existing text-RAG scaffold, PDF extraction, evaluation helpers, and policy diff utilities.
- Calibration report for selective answering and unsupported-question abstention.
- Animated browser demo for cited answers, cited-page thumbnails, highlighted evidence snippets, retrieval trace, abstention, upload flow, and policy-version diff.
- Optional Hugging Face Transformers GPU backends for `colqwen2_hf` / `colqwen2_local` and `colpali_hf` / `colpali_local`.

Still planned:
- Measured CUDA results for ColQwen2/ColPali beyond the current `local_image` baseline.
- Layout/table extraction as explanation and evaluation metadata.
- Multi-file upload sessions. The current browser demo can upload and index one local PDF for Q&A, while the policy-diff animation uses the bundled v1/v2 sample pair.

## Current Reproducible Results

### Real Public PDF Evaluation

Latest local run: **20 public insurance PDFs**, **168 rendered pages**, and **303 QA rows**
from Maryland and North Carolina insurance department documents. The QA set contains
300 answerable examples plus 3 unsupported examples, with document-level split labels.

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text (text + hashing) | 0.2033 | 0.4467 | 0.2974 | 0.3348 |
| visual_stub | 0.1733 | 0.4400 | 0.2733 | 0.3149 |
| local_image (image-aware local baseline) | 0.1733 | 0.4400 | 0.2733 | 0.3149 |
| colqwen2_local / colpali_local (GPU) | pending CUDA run | pending CUDA run | pending CUDA run | pending CUDA run |

Answering and calibration reports from the same run:

| Metric | Value |
| --- | ---: |
| Extractive answer F1 | 0.3695 |
| Citation precision | 0.2533 |
| Evidence recall | 0.2533 |
| Unsupported abstention accuracy | 0.6667 |
| Calibration suggested threshold | 0.40 |
| Calibration coverage at threshold | 0.6898 |

These are local baselines, not final benchmark claims. The close `local_text` vs `local_image`
gap is useful as a sanity check; the next meaningful result is a CUDA run with real
ColQwen2/ColPali embeddings. See `reports/ablation_real_pdfs/`,
`reports/calibration_real_pdfs/`, and `notebooks/colqwen2_gpu_embed.ipynb`.

### Synthetic Policy Smoke Test (reproducibility check)

Local smoke-test results on the bundled synthetic policy QA set:

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text | 0.3125 | 1.0000 | 0.5958 | 0.6978 |
| visual_stub | 0.3125 | 0.6250 | 0.4479 | 0.4933 |
| local_image | 0.3125 | 0.6250 | 0.4479 | 0.4933 |

These numbers are reproducibility checks on synthetic data, not benchmark claims.

## Quickstart

```bash
pip install -r requirements.txt

# End-to-end smoke test (no API keys needed)
make smoke-test

# Or manually:
python main.py build-index data/00_raw/public --index-dir data
python main.py query data/00_raw/public "What is the comprehensive deductible?" --index-dir data --top-k 3
```

Run the animated browser demo:

```bash
.venv/bin/python main.py preprocess-pages data/00_raw/public --output-root data --render-dpi 150
.venv/bin/python main.py generate-qa data/00_raw/public --output-dir data/02_processed
.venv/bin/python main.py build-index data/00_raw/public --index-dir data
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend local_image
.venv/bin/python main.py demo-web --port 7860
```

Then open `http://127.0.0.1:7860`. The UI lets you upload a PDF, asks questions against the active indexed file, and shows grounded citations, confidence/abstention, policy diff, and an expandable retrieval trace with top ranked pages.

Import a small real public PDF set for local experiments:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 150
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend local_image
```

`public_docs` downloads a small set of public insurance PDF samples into `data/00_raw/external/public_docs/`. These files are for local reproducibility and are not intended to be committed to GitHub.

Render PDFs into a page-image dataset:

```bash
.venv/bin/python main.py preprocess-pages data/00_raw/public --output-root data --render-dpi 200
```

Optional OCR auxiliary metadata:

```bash
.venv/bin/python main.py preprocess-pages data/00_raw/public --output-root data --render-dpi 200 --run-ocr
```

Expected outputs:

```text
data/01_interim/page_images/
  <doc_id>_p0001.png
  <doc_id>_p0002.png

data/02_processed/
  documents.parquet
  documents.jsonl
  pages.parquet
  pages.jsonl
  ocr_aux.jsonl
  colqwen2_retrieval_pairs.jsonl
  splits.parquet
  splits.jsonl

data/03_index/colqwen2/
  page_embeddings/
  page_manifest.jsonl

data/manifests/
  preprocess_run_manifest.jsonl
```

Run the older text-RAG scaffold:

```bash
.venv/bin/python main.py build-index data/00_raw/public --render-pdf-pages --pdf-render-dir data/01_interim/legacy_pdf_pages
.venv/bin/python main.py query data/00_raw/public "What does the endorsement cover?"
```

Run a small evaluation smoke test:

```bash
.venv/bin/python main.py evaluate data/00_raw/public data/00_raw/public/synthetic_eval_examples.json --index-dir data --top-k 3
```

Import external CUAD/ACORD manifests and local files:

```bash
.venv/bin/python main.py import-data --output-root data --datasets cuad acord insuranceqa public_docs
```

Generate QA/evidence pairs and hard negatives:

```bash
.venv/bin/python main.py generate-qa data/00_raw/public \
  --output-dir data/02_processed \
  --cuad-master data/00_raw/external/cuad/master_clauses.csv \
  --acord-root data/00_raw/external/acord/extracted \
  --max-cuad 500 \
  --max-acord 500
```

Compute retrieval metrics over generated policy QA:

```bash
.venv/bin/python main.py retrieval-metrics data/00_raw/public data/02_processed/qa_pairs.jsonl --index-dir data --top-k 3
```

Build and evaluate the page-image retrieval interface:

```bash
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend visual_stub
.venv/bin/python main.py visual-retrieval-metrics data/02_processed/qa_pairs.jsonl --index-dir data/03_index/colqwen2 --backend visual_stub --top-k 3
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend local_image
.venv/bin/python main.py visual-retrieval-metrics data/02_processed/qa_pairs.jsonl --index-dir data/03_index/colqwen2 --backend local_image --top-k 3
```

Run a real ColQwen2 / ColPali visual retriever on a GPU machine:

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt

# ColQwen2 via Hugging Face Transformers
export INSURERAG_COLQWEN2_MODEL="vidore/colqwen2-v1.0-hf"
export INSURERAG_VISUAL_BATCH_SIZE=2
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl \
  --index-dir data/03_index/colqwen2 \
  --backend colqwen2_hf
.venv/bin/python main.py visual-retrieval-metrics data/02_processed/qa_pairs.jsonl \
  --index-dir data/03_index/colqwen2 \
  --backend colqwen2_hf \
  --top-k 5

# ColPali alternative
export INSURERAG_COLPALI_MODEL="vidore/colpali-v1.3-hf"
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl \
  --index-dir data/03_index/colqwen2 \
  --backend colpali_hf
```

The `colqwen2_hf` and `colpali_hf` backends use Hugging Face Transformers retrieval classes and save multi-vector page embeddings to `data/03_index/colqwen2/<backend>.pt`. Set `INSURERAG_REQUIRE_CUDA=1` if you want the command to fail instead of falling back to CPU/MPS.

Generate a 200-500 example real-PDF QA/evidence set:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py generate-qa data/00_raw/external/public_docs \
  --output-dir reports/public_docs_qa_v3 \
  --target-count 300
.venv/bin/python main.py build-index data/00_raw/external/public_docs \
  --index-dir reports/public_docs_qa_v3/index
.venv/bin/python main.py retrieval-metrics data/00_raw/external/public_docs \
  reports/public_docs_qa_v3/qa_pairs.jsonl \
  --index-dir reports/public_docs_qa_v3/index \
  --top-k 5
```

The real-PDF QA generator creates evidence-anchored questions, supports multi-positive page labels for broad topics, and writes `qa_splits.jsonl` plus per-example `split` / `split_doc_id` fields to keep evaluation document-aware. The GPU ColQwen2/ColPali run should be reported separately once executed on a CUDA machine.

Run the ablation harness:

```bash
.venv/bin/python main.py run-ablation \
  --data-folder data/00_raw/external/public_docs \
  --qa-path reports/public_docs_qa_v3/qa_pairs.jsonl \
  --output-dir reports/ablation_real_pdfs \
  --index-dir reports/public_docs_qa_v3/index \
  --visual-index-dir data/03_index/colqwen2 \
  --top-k 5
```

Run the calibration and selective-abstention report:

```bash
.venv/bin/python main.py run-calibration \
  --data-folder data/00_raw/external/public_docs \
  --qa-path reports/public_docs_qa_v3/qa_pairs.jsonl \
  --output-dir reports/calibration_real_pdfs \
  --index-dir reports/public_docs_qa_v3/index \
  --top-k 5
```

Outputs:

```text
reports/calibration_real_pdfs/calibration_scores.jsonl
reports/calibration_real_pdfs/calibration_curve.csv
reports/calibration_real_pdfs/summary.md
```

Emit structured grounded-answer JSON:

```bash
.venv/bin/python main.py query data/00_raw/public "What is the comprehensive deductible?" --index-dir data --top-k 3 --json
```

Summarize policy version drift:

```bash
.venv/bin/python main.py policy-diff data/00_raw/public/synthetic_auto_policy.pdf data/00_raw/public/synthetic_auto_policy_v2.pdf --output reports/diff/diff_summary.json
```

Use OpenAI instead of the local baseline by setting:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
```

Use a local Ollama model for open-ended questions:

```bash
ollama pull qwen2.5:3b
export OLLAMA_MODEL="qwen2.5:3b"
export OLLAMA_NUM_PREDICT=384
export OLLAMA_NUM_CTX=4096
python main.py demo-web --port 7860
```

For the fastest deterministic demo, bypass Ollama entirely:

```bash
export INSURERAG_USE_OLLAMA=0
python main.py demo-web --port 7860
```

Token-efficiency controls:

```bash
export INSURERAG_MAX_ANSWER_PAGES=3
export INSURERAG_MAX_PAGE_CHARS=900
export INSURERAG_MAX_CONTEXT_CHARS=2400
export OLLAMA_NUM_PREDICT=384
```

The pipeline sends only question-relevant evidence snippets to the LLM rather than full retrieved pages. Exact policy answers and glossary hits are handled deterministically when possible, so small local LLMs are reserved for open-ended explanation.

Extract PDF text/layout:

```bash
.venv/bin/python main.py extract-pdf policy.pdf
.venv/bin/python main.py extract-pdf policy.pdf --layout
.venv/bin/python main.py extract-pdf policy.pdf --render --output-dir rendered_pages
```

Compare two policy versions:

```bash
.venv/bin/python main.py diff policy_v1.pdf policy_v2.pdf
```

## Evaluation Plan

Core metrics:
- Retrieval: Recall@5, MRR@10, nDCG@10.
- Answering: EM/F1 and ANLS for short answers.
- Evidence: citation precision and evidence recall.
- Abstention: unsupported-question accuracy and selective risk curves.
- Efficiency: p50/p95 latency, index size, and per-query cost.

Key ablations:
- OCR-text RAG vs page-image VLM-RAG.
- Without vs with hard negatives.
- Without vs with citation constraints.
- Qwen2.5-VL vs PaliGemma 2 / Florence-2 baselines.
- Without vs with abstention/calibration.


## Resume Blurb

PolicyLens-VLM / InsureRAG-VLM: Built a citation-grounded multimodal RAG system for insurance policy and endorsement review using page-image retrieval and VLM generation, with selective abstention and clause-diff analysis.

Implemented result-oriented bullet draft:

- Built a reproducible insurance document intelligence pipeline that renders PDFs into page-image manifests, constructs QA/evidence pairs from policy documents plus CUAD/ACORD, and evaluates retrieval with Recall@k, MRR, and nDCG.
- Added a no-key local baseline with structured cited answers and abstention, plus an ablation harness for comparing OCR/text retrieval against page-image retrieval backends.
- Implemented rule-based policy drift summaries for deductible, coverage limit, endorsement, exclusion, and duties-after-loss changes.
