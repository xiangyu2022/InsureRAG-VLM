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

System stages:

```text
PDF / policy packet
  -> document registry, SHA-256 hashes, document split metadata
  -> page rendering into normalized image units
  -> page manifest + auxiliary text/OCR metadata
  -> text index, local image-layout index, or ColQwen2/ColPali GPU index
  -> retrieval + insurance-domain reranking
  -> extractive/LLM answer generation
  -> citation validation + configurable abstention threshold
  -> retrieval, answer, calibration, and error-group reports
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
- Existing text-RAG scaffold, PDF extraction, evaluation helpers, and policy diff utilities.
- Calibration report for selective answering and unsupported-question abstention.
- Reproducible benchmark entrypoint with experiment manifest, dataset hash, environment metadata, retrieval metrics, answer metrics, calibration outputs, and grouped reliability errors.
- Animated browser demo for cited answers, cited-page thumbnails, highlighted evidence snippets, retrieval trace, abstention, upload flow, and policy-version diff.
- Optional Hugging Face Transformers GPU backends for `colqwen2_hf` / `colqwen2_local` and `colpali_hf` / `colpali_local`.

Still planned:
- Cloud GPU run with committed ColQwen2/ColPali benchmark numbers beyond the current `local_image` baseline.
- Layout/table extraction as explanation and evaluation metadata.
- Multi-file upload sessions. The current browser demo can upload and index one local PDF for Q&A; policy diff uses the first two PDFs available in the active real-document folder.

## Current Reproducible Results

### Research-Proof Local Baseline

Current curated-data validation passes on **169 RAG pages**, **1,090 snippets**, **1,259 RAG corpus
records**, and **3,600 SFT records** with **400 unsupported examples**. The validator writes
`reports/research_proof/dataset_validation.json` and regenerates `data/04_curated/dataset_summary.json`.

Latest local CPU benchmark smoke: **20 public insurance PDFs**, **168 rendered pages**, and
**70 QA rows** with **20 unsupported examples**, written to `reports/research_proof_local/`.

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text (text + hashing) | 0.2200 | 0.4800 | 0.3347 | 0.4109 |
| local_image (image-aware local baseline) | 0.2000 | 0.5200 | 0.3429 | 0.4266 |
| colqwen2_local / colpali_local (GPU) | pending cloud GPU run | pending cloud GPU run | pending cloud GPU run | pending cloud GPU run |

Answering and calibration reports from the same run:

| Metric | Value |
| --- | ---: |
| Extractive answer F1 | 0.0801 |
| Citation precision | 0.0800 |
| Evidence recall | 0.0800 |
| Unsupported abstention accuracy | 0.9000 |
| Coverage | 0.3143 |

These are CPU-local smoke numbers, not final model-quality claims. The next research-proof result
should come from `main.py run-gpu-benchmark --backend colqwen2_local` on a cloud GPU and should be
written to `reports/research_proof/`.

## Quickstart

```bash
pip install -r requirements.txt

# Download real public PDFs, then run the no-key smoke test.
python main.py import-data --output-root data --datasets public_docs
make smoke-test

# Or manually:
python main.py build-index data/00_raw/external/public_docs --index-dir data
python main.py query data/00_raw/external/public_docs "What coverage limits are described?" --index-dir data --top-k 3
```

Run the animated browser demo:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 150
.venv/bin/python main.py generate-qa data/00_raw/external/public_docs --output-dir data/02_processed --target-count 300
.venv/bin/python main.py build-index data/00_raw/external/public_docs --index-dir data
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

`public_docs` downloads a small set of public insurance PDFs into `data/00_raw/external/public_docs/`. These files are for local reproducibility and are not intended to be committed to GitHub.

Import a broader real-domain mix:

```bash
.venv/bin/python main.py import-data --output-root data --datasets real_domain_mix
.venv/bin/python main.py build-index data/00_raw/external/real_domain_mix --index-dir data
.venv/bin/python main.py generate-qa data/00_raw/external/real_domain_mix --output-dir data/02_processed --target-count 300
```

`real_domain_mix` maps to the requested data categories:

- Domain-specific text datasets: CUAD and ACORD source files.
- Domain web content: insurance regulator and consumer-information pages.
- Domain news articles: current insurance news/issue pages.
- Industry reports and white papers: NAIC Journal of Insurance Regulation article pages.
- Social media data: manifest-only pointers for API/terms-compliant collection.
- Domain conversation data: insurance FAQ pages and InsuranceQA source pointers.

Downloaded HTML pages are converted to `.txt` for the text retriever. Social/forum and restricted-license sources are registered in manifests instead of scraped by default.

Render PDFs into a page-image dataset:

```bash
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 200
```

Optional OCR auxiliary metadata:

```bash
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 200 --run-ocr
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
.venv/bin/python main.py build-index data/00_raw/external/public_docs --render-pdf-pages --pdf-render-dir data/01_interim/legacy_pdf_pages
.venv/bin/python main.py query data/00_raw/external/public_docs "What coverage does the document describe?"
```

Import external CUAD/ACORD manifests and local files:

```bash
.venv/bin/python main.py import-data --output-root data --datasets cuad acord insuranceqa public_docs
```

Generate QA/evidence pairs and hard negatives:

```bash
.venv/bin/python main.py generate-qa data/00_raw/external/public_docs \
  --output-dir data/02_processed \
  --cuad-master data/00_raw/external/cuad/master_clauses.csv \
  --acord-root data/00_raw/external/acord/extracted \
  --max-cuad 500 \
  --max-acord 500
```

Compute retrieval metrics over generated policy QA:

```bash
.venv/bin/python main.py retrieval-metrics data/00_raw/external/public_docs data/02_processed/qa_pairs.jsonl --index-dir data --top-k 3
```

Build and evaluate the page-image retrieval interface:

```bash
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend visual_stub
.venv/bin/python main.py visual-retrieval-metrics data/02_processed/qa_pairs.jsonl --index-dir data/03_index/colqwen2 --backend visual_stub --top-k 3
.venv/bin/python main.py build-visual-index data/03_index/colqwen2/page_manifest.jsonl --index-dir data/03_index/colqwen2 --backend local_image
.venv/bin/python main.py visual-retrieval-metrics data/02_processed/qa_pairs.jsonl --index-dir data/03_index/colqwen2 --backend local_image --top-k 3
```

Run the reproducible benchmark harness:

```bash
.venv/bin/python main.py validate-curated-data \
  --dataset-dir data/04_curated \
  --output-dir reports/research_proof

.venv/bin/python main.py run-gpu-benchmark \
  --data-folder data/00_raw/external/public_docs \
  --output-dir reports/research_proof_local \
  --backend local_image \
  --target-count 50 \
  --unsupported-count 20 \
  --allow-backend-failures

.venv/bin/python main.py run-gpu-benchmark \
  --data-folder data/00_raw/external/public_docs \
  --output-dir reports/research_proof \
  --backend colqwen2_local \
  --target-count 300 \
  --unsupported-count 50 \
  --top-k 10
```

This command renders pages, generates answerable and unsupported QA, builds the text baseline,
builds `local_image`, builds the requested GPU visual backend, and writes:

```text
reports/research_proof/summary.md
reports/research_proof/experiment_manifest.json
reports/research_proof/retrieval_metrics.csv
reports/research_proof/answer_metrics.json
reports/research_proof/error_cases_by_type.jsonl
reports/research_proof/calibration/
```

The manifest records the git commit, dataset hash/counts, CUDA/PyTorch environment, GPU name,
backend, model environment variables, dtype, batch size, indexing time, latency, and peak CUDA memory.
For a CPU-only dry run, use `--backend local_image --allow-backend-failures`.

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

Fine-tune Qwen 7B with LoRA/QLoRA on the curated SFT dataset:

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt

# Validates CUDA, tokenizer formatting, and the first curated SFT record.
.venv/bin/python main.py sft-lora-smoke-test \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct

# Dependency-free local check for the curated SFT prompt/label formatting.
.venv/bin/python main.py sft-lora-smoke-test \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --format-only

# Tiny end-to-end GPU demo: loads Qwen 7B in 4-bit, trains one optimizer step,
# and writes a LoRA adapter to models/qwen7b-insurerag-lora-smoke.
.venv/bin/python main.py sft-lora-qwen \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/qwen7b-insurerag-lora-smoke \
  --max-samples 2 \
  --max-steps 1 \
  --logging-steps 1 \
  --save-steps 1

# Full curated-set LoRA run.
.venv/bin/python main.py sft-lora-qwen \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --output-dir models/qwen7b-insurerag-lora

# More frequent checkpoints for cluster jobs.
.venv/bin/python main.py sft-lora-qwen \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --output-dir models/qwen7b-insurerag-lora \
  --save-steps 50 \
  --save-total-limit 4

# Resume from the latest checkpoint under the output directory.
.venv/bin/python main.py sft-lora-qwen \
  --dataset-path data/04_curated/sft_dataset.jsonl \
  --output-dir models/qwen7b-insurerag-lora \
  --auto-resume
```

The SFT command requires `torch.cuda.is_available()` to be true. By default it uses 4-bit QLoRA with LoRA adapters on Qwen attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
During training, Hugging Face checkpoints are saved every `--save-steps` steps, `sft_progress.json` is updated in the output directory, and `SIGTERM` / `SIGINT` will request a final checkpoint before stopping so cluster jobs can resume from the latest saved state.

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

Unsupported abstention coverage defaults to 50 generated unsupported questions. Use
`--unsupported-count` on `generate-qa` or `run-gpu-benchmark` to change that validation set size.

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
.venv/bin/python main.py query data/00_raw/external/public_docs "What coverage limits are described?" --index-dir data --top-k 3 --json
```

Summarize policy version drift:

```bash
.venv/bin/python main.py policy-diff path/to/original_policy.pdf path/to/revised_policy.pdf --output reports/diff/diff_summary.json
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

## System Design Notes

- Scalability: indexing is separated from query serving; page rendering, visual embedding, and scoring can be batched offline and cached by dataset hash.
- GPU indexing: cloud GPU runs should use small visual batches, record dtype/device settings, and commit only compact reports rather than generated embeddings or page images.
- Reliability: answer serving validates cited evidence before returning a response and uses `INSURERAG_ABSTAIN_THRESHOLD` plus calibration curves to tune selective answering.
- Privacy: internal policy PDFs belong under ignored local data folders; reports should contain manifests, hashes, metrics, and redacted snippets rather than raw private documents.
- Failure analysis: benchmark reports group errors into retrieval misses, citation mismatches, unsupported false positives, and weak answer extraction cases.

## Resume Bullets

- Built a multimodal RAG system for insurance policy PDFs with page-image retrieval, citation-grounded answering, and abstention over public regulatory documents.
- Implemented a reproducible benchmark harness comparing text, local image-layout, and ColQwen2/ColPali GPU visual retrieval with dataset hashes, CUDA environment manifests, latency, indexing time, and calibration reports.
- Designed an evaluation pipeline measuring Recall@K, MRR, nDCG, citation precision, evidence recall, unsupported abstention accuracy, coverage, selective risk, and grouped error cases.
