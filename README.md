# InsureRAG-VLM

Insurance RAG breaks when the system treats a policy packet like ordinary text.
This project is built around that premise.

**InsureRAG-VLM** is a citation-grounded, hybrid multimodal RAG system for insurance policy review.
It is designed for the questions that actually matter in policy packets:

- What is the deductible, limit, or premium?
- Does an endorsement override the base exclusion?
- Is the answer on the declarations page, the schedule, or the main form?
- Is the evidence strong enough to answer at all?

The current default pipeline is not image-only and not text-only. It uses **hybrid text retrieval as
the backbone**, adds **lightweight page-image layout priors**, performs **table-aware and graph-aware
retrieval**, and returns **structured answers with page citations, evidence roles, conflict notes,
and abstention when support is weak**.

Heavy ColQwen2/ColPali-style page-image retrieval remains in the repo as an **optional research
backend**, but it is no longer the main product path.

## Demo Walkthrough

![InsureRAG-VLM animated demo](assets/demo/insurerag_vlm_demo.gif)

The browser demo shows the full loop:

- Upload or select a policy PDF
- Ask deductible, limit, endorsement, exclusion, declaration, glossary, or policy-diff questions
- Inspect ranked pages, evidence snippets, and retrieval traces
- Get a structured answer with citations, confidence, conflict notes, and caveats
- Abstain when the evidence does not support a reliable answer

Run it locally:

```bash
.venv/bin/python main.py demo-web --port 7860
```

Then open `http://127.0.0.1:7860`.

## Why This Project Is Different

Most RAG systems stop at "retrieve top-k text chunks and ask an LLM."
Insurance review needs more:

- **Declarations-aware retrieval** because limits often live outside the main form text
- **Endorsement-aware reasoning** because base policy language may be modified later
- **Table-aware evidence** because deductible, premium, and schedule questions are numeric and brittle
- **Structure-aware ranking** because section titles, clause types, and form codes matter
- **Selective answering** because unsupported answers are worse than abstaining

This repo is optimized around that workflow rather than generic semantic search.

## What The System Returns

Input:
- a policy packet, endorsement packet, claim or billing PDF, or curated insurance dataset
- a user question

Output:
- a grounded answer
- page-level citations
- supporting snippets
- structured fields such as `coverage`, `limit`, and `evidence_role` when available
- conflict / override notes for declaration-vs-policy and endorsement-vs-exclusion cases
- abstention when support is insufficient

## Default Architecture

```text
Insurance PDFs / images
  -> document registry, hashes, and split metadata
  -> snippet / page / table corpus construction
  -> document typing, clause typing, coverage tags, section parsing, form-code hints
  -> dense text index + sparse text index + table sparse index + lightweight page-image auxiliary index + document graph
  -> query understanding
  -> dense retrieval + sparse retrieval + metadata-targeted retrieval + graph expansion
  -> reciprocal-rank fusion + insurance-aware reranking
  -> snippet-to-page rollup + insurance-logic context packing
  -> extractive / LLM answering
  -> citation validation + abstention + structured output
```

The default online path is intentionally **low latency**:

- lightweight query understanding instead of an extra LLM call
- lightweight table normalization instead of full table reconstruction
- lightweight page-image features instead of mandatory VLM inference
- rule and metadata signals before any expensive multimodal escalation

## Core Capabilities

Implemented today:

- Hybrid multimodal default retrieval with dense text, sparse text, reciprocal-rank fusion, and lightweight page-image auxiliary scoring
- Snippet-to-page rollup with insurance-logic long-context packing
- Rule-based query understanding for limits, deductibles, declarations, definitions, endorsements, exclusions, and policy-diff questions
- Lightweight section parsing with `section_titles`, `section_path`, `section_anchor`, and endorsement form-code hints
- Table extraction and normalization for limit, deductible, premium, and schedule-style evidence
- Insurance document structure heuristics for `document_type`, `clause_type`, `coverage_tags`, and graph expansion across declarations, endorsements, exclusions, and definitions
- Metadata-targeted retrieval prioritization and lightweight override/conflict summaries
- Structured answer JSON with citations, caveats, `coverage`, `limit`, `evidence_role`, `conflicts`, and `override_summary`
- Policy-version diff utilities
- Reproducible evaluation, calibration, and error-group reporting
- QLoRA SFT for Qwen 7B plus dense retriever training and retrieval-conditioned SFT data generation

Still intentionally lightweight or still in progress:

- stronger clause-level override resolution
- stronger table/layout extraction
- post-migration committed benchmark numbers for the new default stack
- gated VLM escalation for visually difficult scanned pages
- learned rerankers beyond the current low-latency rule stack

## Repository Map

Main code lives in:

- `src/insurerag_vlm/hybrid_pipeline.py`: default retrieval and answering pipeline
- `src/insurerag_vlm/query_understanding.py`: low-latency insurance query schema inference
- `src/insurerag_vlm/tables.py`: table normalization and field extraction
- `src/insurerag_vlm/graph.py`: lightweight document graph construction and expansion
- `src/insurerag_vlm/insurance_structure.py`: document typing, clause typing, section parsing, coverage tagging
- `src/insurerag_vlm/training_data.py`: retrieval triples and retrieval-conditioned SFT corpus builder
- `src/insurerag_vlm/dense_training.py`: trainable dense retriever

## Data

The repo supports two main modes:

- **Curated mode** via `data/04_curated/` for development, SFT, and structured training
- **Raw document mode** via imported or local PDFs for end-to-end document experiments

Recommended public-data mix:

- CUAD for clause extraction and QA supervision
- ACORD for clause retrieval and hard negatives
- InsuranceQA for insurance question language style
- FUNSD/CORD as structured-document proxies
- public policy PDFs for real page-level retrieval demos

Private policy documents should stay outside Git. Keep raw internal files in ignored local folders
and commit only manifests, hashes, scripts, and redacted reports.

## Current Status

The system has already moved beyond the original page-image-first prototype.

What changed in the current generation:

- the default stack is now `hybrid_multimodal`
- the answer path is centered on `text evidence -> page citation`, not pure image retrieval
- table and graph signals are first-class retrieval inputs
- the project now includes a full training path:
  - build retrieval triples
  - train a dense retriever
  - rebuild retrieval-conditioned corpora
  - continue QLoRA SFT from an existing adapter

For migration details, see `reports/hybrid_multimodal/summary.md`.

## Current Reproducible State

Curated-data validation currently passes on:

- **169** RAG pages
- **1,090** snippets
- **1,259** RAG corpus records
- **3,600** SFT records
- **400** unsupported examples

The validator writes `reports/research_proof/dataset_validation.json` and regenerates
`data/04_curated/dataset_summary.json`.

The latest committed local benchmark numbers in the repo are still **legacy baselines** from before
the hybrid-multimodal migration. They are useful as a historical floor, not as the final claim for
the current default system.

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text (text + hashing) | 0.2200 | 0.4800 | 0.3347 | 0.4109 |
| local_image (image-aware local baseline) | 0.2000 | 0.5200 | 0.3429 | 0.4266 |
| colqwen2_local / colpali_local (GPU) | pending | pending | pending | pending |

Legacy smoke-run answer metrics:

| Metric | Value |
| --- | ---: |
| Extractive answer F1 | 0.0801 |
| Citation precision | 0.0800 |
| Evidence recall | 0.0800 |
| Unsupported abstention accuracy | 0.9000 |
| Coverage | 0.3143 |

The next benchmark that matters should be a **post-migration** run for `hybrid_multimodal` and
`hybrid_text`, written to `reports/research_proof/`.

## Quickstart

Fastest path to a working local demo:

```bash
pip install -r requirements.txt

# 1. Download a small public insurance PDF set.
python main.py import-data --output-root data --datasets public_docs

# 2. Optional smoke test.
make smoke-test

# 3. Query with the default hybrid multimodal pipeline.
python main.py build-index data/00_raw/external/public_docs --index-dir data --retrieval-mode hybrid_multimodal
python main.py query data/00_raw/external/public_docs "What coverage limits are described?" --index-dir data --top-k 3 --retrieval-mode hybrid_multimodal
```

If you want the browser demo instead of CLI querying, jump straight to `demo-web` below.

Run the animated browser demo:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 150
.venv/bin/python main.py generate-qa data/00_raw/external/public_docs --output-dir data/02_processed --target-count 300
.venv/bin/python main.py build-index data/00_raw/external/public_docs --index-dir data --retrieval-mode hybrid_multimodal
.venv/bin/python main.py demo-web --port 7860
```

Then open `http://127.0.0.1:7860`. The UI lets you upload a PDF, asks questions against the active indexed file, and shows grounded citations, confidence/abstention, policy diff, and an expandable retrieval trace with top ranked pages.

Import a small real public PDF set for local experiments:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py preprocess-pages data/00_raw/external/public_docs --output-root data --render-dpi 150
.venv/bin/python main.py build-index data/00_raw/external/public_docs --index-dir data --retrieval-mode hybrid_multimodal
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

data/
  hybrid_snippets_dense.npy
  hybrid_snippets_sparse.json
  hybrid_snippets.jsonl
  hybrid_pages_dense.npy
  hybrid_pages_sparse.json
  hybrid_pages.jsonl
  hybrid_tables_sparse.json
  hybrid_tables.jsonl
  hybrid_graph.jsonl
  hybrid_page_image.npy
  hybrid_page_image_pages.jsonl

data/manifests/
  preprocess_run_manifest.jsonl
```

Run the hybrid text-only path without the image auxiliary signal:

```bash
.venv/bin/python main.py build-index data/00_raw/external/public_docs --retrieval-mode hybrid_text --disable-image-signal
.venv/bin/python main.py query data/00_raw/external/public_docs "What coverage does the document describe?" --retrieval-mode hybrid_text --disable-image-signal
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
.venv/bin/python main.py retrieval-metrics data/00_raw/external/public_docs data/02_processed/qa_pairs.jsonl --index-dir data --top-k 3 --retrieval-mode hybrid_multimodal
```

Build and evaluate the optional page-image retrieval interface:

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

The default application path does not depend on these visual-only commands. They are intended for
side-by-side comparison against the hybrid multimodal default rather than for day-to-day querying.

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

Build staged training corpora and a trainable dense retriever:

```bash
# Build doc-disjoint retrieval triples, retrieval-conditioned SFT, and calibration manifests.
.venv/bin/python main.py build-training-corpora \
  --data-folder data/04_curated \
  --output-dir reports/training_data \
  --index-dir reports/training_data/index \
  --retrieval-model local-hashing \
  --retrieval-mode hybrid_multimodal \
  --corpus-source curated \
  --disable-image-signal

# Train a local dense retriever from retrieval_train.jsonl.
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python main.py train-dense-retriever \
  --dataset-path reports/training_data/retrieval_train.jsonl \
  --output-dir models/retrieval/bge-base-insurerag

# Rebuild the hybrid index with the trained dense model and evaluate retrieval.
.venv/bin/python main.py build-index data/04_curated \
  --index-dir reports/training_data/dense_index \
  --retrieval-model models/retrieval/bge-base-insurerag \
  --retrieval-mode hybrid_multimodal \
  --corpus-source curated \
  --disable-image-signal
.venv/bin/python main.py retrieval-metrics data/04_curated \
  reports/training_data/calibration_test.jsonl \
  --index-dir reports/training_data/dense_index \
  --retrieval-model models/retrieval/bge-base-insurerag \
  --retrieval-mode hybrid_multimodal \
  --corpus-source curated \
  --disable-image-signal

# Continue QLoRA from the current adapter using retrieval-conditioned evidence.
.venv/bin/python main.py sft-lora-qwen \
  --dataset-path reports/training_data/rag_sft_train.jsonl \
  --output-dir models/qwen7b-insurerag-lora-rag \
  --adapter-path models/qwen7b-insurerag-lora \
  --auto-resume
```

`build-training-corpora` writes:
- `retrieval_train.jsonl`, `retrieval_dev.jsonl`, `retrieval_test.jsonl`
- `rag_sft_train.jsonl`, `rag_sft_dev.jsonl`, `rag_sft_test.jsonl`
- `calibration_dev.jsonl`, `calibration_test.jsonl`
- `training_corpora_summary.json`

Generate a 200-500 example real-PDF QA/evidence set:

```bash
.venv/bin/python main.py import-data --output-root data --datasets public_docs
.venv/bin/python main.py generate-qa data/00_raw/external/public_docs \
  --output-dir reports/public_docs_qa_v3 \
  --target-count 300
.venv/bin/python main.py build-index data/00_raw/external/public_docs \
  --index-dir reports/public_docs_qa_v3/index \
  --retrieval-mode hybrid_multimodal
.venv/bin/python main.py retrieval-metrics data/00_raw/external/public_docs \
  reports/public_docs_qa_v3/qa_pairs.jsonl \
  --index-dir reports/public_docs_qa_v3/index \
  --top-k 5 \
  --retrieval-mode hybrid_multimodal
```

The real-PDF QA generator creates evidence-anchored questions, supports multi-positive page labels for broad topics, and writes `qa_splits.jsonl` plus per-example `split` / `split_doc_id` fields to keep evaluation document-aware. The post-migration `hybrid_multimodal` run should be reported first; GPU ColQwen2/ColPali numbers should then be reported separately as optional visual-backend comparisons.

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

The structured query response now includes:
- `query_understanding` flags such as `needs_limit`, `needs_endorsement_check`, and `needs_table_lookup`.
- citation metadata including `document_type` and `primary_clause_type`.
- normalized structured fields such as `coverage`, `limit`, and `evidence_role` when they can be extracted from the cited evidence.
- `conflict_notes`, `conflicts`, and `override_summary` for low-latency declarations/endorsement/exclusion review signals.
- `caveats` when the retrieved evidence suggests a declarations-page or endorsement override review is still warranted.

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
- Hybrid text-only vs hybrid multimodal.
- Hybrid multimodal vs optional page-image-only visual backends.
- Without vs with table retrieval and graph expansion.
- Without vs with hard negatives.
- Without vs with citation constraints.
- Qwen2.5-VL vs PaliGemma 2 / Florence-2 baselines.
- Without vs with abstention/calibration.

