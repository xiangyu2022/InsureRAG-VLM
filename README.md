# InsureRAG-VLM

Citation-grounded multimodal RAG for insurance policy and endorsement review. The project is designed around page-image retrieval first: PDFs are rendered into full-page visual units for ColQwen2/ColPali-style retrieval, while OCR and text layers are kept as auxiliary evidence for snippets, weak labels, baselines, and error analysis.

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
- Weak query-page training pairs from section hints.
- Document-level train/valid/test splits.
- Existing text-RAG scaffold, PDF extraction, evaluation helpers, and policy diff utilities.

Still planned:
- Real ColQwen2/ColPali embedding generation and FAISS/Vespa index build.
- Layout/table extraction as explanation and evaluation metadata.
- CUAD/ACORD ingestion and hard-negative construction.
- Evidence recall, nDCG/MRR, selective abstention metrics, and ablation harness.
- Web demo with cited page previews.

## Quickstart

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

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
