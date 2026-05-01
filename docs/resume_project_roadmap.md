# InsureRAG-VLM Resume Project Roadmap

This document tracks what is already implemented, what should be improved next,
and which interview questions the project should be ready to answer.

## Current Verified MVP

The current project is a reproducible local MVP for insurance document
intelligence:

- PDF-to-page-image preprocessing with document and page metadata.
- `page_manifest.jsonl` compatible with future ColPali/ColQwen2-style backends.
- Auxiliary text/OCR metadata for snippets, weak labels, and OCR baselines.
- Synthetic insurance policy packet v1/v2 for smoke tests.
- Policy QA/evidence generation and hard-negative construction.
- Local no-key hashing retriever.
- `visual_stub` page-image retrieval interface.
- `local_image` no-key image-aware retrieval baseline that reads rendered page images and keeps the same ranked-page schema as future ColPali/ColQwen2 backends.
- Optional Hugging Face Transformers GPU backends: `colqwen2_hf` / `colqwen2_local` and `colpali_hf` / `colpali_local`.
- Public insurance PDF importer for real document smoke tests.
- Real public-PDF QA/evidence generator with `--target-count`; verified at 250 answerable examples plus unsupported questions.
- Structured grounded answer schema with citations, confidence, and abstention.
- Calibration runner for confidence thresholds, selective risk, coverage, and unsupported abstention accuracy.
- Knowledge-base answers for common insurance terms and acronyms.
- Ollama integration with `qwen2.5:3b` for open-ended local LLM answers.
- Token-efficient evidence selection before LLM generation.
- Policy diff summaries for deductible, coverage, endorsement, exclusions, and duties.
- Ablation runner that writes retrieval metrics, answer metrics, error cases, and summary.
- Animated browser demo with cited answers, policy diff, confidence, abstention, upload flow, and retrieval trace.

Latest local smoke-test metrics on bundled synthetic policy QA:

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text | 0.3125 | 0.8125 | 0.5521 | 0.6198 |
| visual_stub | 0.3125 | 0.6250 | 0.4479 | 0.4933 |
| local_image | 0.3125 | 0.6250 | 0.4479 | 0.4933 |

These are reproducibility metrics, not final benchmark claims.

## Next Milestones

### M5: Real Visual Retrieval Backend

Goal: move from the lightweight local image-aware baseline to measured GPU visual retrieval results.

Implementation targets:

- Add `colpali_local` or `colqwen2_local` backend behind the existing visual API. Status: interface implemented through Hugging Face Transformers retrieval classes.
- Store page embeddings under `data/03_index/colqwen2/page_embeddings/`.
- Keep the output schema identical to `visual_stub` and `local_image`.
- Compare `local_text` vs `visual_stub` vs `local_image` vs real visual retrieval.
- Run the GPU command on A100/H100 or another CUDA machine and commit only compact metric reports, not raw model embeddings.

Why it matters for MLE resumes:

- Shows model integration beyond prompt engineering.
- Demonstrates retrieval-system design and evaluation.
- Connects directly to multimodal enterprise search roles.

### M6: Better Evaluation Set

Goal: move beyond synthetic-only metrics.

Implementation targets:

- Expand the new `public_docs` importer into 20-50 public policy PDFs or public sample forms.
- Keep document-level train/valid/test split.
- Create 200-500 QA pairs with evidence page IDs. Status: `generate-qa --target-count 250` is implemented and verified on the current public PDF set.
- Add unsupported questions and version-comparison questions.
- Add manual error buckets in `reports/error_analysis/`.

Resume value:

- Stronger evidence that the system was evaluated seriously.
- Lets you discuss data leakage, QA construction, and hard negatives.

### M7: Calibration and Abstention

Goal: make refusal behavior measurable, not just heuristic.

Implementation targets:

- Extend the current `run-calibration` report with validation-set threshold fitting.
- Save retrieval score, evidence overlap, answer-source agreement, and LLM confidence proxy.
- Fit a threshold on validation data.
- Report selective risk, coverage, and unsupported abstention accuracy.
- Add examples where the model correctly refuses to answer.

Resume value:

- This is especially relevant for insurance, legal, healthcare, and finance AI.
- Shows statistical background in a way that maps to modern GenAI safety.

### M8: Demo Polish

Goal: make the demo interview-friendly.

Implementation targets:

- Show uploaded PDF page thumbnails.
- Highlight cited evidence on the selected page.
- Continue polishing the implemented retrieval trace: query -> top pages -> evidence snippet -> answer.
- Add a toggle for `deterministic`, `qwen2.5:3b`, and future `ColPali`.
- Add canned questions for deductible, limits, exclusions, endorsements, and unsupported queries.

Resume value:

- Recruiters and interviewers can understand the system in 60 seconds.
- The UI makes citations, abstention, and policy diff visible.

### M9: Packaging and Reproducibility

Goal: make the repo look like a real research engineering artifact.

Implementation targets:

- Add `make smoke-test` or `scripts/smoke_test.sh`.
- Add Dockerfile or explicit Python environment setup.
- Add model/data card with licenses and limitations.
- Add GitHub Actions for compile and no-key smoke tests.
- Commit generated reports as small sample artifacts, not raw external data.

Resume value:

- Signals production habits: reproducibility, CI, documentation, and operational clarity.

## Common Interview Questions and How This Project Should Answer Them

### Why not OCR-only RAG?

Expected answer:

Insurance packets contain tables, declarations pages, endorsements, section hierarchy,
and visually meaningful layout. OCR-only chunking can separate values from labels or
lose page-level context. This project treats a page image as the primary retrieval
unit and keeps OCR/text as auxiliary evidence and baseline data.

Project evidence:

- `preprocess-pages` renders PDFs into page images.
- `visual_stub` fixes the page-image retrieval interface.
- Ablation compares local text retrieval against page-image retrieval interface.

### How do you prevent hallucinations?

Expected answer:

The system emits structured answers with citations, computes confidence from retrieval
score and evidence overlap, and abstains when evidence is insufficient. For exact
policy facts, deterministic extraction is preferred over free-form generation.

Project evidence:

- `query --json` returns `answer`, `citations`, `confidence`, `abstain`.
- Unsupported question smoke test returns `abstain=true`.
- Qwen is reserved for open-ended explanation where deterministic evidence is unavailable.

### How do you know retrieval is good?

Expected answer:

Retrieval is evaluated with Recall@1/5, MRR@10, and nDCG@10 at the document/page level.
Hard negatives are constructed from similar policy clauses and versions.

Project evidence:

- `retrieval-metrics`
- `visual-retrieval-metrics`
- `run-ablation`
- `hard_negatives.jsonl`

### How do you avoid data leakage?

Expected answer:

Splits should be document-level, not paragraph-level or page-level, because policy pages
inside the same packet are highly correlated. The preprocessing pipeline writes split
metadata so evaluation can be reproduced.

Project evidence:

- `splits.parquet`
- document registry with PDF hashes
- explicit `doc_id` and `page_id`

### Why use Qwen2.5 3B locally?

Expected answer:

It is small enough for a laptop demo and better suited than reasoning-first models for
instruction following and structured answers. The system also controls token usage by
sending only evidence snippets, not full pages.

Project evidence:

- Ollama backend detects `qwen2.5:3b`.
- `INSURERAG_MAX_ANSWER_PAGES`, `INSURERAG_MAX_PAGE_CHARS`, and
  `INSURERAG_MAX_CONTEXT_CHARS` control prompt size.

### What is the MLE part of this project?

Expected answer:

The project includes data preprocessing, dataset construction, retrieval model interface
design, evaluation harnesses, ablation, calibration/abstention, and deployment-oriented
demo work. Future work adds real ColPali/ColQwen2 embeddings and PEFT fine-tuning.

Project evidence:

- reproducible CLI pipeline
- metrics reports
- ablation outputs
- model/backend abstraction
- planned real visual backend

### What are the current limitations?

Expected answer:

The current public demo uses a synthetic dataset and a `visual_stub`; real ColPali or
ColQwen2 embeddings still need to be integrated. The policy diff is rule-based and
should be evaluated on more public policy versions. The browser demo should add page
thumbnail highlighting and multi-file upload sessions.

Project evidence:

- README states current reproducible results separately from planned claims.
- This roadmap separates implemented features from future milestones.

## Resume Bullet Template

Use only measured results once they are final:

- Built InsureRAG-VLM, a citation-grounded multimodal RAG pipeline for insurance
  policy review, with PDF page-image preprocessing, QA/evidence generation, structured
  citations, abstention, and policy-version diff.
- Implemented retrieval/evaluation harness with Recall@k, MRR, nDCG, citation precision,
  evidence recall, hard negatives, and ablation reports comparing OCR/text and page-image
  retrieval paths.
- Added token-efficient local LLM integration with Qwen2.5-3B and deterministic evidence
  extraction to reduce hallucination risk on high-stakes policy questions.

## Immediate Action Plan

1. Commit the current reproducible MVP.
2. Add 20-50 real public insurance PDFs.
3. Build a 200-500 question evaluation set with document-level splits.
4. Integrate real ColPali/ColQwen2 local backend.
5. Improve the demo with page thumbnails and evidence highlighting.
6. Add calibration curves and unsupported-question metrics.
7. Write final README results and resume bullets only after metrics are rerun.
