# InsureRAG-VLM Resume Project Roadmap

This document tracks what is already implemented, what was verified locally, and
what remains before the project should be presented as a polished MLE portfolio
project.

## Current Verified MVP

The current project is a reproducible local MVP for insurance document
intelligence:

- PDF-to-page-image preprocessing with document and page metadata.
- `page_manifest.jsonl` compatible with ColPali/ColQwen2-style backends.
- Auxiliary text/OCR metadata for snippets, weak labels, and OCR baselines.
- Synthetic insurance policy packet v1/v2 for smoke tests.
- Public insurance PDF importer with 20 working state insurance department PDFs.
- Real public-PDF QA/evidence generator with `--target-count`; verified at 300
  answerable examples plus 3 unsupported questions.
- QA-level document split labels via `split` and `split_doc_id`, plus
  `qa_splits.jsonl`.
- Policy QA/evidence generation and hard-negative construction.
- Local no-key hashing retriever.
- `visual_stub` page-image retrieval interface.
- `local_image` no-key image-aware retrieval baseline that reads rendered page
  images and keeps the same ranked-page schema as future ColPali/ColQwen2
  backends.
- Optional Hugging Face Transformers GPU backends: `colqwen2_hf` /
  `colqwen2_local` and `colpali_hf` / `colpali_local`.
- Structured grounded answer schema with citations, confidence, and abstention.
- Calibration runner for confidence thresholds, selective risk, coverage, and
  unsupported abstention accuracy.
- Deterministic knowledge-base answers for 250+ insurance terms, acronyms,
  metrics, roles, and common aliases.
- Ollama integration with `qwen2.5:3b` for open-ended local LLM answers.
- Token-efficient evidence selection before LLM generation.
- Policy diff summaries for deductible, coverage, endorsement, exclusions, and
  duties.
- Ablation runner that writes retrieval metrics, answer metrics, error cases,
  and summary.
- Animated browser demo with cited answers, cited-page thumbnails, highlighted
  evidence snippets, policy diff, confidence, abstention, upload flow, and
  retrieval trace.
- `make smoke-test` / `scripts/smoke_test.sh` for no-key reproducibility.

## Latest Verified Metrics

### Synthetic Smoke Test

Latest local smoke-test metrics on the bundled synthetic policy QA:

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text | 0.3125 | 1.0000 | 0.5958 | 0.6978 |
| visual_stub | 0.3125 | 0.6250 | 0.4479 | 0.4933 |
| local_image | 0.3125 | 0.6250 | 0.4479 | 0.4933 |

These are reproducibility metrics, not final benchmark claims.

### Real Public-PDF Evaluation

Latest local real-PDF run:

- Documents: 20 public insurance PDFs.
- Rendered pages: 168.
- QA rows: 303 total, 300 answerable, 3 unsupported.
- Answerable examples cover 12 documents.
- Reports: `reports/ablation_real_pdfs/` and `reports/calibration_real_pdfs/`.

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| local_text | 0.2033 | 0.4467 | 0.2974 | 0.3348 |
| visual_stub | 0.1733 | 0.4400 | 0.2733 | 0.3149 |
| local_image | 0.1733 | 0.4400 | 0.2733 | 0.3149 |

Answering and abstention from the same run:

| Metric | Value |
| --- | ---: |
| Extractive answer F1 | 0.3695 |
| Citation precision | 0.2533 |
| Evidence recall | 0.2533 |
| Unsupported abstention accuracy | 0.6667 |
| Calibration threshold | 0.40 |
| Coverage at calibration threshold | 0.6898 |

Interpretation: these are local baselines. The lightweight `local_image` backend is
mainly an interface and sanity check; the next meaningful comparison is against
`colqwen2_hf` or `colpali_hf` on a CUDA machine.

## Completed Since Initial MVP

1. Added 20 real public insurance PDF sources and verified 20/20 downloads.
2. Built a 300-answerable-example real-PDF QA/evidence set with document-aware
   split metadata.
3. Integrated ColPali/ColQwen2-compatible GPU backend interfaces behind the same
   visual retrieval schema.
4. Improved the demo with cited-page thumbnails and highlighted evidence snippets.
5. Added calibration curve outputs and unsupported-question metrics.
6. Reran local smoke tests, real-PDF ablation, and calibration reports.
7. Updated README and this roadmap to separate measured results from future work.

## Next Milestones

### M5: Measured Real Visual Retrieval

Goal: move from the lightweight local image-aware baseline to measured GPU visual
retrieval results.

Implementation targets:

- Run `colqwen2_hf` or `colpali_hf` on an A100/H100 or other CUDA machine.
- Store embeddings locally under `data/03_index/colqwen2/`, but commit only
  compact metric reports.
- Compare `local_text` vs `visual_stub` vs `local_image` vs real visual
  retrieval.
- Add the measured GPU row to README after the run.

Resume value:

- Shows model integration beyond prompt engineering.
- Demonstrates retrieval-system design and evaluation.
- Connects directly to multimodal enterprise search roles.

### M6: Stronger Evaluation Set

Goal: make the evaluation more credible than automatically generated smoke data.

Implementation targets:

- Manually audit 50-100 QA/evidence examples.
- Add version-comparison questions from paired policy PDFs.
- Add error buckets in `reports/error_analysis/`.
- Keep all reported metrics document-level and reproducible.

Resume value:

- Lets you discuss data leakage, label quality, hard negatives, and evaluation
  design.
- Prevents the project from looking like only a demo wrapper.

### M7: Calibration Improvements

Goal: make refusal behavior more measurable and more useful.

Implementation targets:

- Fit the threshold on a validation split, then report test-set selective risk.
- Save retrieval score, evidence overlap, answer-source agreement, and LLM
  confidence proxy.
- Plot coverage-risk curves from `calibration_curve.csv`.
- Add examples where the model correctly refuses to answer.

Resume value:

- This is especially relevant for insurance, legal, healthcare, and finance AI.
- Shows statistical background in a way that maps to GenAI reliability.

### M8: Demo Polish

Goal: make the demo interview-friendly in under 60 seconds.

Implementation targets:

- Add multi-file upload sessions.
- Add a backend toggle for deterministic, `qwen2.5:3b`, and future ColPali.
- Add canned questions for deductible, limits, exclusions, endorsements, and
  unsupported queries.
- Add bbox-level highlights once layout extraction is available.

Resume value:

- Recruiters and interviewers can understand the system quickly.
- The UI makes citations, abstention, and policy diff visible.

### M9: Packaging and Reproducibility

Goal: make the repo look like a real research engineering artifact.

Implementation targets:

- Add Dockerfile or explicit Python environment setup.
- Add model/data card with licenses and limitations.
- Add GitHub Actions for compile and no-key smoke tests.
- Commit compact reports, not raw external PDFs or embeddings.

Resume value:

- Signals production habits: reproducibility, CI, documentation, and operational
  clarity.

## Common Interview Questions and How This Project Should Answer Them

### Why not OCR-only RAG?

Expected answer:

Insurance packets contain tables, declarations pages, endorsements, section
hierarchy, and visually meaningful layout. OCR-only chunking can separate values
from labels or lose page-level context. This project treats a page image as the
primary retrieval unit and keeps OCR/text as auxiliary evidence and baseline data.

Project evidence:

- `preprocess-pages` renders PDFs into page images.
- `visual_stub`, `local_image`, and GPU backends share the page-image retrieval
  schema.
- Ablation compares local text retrieval against page-image retrieval paths.

### How do you prevent hallucinations?

Expected answer:

The system emits structured answers with citations, computes confidence from
retrieval score and evidence overlap, and abstains when evidence is insufficient.
For exact policy facts, deterministic extraction is preferred over free-form
generation.

Project evidence:

- `query --json` returns `answer`, `citations`, `confidence`, and `abstain`.
- Unsupported-question metrics are reported in ablation and calibration outputs.
- Qwen is reserved for open-ended explanation where deterministic evidence is
  unavailable.

### How do you know retrieval is good?

Expected answer:

Retrieval is evaluated with Recall@1/5, MRR@10, and nDCG@10 at the page level.
Hard negatives are constructed from similar policy clauses and versions.

Project evidence:

- `retrieval-metrics`
- `visual-retrieval-metrics`
- `run-ablation`
- `hard_negatives.jsonl`

### How do you avoid data leakage?

Expected answer:

Splits should be document-level, not paragraph-level or page-level, because policy
pages inside the same packet are highly correlated. The preprocessing pipeline and
QA generator write split metadata so evaluation can be reproduced.

Project evidence:

- `splits.parquet` / `splits.jsonl`
- `qa_splits.jsonl`
- document registry with PDF hashes
- explicit `doc_id`, `page_id`, `split`, and `split_doc_id`

### Why use Qwen2.5 3B locally?

Expected answer:

It is small enough for a laptop demo and better suited than reasoning-first models
for instruction following and structured answers. The system also controls token
usage by sending only evidence snippets, not full pages.

Project evidence:

- Ollama backend detects `qwen2.5:3b`.
- `INSURERAG_MAX_ANSWER_PAGES`, `INSURERAG_MAX_PAGE_CHARS`, and
  `INSURERAG_MAX_CONTEXT_CHARS` control prompt size.

### What is the MLE part of this project?

Expected answer:

The project includes data preprocessing, dataset construction, retrieval model
interface design, evaluation harnesses, ablation, calibration/abstention, latency
reporting, and deployment-oriented demo work. Future work adds measured
ColPali/ColQwen2 GPU results and PEFT fine-tuning.

Project evidence:

- reproducible CLI pipeline
- metrics reports
- ablation outputs
- calibration curves
- model/backend abstraction
- browser demo

### What are the current limitations?

Expected answer:

The real ColPali/ColQwen2 backend interface is implemented, but measured CUDA
metrics are still pending. The real-PDF QA set is automatically generated and needs
manual auditing. The policy diff is rule-based and should be evaluated on more
public policy-version pairs. The browser demo still needs multi-file sessions and
bbox-level highlighting.

Project evidence:

- README states current reproducible results separately from planned claims.
- This roadmap separates implemented features from future milestones.

## Resume Bullet Template

Use only measured results once they are final:

- Built InsureRAG-VLM, a citation-grounded multimodal RAG pipeline for insurance
  policy review, with PDF page-image preprocessing, QA/evidence generation,
  structured citations, abstention, and policy-version diff.
- Created a 20-document public insurance PDF evaluation set with 300 answerable
  QA/evidence examples, hard negatives, document-aware split metadata, and
  local retrieval metrics across text and page-image baselines.
- Implemented retrieval/evaluation harnesses with Recall@k, MRR, nDCG, citation
  precision, evidence recall, unsupported abstention accuracy, latency reporting,
  and calibration curves.
- Added token-efficient local LLM integration with Qwen2.5-3B and deterministic
  evidence extraction plus a 250+ term insurance glossary to reduce hallucination
  risk on high-stakes policy questions.

## Immediate Action Plan

1. Commit the current reproducible MVP.
2. Run `colqwen2_hf` or `colpali_hf` on CUDA and add measured GPU retrieval
   metrics.
3. Manually audit a subset of the 300-example real-PDF QA set.
4. Add error-analysis buckets and calibration plots.
5. Add multi-file upload sessions and bbox-level evidence highlighting.
6. Add Docker/model card/data card polish.
