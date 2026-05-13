# InsureRAG-VLM GPU Benchmark

> Note: This benchmark summary predates the `hybrid_multimodal` default pipeline. The recorded
> `local_text` and `local_image` numbers remain useful as historical baselines, but they do not
> represent the current default query path after the hybrid multimodal migration.

## Run

- Data folder: `data\00_raw\external\public_docs`
- Dataset hash: `3b1b961d8ce3`
- PDFs: 20
- Pages: 168
- QA rows: 70 (20 unsupported)
- Primary backend: `local_image`
- CUDA device: not available

## Retrieval

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Index sec | Peak CUDA MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local_text | 0.2200 | 0.4800 | 0.3347 | 0.4109 | 30.9 | 44.7 | 0.3 | 0 |
| local_image | 0.2000 | 0.5200 | 0.3429 | 0.4266 | 21.3 | 32.7 | 1.7 | 0 |

## Answering And Abstention

- F1: 0.0801
- Citation precision: 0.0800
- Evidence recall: 0.0800
- Unsupported abstention accuracy: 0.9000
- Coverage: 0.3143

## Error Groups

- citation_mismatch: 3
- retrieval_miss: 27
- unsupported_false_positive: 2
- weak_answer_extraction: 16
