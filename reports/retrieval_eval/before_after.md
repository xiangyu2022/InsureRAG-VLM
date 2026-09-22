# Retrieval Before/After Evaluation

> **Historical evaluation — not a v2 benchmark.** Historical result: generated before the binary_page_v2 evaluation contract. Repeated synthetic questions, legacy first-hit scoring and the former heuristic dense_only baseline limit interpretation. These scores have not been recomputed. See docs/retrieval_evaluation.md and reports/retrieval_eval/v2/manifest_audit.json.

- Data folder: `data/04_curated`
- Retrieval mode: `hybrid_multimodal`
- Corpus source: `curated`
- Image signal enabled: `False`
- Top-k: `10`

## Dev

- QA file: `reports/training_data_dense/retrieval_dev.jsonl`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 29 | 29 | 0 |
| recall_at_1 | 0.0000 | 0.0000 | 0.0000 |
| recall_at_5 | 0.0000 | 0.0000 | 0.0000 |
| mrr_at_10 | 0.0000 | 0.0000 | 0.0000 |
| ndcg_at_10 | 0.0000 | 0.0000 | 0.0000 |

## Test

- QA file: `reports/training_data_dense/retrieval_test.jsonl`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 298 | 298 | 0 |
| recall_at_1 | 0.0000 | 0.0000 | 0.0000 |
| recall_at_5 | 0.0000 | 0.0000 | 0.0000 |
| mrr_at_10 | 0.0000 | 0.0000 | 0.0000 |
| ndcg_at_10 | 0.0000 | 0.0000 | 0.0000 |
