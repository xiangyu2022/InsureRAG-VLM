# Hybrid vs Dense Retrieval Evaluation

- Manifest root: `reports/retrieval_eval/expanded_targeted`
- Retrieval model: `models/retrieval/bge-base-insurerag`
- Hybrid mode: `hybrid_multimodal`
- Dense mode: `dense_only`
- Corpus source: `curated`
- Image signal enabled: `False`
- Top-k: `10`

## Composition

- Total examples: `63`
- Document types: `{'declarations': 18, 'endorsement': 20, 'base_policy': 20, 'claim_form': 1, 'schedule': 1, 'billing': 3}`
- Primary clause types: `{'endorsement': 31, 'limit': 13, 'deductible': 4, 'premium': 3, 'exception': 2, 'definition': 9, 'exclusion': 1}`

## Valid

- QA file: `reports/retrieval_eval/expanded_targeted/valid.jsonl`

| metric | dense_only | hybrid | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 20 | 20 | 0 |
| recall_at_1 | 0.0000 | 0.0000 | 0.0000 |
| recall_at_5 | 0.2500 | 0.1500 | -0.1000 |
| mrr_at_10 | 0.0875 | 0.0617 | -0.0258 |
| ndcg_at_10 | 0.1281 | 0.1162 | -0.0118 |

## Test

- QA file: `reports/retrieval_eval/expanded_targeted/test.jsonl`

| metric | dense_only | hybrid | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 43 | 43 | 0 |
| recall_at_1 | 0.0233 | 0.0465 | 0.0233 |
| recall_at_5 | 0.0698 | 0.1395 | 0.0698 |
| mrr_at_10 | 0.0468 | 0.0865 | 0.0397 |
| ndcg_at_10 | 0.0727 | 0.1196 | 0.0469 |
