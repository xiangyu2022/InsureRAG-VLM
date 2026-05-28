# Clean Exact-Source Retrieval Evaluation

- Manifest: `reports/retrieval_eval/clean_exact_source.jsonl`
- Example count: `59`
- Retrieval mode: `hybrid_multimodal`
- Corpus source: `curated`
- Image signal enabled: `False`
- Top-k: `10`

## Composition

- Document types: `{'declarations': 17, 'base_policy': 17, 'claim_form': 1, 'endorsement': 20, 'billing': 3, 'schedule': 1}`
- Primary clause types: `{'endorsement': 31, 'deductible': 4, 'limit': 9, 'definition': 9, 'premium': 3, 'exception': 2, 'exclusion': 1}`

## Before vs After

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 59 | 59 | 0 |
| recall_at_1 | 0.0339 | 0.0339 | 0.0000 |
| recall_at_5 | 0.1017 | 0.1356 | 0.0339 |
| mrr_at_10 | 0.0770 | 0.0789 | 0.0018 |
| ndcg_at_10 | 0.1064 | 0.1151 | 0.0087 |
