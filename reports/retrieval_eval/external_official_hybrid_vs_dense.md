# External Official Hybrid vs Dense Evaluation

- Corpus root: `reports/retrieval_eval/external_official/corpus`
- Manifest root: `reports/retrieval_eval/external_official`
- Retrieval model: `models/retrieval/bge-base-insurerag`
- Hybrid mode: `hybrid_text`
- Dense mode: `dense_only`
- Top-k: `10`

## Dataset

- Total examples: `137`
- Selected documents: `20`
- Source families: `{'state_public_doc': 101, 'naic': 22, 'state_doi': 14}`
- Document kinds: `{'glossary': 22, 'consumer_guide': 96, 'declarations_guide': 3, 'topic_guide': 9, 'coverage_guide': 6, 'consumer_advisory': 1}`
- Preferred terms: `{'liability': 30, 'limit': 5, 'deductible': 16, 'premium': 15, 'coverage': 41, 'replacement cost': 18, 'actual cash value': 11, 'declarations': 1}`

## Valid

- QA file: `reports/retrieval_eval/external_official/valid.jsonl`

| metric | dense_only | hybrid | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 42 | 42 | 0 |
| recall_at_1 | 0.0714 | 0.0238 | -0.0476 |
| recall_at_5 | 0.3333 | 0.1905 | -0.1429 |
| mrr_at_10 | 0.1810 | 0.1011 | -0.0799 |
| ndcg_at_10 | 0.2504 | 0.1602 | -0.0902 |

## Test

- QA file: `reports/retrieval_eval/external_official/test.jsonl`

| metric | dense_only | hybrid | delta |
| --- | ---: | ---: | ---: |
| evaluated_count | 95 | 95 | 0 |
| recall_at_1 | 0.0211 | 0.0526 | 0.0316 |
| recall_at_5 | 0.0842 | 0.1158 | 0.0316 |
| mrr_at_10 | 0.0459 | 0.0842 | 0.0383 |
| ndcg_at_10 | 0.0666 | 0.1150 | 0.0483 |
