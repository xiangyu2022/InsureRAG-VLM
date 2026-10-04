# CPU retrieval diagnostic

CPU local-hashing retrieval and label-integrity diagnostic; no trained BGE or Qwen inference.

## Corpus audit

- answerable_sft_records: `3400`
- unique_answerable_questions: `39`
- questions_with_multiple_gold_pages: `39`
- records_with_ambiguous_question: `3400`
- most_common_questions: `[('Summarize the insurance-specific point supported by this evidence.', 908), ('What coverage information is explained by the evidence?', 587), ('How would you summarize the coverage point on this page?', 587), ('What does this evidence say about the deductible?', 107), ('How should the deductible be explained from this page?', 107)]`
- source_document_splits: `{'train': 37, 'valid': 4, 'test': 6}`
- test_questions: `115`
- test_unique_questions: `23`
- indexed_pages: `201`
- indexed_snippets: `1180`
- test_gold_sources_present: `115`

## current_sft_document_test_split

| Mode | Examples | Hit@1 | Hit@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| dense_only | 115 | 0.0522 | 0.0522 | 0.0728 |
| sparse_only | 115 | 0.0435 | 0.0609 | 0.0510 |
| hybrid_text | 115 | 0.0435 | 0.0522 | 0.0521 |

## legacy_59_development_examples

| Mode | Examples | Hit@1 | Hit@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| dense_only | 59 | 0.0169 | 0.1017 | 0.0535 |
| sparse_only | 59 | 0.0339 | 0.1017 | 0.0694 |
| hybrid_text | 59 | 0.0339 | 0.0678 | 0.0642 |

## Interpretation

- Question templates are reused across many source pages, making exact-source retrieval underdetermined without document context.
- The document split is regenerated from the current 3,850-record dataset; original historical train/test manifests and BGE weights are absent.
- No retriever training, LLM generation, adapter evaluation, image model, or GPU execution occurs in this diagnostic.
- The 59-example suite is selected from SFT data and is a development diagnostic, not a held-out generalization benchmark.
- dense_only and sparse_only modes retain the repository's metadata reranking, table signals, and graph expansion; these are pipeline ablations.
- Hit@k is the fraction of queries with any labeled page in the first k results, not answer correctness or hallucination rate.

## Reproduce

From the repository root, with requirements.txt installed:

```bash
python -m scripts.eval_cpu_diagnostic
```

The script rebuilds CPU indexes and writes input hashes, a test manifest, per-example ranked-source traces, and these summaries.
