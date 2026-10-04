# Evaluation audit — 2026-09-20

This audit preserves historical reports. New CPU results are in
[`retrieval_eval/cpu_diagnostic/summary.md`](retrieval_eval/cpu_diagnostic/summary.md).
No BGE checkpoint, Qwen adapter, or GPU training was run for this audit.

## Verified corpus and historical training

- Current corpus: 56 source documents, 201 pages, 1,180 snippets, and 1,381 combined records.
- Current SFT pool: 3,850 records, including 3,400 answerable records and 450 unsupported records.
- Recorded model: Qwen2.5-7B-Instruct text model with QLoRA. Serving providers and optional ColQwen2 page-image retrieval are separate components.
- Historical continuation report: 3,231 training examples, 2 epochs, 1 NVIDIA A40, approximately 10.9 GB peak CUDA memory. These are historical run values, not a rerun on the current dataset.
- Historical spot check: 8 examples, including 4 unsupported examples. Token F1 of 0.2168 versus 0.7443 measures overlap on that tiny development sample. Neither `1 - F1` nor `4/4` abstentions is a defensible population hallucination/error rate.

## Bugs fixed

1. Training manifests write `gold_sources`; retrieval evaluators previously read only `evidence_sources` / `citations`. The shared matcher now accepts all three schemas and explicit `gold_page_keys`.
2. The benchmark evaluator used exact source strings, unlike the QA evaluator. Both now use the same page-key and normalized-source matching; a web URL and its `#page=1` alias match.
3. Answerable examples without any gold source or page key now raise an error instead of being silently counted as retrieval misses.
4. Training-negative construction could count a web URL alias of the positive page as a hard negative, and `gold_in_context` could be false for the same alias. Both now normalize consistently.
5. The training builder and before/after evaluator now pass their requested data folder as the curated dataset directory. Previously a nondefault folder could accidentally resolve to the repository's default curated corpus.
6. Before/after evaluation now requires an available local checkpoint or the explicit `local-hashing` baseline, checks identical question/label multisets, and records model names and manifest hashes. A missing BGE directory must not silently become a hashing run labeled as BGE.

The original manifests referenced by `retrieval_eval/before_after.md` and trained BGE weights
are not included in the checkout. Consequently the historical all-zero table cannot be
reconstructed here, and these bugs alone do not prove its exact cause. It remains unchanged.

## Dataset limitation exposed by the CPU run

There are only **39 distinct questions across 3,400 answerable SFT records**, and every
question maps to more than one gold source page. The three most common generic prompts
account for **2,082 records (61.24%)**. Such prompts are meaningful when their evidence is
supplied to a generator, but do not identify a particular policy/page for global retrieval.
For example, “What coverage information is explained by the evidence?” cannot identify
which of many documents the evaluator intends without additional context.

The diagnostic regenerates a deterministic source-document split with seed 42: 37 train,
4 dev, and 6 test source documents. The test contains **115 answerable records but only
23 distinct questions**. All 115 gold sources occur in the 201-page index. Hybrid-text
Hit@5 is **5.22%**, with MRR@10 **0.0521**. The separate 59-example legacy development
selection has Hit@5 **6.78%** and MRR@10 **0.0642**. These values neither reproduce the
historical BGE run nor establish a model improvement; repeated template rows are not
independent evidence of generalization.

`dense_only` and `sparse_only` are existing pipeline modes: metadata reranking, table
lookup, and graph expansion still apply. The dense branch uses local hashing, not BGE.
The script records every ranked source list, input SHA-256 hashes, and its exact test
manifest for reproducibility.

## Next valid measurement

Create document/packet-scoped questions that identify the intended policy and numeric or
coverage question; adjudicate all acceptable source pages. Freeze source-family-disjoint
splits before the first adapter training, and build dev/test questions independently of
training templates. Do not use evidence text or the gold source to alter test queries.

For an answer-error claim, predefine separate metrics and denominators: incorrect answers
among answered questions, unsupported-question answer rate, citation-support accuracy,
and answer coverage/abstention rate. Report counts and confidence intervals; select
thresholds on dev, evaluate once on test. Keep prompt, checkpoint, input hashes, predictions,
and adjudication criteria. No improved error percentage is claimed by this audit.

## Reproduce

```bash
python -m pytest tests/test_retrieval_evaluation.py tests/test_training_data.py tests/test_reliability.py -q
python -m scripts.eval_cpu_diagnostic
```
