# Packing repair: exploratory regression

**These test examples were previously observed. This is an engineering regression, not a new independent benchmark. No training or checkpoint selection occurred.**

The legacy packer reordered pages by role, copied long raw snippets, then trimmed the whole concatenation. The new packer preserves retrieval rank, deduplicates query-selected sentence fragments, allocates a separate budget per source, and records exact source/fragment associations. The HF path checks the complete Qwen-tokenized prompt against 2,048 input tokens, reserves 128 output tokens, and verifies the model context limit without tokenizer truncation.

## Evidence availability

| Split | Packing | Top-5 gold source | Packed gold source | Exact reference span | Max prompt tokens |
|---|---|---:|---:|---:|---:|
| dev (25) | legacy | 13 | 5 | 1 | 862 |
| dev (25) | balanced | 13 | 11 | 0 | 782 |
| test (62) | legacy | 34 | 12 | 0 | 887 |
| test (62) | balanced | 34 | 20 | 6 | 829 |

Source presence and exact-span retention are proxies; neither proves relevance or completeness. Gold is used for these measurements only after both contexts are produced, never for page/fragment selection.

## Paired generation

| Split | Model / packing | Content F1 | Gold-source precision | Packed-source resolution | Lexical-support proxy | Abstentions |
|---|---|---:|---:|---:|---:|---:|
| dev | base_legacy | 0.1422 | 0.1176 | 0.3529 | 0.0000 | 19 |
| dev | adapter_legacy | 0.0347 | 0.4167 | 0.9167 | 0.2000 | 13 |
| dev | base_balanced | 0.1524 | 0.2727 | 0.5455 | 0.0000 | 17 |
| dev | adapter_balanced | 0.1013 | 0.7692 | 1.0000 | 0.4000 | 11 |
| test | base_legacy | 0.1329 | 0.2308 | 0.4872 | 0.0161 | 36 |
| test | adapter_legacy | 0.0830 | 0.2703 | 0.9189 | 0.1613 | 24 |
| test | base_balanced | 0.1460 | 0.3143 | 0.4571 | 0.0161 | 39 |
| test | adapter_balanced | 0.1613 | 0.3830 | 1.0000 | 0.2903 | 15 |

All original positive examples are retained (25 dev, 62 exposed-test), with the same document-scoped questions. No topic-based filtering or favorable-subset headline is used. These queries include a document identifier as in the initial benchmark, not a gold page or gold answer. The unchanged base and fixed step-300 use greedy decoding with thinking disabled and the same output limit.

## Interpretation and limits

- Separate retrieval misses (gold absent from top-5), packing losses (retrieved gold source omitted), and possible generation errors. Even when the source is present, an absent reference span is not proof that the packed evidence is unanswerable.
- Context-level abstention precision/recall are NOT MEASURED. Original positive labels do not transfer automatically to truncated/retrieved contexts. Raw aggregator zero-denominator values are not valid refusal-quality scores here.
- Some synthetic questions ask only to summarize evidence but their references name a specific definition from a long glossary. Reference F1 cannot certify unrestricted RAG correctness for such under-specified queries.
- Citation resolution only checks that a generated identifier names an actually packed source; gold-source matching and lexical coverage do not establish entailment or factual correctness.
- The final base-model CLI now retains/cites the travel-insurance page, but turns its rhetorical cruise/travel-agency example into a supposed prerequisite. This qualitative failure (`cli_review.json`) remains despite a valid citation. The adapter is not promoted; the CLI requires an explicit adapter argument.
- The token capacity guard applies to the explicit local HF experiment/CLI. Other provider clients still need their own tokenizer-aware envelope; generic packing retains its character-budget fallback.
- Sentence fragments may be shortened within a source; truncation is recorded in the packing audit. Preserved source IDs do not guarantee all qualifications survive a finite context budget.
- New-holdout feasibility: 8 unseen-source candidates after excluding prior SFT and the inspected travel CLI document; 5 have high shared-clause/duplicate risk, leaving only 3 lexically unflagged sources with no independent labels. No new confirmatory holdout was manufactured.
- Complete regression suite: 53 passed. Actual Qwen-tokenizer Unicode fixture retains three source identifiers within 400 prompt tokens. Real generated-PDF tests cover relevant-page retention and exact source mapping.

## Reproduce and inspect

Use `docs/packing_experiment_20261004.md` in a fresh checkout. The frozen protocol records source/data/checkpoint hashes. Existing first-round evidence under `reports/local_20261004` is untouched. `first_round_code.patch` preserves the pre-repair implementation.
Supporting files: `frozen_protocol.json`, `holdout_feasibility.json`, `*_packing_audit.json`, per-example predictions, `failure_examples.json`, `comparison.json`, `unit_tests.log`, and `artifact_hashes.json`.
