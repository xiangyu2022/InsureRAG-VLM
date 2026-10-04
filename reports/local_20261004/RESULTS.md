# Local Qwen3.5-2B BF16 LoRA experiment

Report generated: 2026-10-04T01:56:26.736234+00:00

## Outcome

Completed 300 optimizer steps over 1200 sample presentations. Results below are paired on the same frozen 96-example test split; 62 answerable and 34 synthetic unsupported examples. Test evidence comes from 9 documents excluded from adaptation training. This is a given-evidence, synthetic-label experiment, not a production or human-reviewed benchmark.

| Metric | Official base | Adapter |
| --- | ---: | ---: |
| Answerable content token F1 | 0.2962 | 0.5153 |
| Exact-source citation precision | 0.9891 | 1.0000 |
| Answerable exact-citation rate | 0.9677 | 1.0000 |
| Answerable lexical-support proxy | 0.2419 | 1.0000 |
| Synthetic abstention precision | 0.8824 | 1.0000 |
| Synthetic abstention recall | 0.8824 | 1.0000 |
| Latency p50 seconds | 3.0721 | 2.5960 |
| Latency p95 seconds | 4.8150 | 4.5357 |

**Do not promote this adapter into the end-to-end answer path.** In the corrected paired retrieved-context diagnostic, content F1 changes from 0.1329 to 0.0830. The primary improvement above applies to supplied-evidence adaptation only; the existing retrieval/packing path remains inadequate.

Answerable F1 delta: +0.2191; document-cluster bootstrap 95% interval [+0.0519, +0.3719] over 9 document groups. The small, synthetic sample limits generalization.
Paired answerable cases: 38 improved, 23 worse, 1 tied.

## Training evidence

- Model: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` (Apache-2.0).
- GPU: RTX 4070 Laptop 8GB; BF16 LoRA; text-only inputs; frozen visual encoder.
- 600 training rows; rank 8, alpha 16; batch 1, accumulation 4; two epochs; seed 20261004.
- Trainable parameters: 5,455,872; peak allocated GPU memory: 4684.6 MiB.
- Training wall time: 588.2s; mean training loss: 0.0363.
- Dev token NLL: base 0.4028; selected 0.0590.
- Selected adapter: `models\local_20261004_lora\step-300`; final candidate: `models\local_20261004_lora\final`.
- Selection uses dev NLL only and includes the base candidate. If selected_adapter is null, the adapter did not beat base under the frozen selection rule.

## Data, evaluation and limitations

- Curated input: 3850 rows across 47 source identifiers. Exact/near-duplicate evidence groups reduced this to 43 independent splitting groups.
- Filtering: `{"topic_missing_from_evidence": 112, "duplicate_question_evidence": 138}`. Split hashes and document assignments are in `data_audit.json`.
- Questions, references and unsupported labels are rule-generated from public insurance sources. No human labeling or semantic correctness certification is claimed.
- Content F1 excludes templated introductory text and citation suffixes. It still rewards matching synthetic extractive references; valid paraphrases can score lower.
- Exact citation matching and lexical/number support are automatic proxies, not entailment judgments. Labels can be incomplete or semantically noisy.
- No overlap of original document groups or exact evidence hashes is allowed between adaptation train/dev/test. Near duplicates below the lexical threshold may remain.
- Inference is greedy with thinking disabled and a 128-token output cap for both models. Desktop load is uncontrolled, so latency is indicative.
- This does not rerun historical trained-BGE scores or prove end-to-end retrieval gains.

## Supporting artifacts

- `frozen_experiment_plan.json`, `hardware.json`, `requirements-lock.txt`
- `data_audit.json`, `train.jsonl`, `dev.jsonl`, `test.jsonl`
- `training_log.jsonl`, `training_summary.json`, adapter checkpoints and optimizer state
- `base_test_predictions.jsonl`, `adapter_test_predictions.jsonl`, `comparison.json`
- `largest_answer_regressions.json`, `retrieval_baseline.json`, `unit_tests.log`
- Reproduction instructions: `docs/local_experiment_20261004.md`

## Supplemental base_retrieved_context_test

```json
{
  "n": 62,
  "n_answerable": 62,
  "n_unsupported": 0,
  "answerable_content_f1": 0.1329236019945298,
  "citation_precision_exact_source": 0.23076923076923078,
  "answerable_citation_rate": 0.14516129032258066,
  "answerable_lexical_support_proxy": 0.016129032258064516,
  "abstention_precision": null,
  "abstention_recall": null,
  "abstention_counts": {
    "tp": 0,
    "fp": 36,
    "fn": 0
  },
  "latency_p50_seconds": 3.801706950063817,
  "latency_p95_seconds": 5.855794930050615,
  "generated_tokens": 5357
}
```

This uses fixed retrieved/packed contexts and original synthetic positive labels only. Unsupported labels were excluded because their original labels only apply to supplied evidence, not the whole corpus. Abstention precision/recall are NOT MEASURED here: retrieval can remove support, and we have no new context-level answerability judgments. Raw metric files retain the aggregator's zero-denominator convention; null above is the appropriate interpretation. Gold page IDs are not supplied in the prompt. It is a retrieval-context diagnostic, not an additional independent benchmark.

## Supplemental adapter_retrieved_context_test

```json
{
  "n": 62,
  "n_answerable": 62,
  "n_unsupported": 0,
  "answerable_content_f1": 0.08296976833184978,
  "citation_precision_exact_source": 0.2702702702702703,
  "answerable_citation_rate": 0.16129032258064516,
  "answerable_lexical_support_proxy": 0.16129032258064516,
  "abstention_precision": null,
  "abstention_recall": null,
  "abstention_counts": {
    "tp": 0,
    "fp": 24,
    "fn": 0
  },
  "latency_p50_seconds": 2.7814333999995142,
  "latency_p95_seconds": 7.366090559947765,
  "generated_tokens": 2560
}
```

This uses fixed retrieved/packed contexts and original synthetic positive labels only. Unsupported labels were excluded because their original labels only apply to supplied evidence, not the whole corpus. Abstention precision/recall are NOT MEASURED here: retrieval can remove support, and we have no new context-level answerability judgments. Raw metric files retain the aggregator's zero-denominator convention; null above is the appropriate interpretation. Gold page IDs are not supplied in the prompt. It is a retrieval-context diagnostic, not an additional independent benchmark.

## Supplemental prompt correction

The initial supplemental prompt put a citation-selection instruction in a Source field while requiring that field to be copied exactly. The adapter often copied the instruction as its citation. Complete first-run artifacts are preserved as `prompt_v1_*` (base/adapter content F1 0.1373/0.0856; citation precision 0.3103/0). The corrected multi-source prompt omits that contradictory field; both base and fixed step-300 are rerun on the same contexts/decoding/scoring. Primary results are unaffected. See `supplemental_prompt_revision.json` and the protocol document for disclosure; no model tuning followed test inspection.

## Failure analysis and decision

Use this adapter as a local research candidate for concise source-formatted answers; do not replace a production answerer or claim general insurance accuracy from these labels.

- 23/62 answerable primary-test cases regress in content F1. Example `a0c5012fc2fc17b97854`: asked to explain actual cash value, the adapter extracts a deductible definition instead. It passes exact citation and lexical support, demonstrating those proxies do not measure relevance.
- Example `4a505aeba950d93a5046`: reference concerns HO-6 contents/interior coverage; the adapter extracts a generic suggestion to check covered/excluded perils. F1 decreases by 0.5771 despite a valid source string.
- All questions/references and unsupported labels are synthetic. Perfect refusal precision/recall on 34 negatives does not establish robustness to genuine missing-evidence questions.
- Independent audits find no forbidden document-group/hash/near-duplicate overlap under the frozen criteria, but train/test share 33 question templates. Semantic publication-version independence and pretraining contamination are unverified.
- Base hits the 128-token cap on 5/96 primary-test outputs; adapter on 0/96. This is the same frozen generation protocol, but truncation and output brevity affect F1/citation comparisons.
- Human relevance, semantic entailment, visual-input adaptation, production serving, robust latency and historical trained-BGE/QLoRA reruns are unmeasured.

- Fixed retrieval contexts preserve the gold source for only 12/62 examples, although 34/62 have that source in top-5 retrieval. Role ordering and the character budget can exclude the relevant retrieved page. This diagnostic measures the existing packer as well as answer adaptation; it cannot isolate model quality.
- CLI execution succeeds, but the travel-insurance smoke retrieves a travel page and packs auto-insurance text; an abstention is a functional smoke outcome, not proof of a useful travel answer.

The fixed step-300 checkpoint was not changed after observing test results. Scoring/prompt code was extracted into a CPU-only module during primary inference without changing behavior; independent re-scoring matches saved outputs/metrics. Initial baseline/training invocation hashes were not captured; subsequent invocation hashes and final source/artifact hashes are provided without retroactively claiming that provenance.
