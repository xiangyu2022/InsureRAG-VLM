# Expanded document-cluster diagnostic

Deterministic contract success, not semantic insurance accuracy; authored document clusters are not a representative population sample.

| Mode | Metric | Success / denominator | Rate | Document-cluster 95% interval |
| --- | --- | ---: | ---: | --- |
| retrieved | gold_evidence_in_context | 131/180 | 72.78% | 61.11%–82.78% |
| retrieved | supported_context_contract_success | 72/180 | 40.00% | 31.11%–48.33% |
| retrieved | supported_answer_coverage | 146/180 | 81.11% | 71.67%–89.44% |
| retrieved | conditional_supported_contract_accuracy | 72/146 | 49.32% | 38.99%–59.20% |
| retrieved | answer_coverage | 147/240 | 61.25% | 53.75%–67.50% |
| retrieved | conditional_answer_contract_precision | 72/147 | 48.98% | 38.71%–58.60% |
| retrieved | unsupported_strict_abstention | 59/60 | 98.33% | 95.00%–100.00% |
| retrieved | unsupported_answer_rate | 1/60 | 1.67% | 0.00%–5.00% |
| retrieved | unsupported_failure_to_strictly_abstain | 1/60 | 1.67% | 0.00%–5.00% |
| retrieved | supported_strict_abstention | 34/180 | 18.89% | 10.56%–28.33% |
| retrieved | retrieval_hit_at_k | 176/180 | 97.78% | 95.56%–99.44% |
| retrieved | generation_complete | 240/240 | 100.00% | 100.00%–100.00% |
| retrieved | json_contract_success | 240/240 | 100.00% | 100.00%–100.00% |
| retrieved | request_error_rate | 0/240 | 0.00% | 0.00%–0.00% |

Full per-document, per-category, and document-family sensitivity tables are in summary.json.
Intervals resample whole documents and remain descriptive for this authored, non-random fixture. Conditional contract accuracy must be interpreted together with answer coverage and unsupported-question behavior.
