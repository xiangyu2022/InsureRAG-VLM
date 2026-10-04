# Expanded document-cluster diagnostic

Deterministic contract success, not semantic insurance accuracy; authored document clusters are not a representative population sample.

| Mode | Metric | Success / denominator | Rate | Document-cluster 95% interval |
| --- | --- | ---: | ---: | --- |
| retrieved | gold_evidence_in_context | 29/45 | 64.44% | 42.22%–80.00% |
| retrieved | supported_context_contract_success | 23/45 | 51.11% | 35.56%–64.44% |
| retrieved | supported_answer_coverage | 28/45 | 62.22% | 40.00%–80.00% |
| retrieved | conditional_supported_contract_accuracy | 23/28 | 82.14% | 67.65%–100.00% |
| retrieved | answer_coverage | 28/60 | 46.67% | 30.00%–60.00% |
| retrieved | conditional_answer_contract_precision | 23/28 | 82.14% | 67.65%–100.00% |
| retrieved | unsupported_strict_abstention | 15/15 | 100.00% | 100.00%–100.00% |
| retrieved | unsupported_answer_rate | 0/15 | 0.00% | 0.00%–0.00% |
| retrieved | unsupported_failure_to_strictly_abstain | 0/15 | 0.00% | 0.00%–0.00% |
| retrieved | supported_strict_abstention | 17/45 | 37.78% | 20.00%–60.00% |
| retrieved | retrieval_hit_at_k | 45/45 | 100.00% | 100.00%–100.00% |
| retrieved | generation_complete | 60/60 | 100.00% | 100.00%–100.00% |
| retrieved | json_contract_success | 60/60 | 100.00% | 100.00%–100.00% |
| retrieved | request_error_rate | 0/60 | 0.00% | 0.00%–0.00% |

Full per-document, per-category, and document-family sensitivity tables are in summary.json.
Intervals resample whole documents and remain descriptive for this authored, non-random fixture. Conditional contract accuracy must be interpreted together with answer coverage and unsupported-question behavior.
