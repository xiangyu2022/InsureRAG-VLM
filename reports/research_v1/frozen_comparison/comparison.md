# Matched frozen-test deployment comparison

Left: `qwen2.5:3b`. Right: `qwen3.5:4b`.

complete request schedule, exact input labels, benchmark/prompt/code/decoding identities and prediction checksums matched.

| Mode / metric | Left successes / n | Right successes / n | Right minus left (pp) |
| --- | ---: | ---: | ---: |
| oracle / answer_key_match | 14/16 | 15/16 | +6.25 |
| oracle / citation_match | 13/16 | 13/16 | +0.00 |
| oracle / context_grounded_key_pass | 7/16 | 11/16 | +25.00 |
| oracle / completed_strict_unsupported_abstention | 1/8 | 8/8 | +87.50 |
| retrieved / retrieval_hit_at_k | 14/16 | 14/16 | +0.00 |
| retrieved / answer_key_match | 13/16 | 12/16 | -6.25 |
| retrieved / citation_match | 12/16 | 11/16 | -6.25 |
| retrieved / context_grounded_key_pass | 4/16 | 9/16 | +31.25 |
| retrieved / completed_strict_unsupported_abstention | 2/8 | 8/8 | +75.00 |

- Model family, parameter count, and quantized artifact differ; this does not isolate a causal version effect.
- Tiny AI-authored/agent-reviewed public-guide diagnostic; no independent human adjudication.
- Shared documents make per-question Wilson intervals descriptive; no significance or population-generalization claim.
- Strict answer/evidence contracts are not semantic correctness or a production error rate.
- Transport failures remain in denominators; they are not dropped from a comparison.
