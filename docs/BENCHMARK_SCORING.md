# Research benchmark scoring contract

The 36 manually checked guide questions and their document-separated development/test split are frozen in `data/benchmarks/research_v1/manifest.lock.json`. These are archived public guides, not private policy records. The held-out test split contains 24 questions across four documents; it is not document-disjoint from the historical project SFT corpus. The compared Ollama base deployments do not load that adapter. Public pretraining exposure is unknown.

## Versions and primary metric

`v1_corpus_quote` is the historical scoring contract. Its `grounded_key_pass` requires an expected answer key, the gold page citation, a verbatim quote in the full corpus page, and a gold evidence span covered by that quote. It did not verify that the quoted passage was supplied to the model. Original runs are retained unchanged.

`v2_context_audit` retains those four checks under the same historical name and adds:

- `source_in_context`: the returned source occurs as a SOURCE header in the evidence section of the actual recorded prompt.
- `evidence_in_context`: the complete returned quote occurs in that source's supplied section after whitespace normalization. A matching quote on another supplied page, a partially truncated quote, or a source merely mentioned in the question does not pass.
- `generation_complete`: transport completed and the response was not stopped at the output-token limit. Missing completion metadata fails closed.
- **`context_grounded_key_pass`**, the primary answer metric: all historical checks, both context checks, and completion must pass.
- `completed_strict_unsupported_abstention`: a completed response with `abstain=true` and empty answer, evidence, and source strings. Historical `strict_unsupported_abstention` remains available for comparison.

Errors stay in the appropriate denominator. Empty groups have null rates. Supported-answer metrics divide by answerable cases; unsupported abstention divides by unsupported cases. Retrieval hit measures whether the gold page appeared in the retrieved ranking, independently of whether context packing retained its evidence. Oracle mode deliberately supplies full gold pages and must be reported separately.

These are deterministic key and quote checks, not semantic entailment, general insurance accuracy, or a production error rate. Wilson intervals assume independent questions and do not account for document clustering. A passage can contain an expected key without supporting every clause of a response; full-answer semantic assessment still requires review.

## Historical audit without new inference

```powershell
python scripts/rescore_research_benchmark.py --run reports/research_v1/qwen35_dev_v1 --output reports/research_v1/qwen35_dev_v1_score_v2
```

The script requires a new output directory, a completed source run, matching recorded prediction and frozen-benchmark hashes, and annotation-consistent request identities. It copies raw prompts/responses, retains `previous_scores`, and writes a separate summary plus source/scorer hashes. It makes no model request and never updates the original files. Compare the v2 primary metric only between v2-scored reports; the summary function rejects historical or mixed score versions.

The first Qwen3.5-4B and Qwen2.5-3B development runs were each audited on 24 recorded requests. Neither had a historical passing answer invalidated by the new context check. This closes a scoring gap; it is not a measured model improvement. The v2 scoring contract was defined during development, before the final test runs.
