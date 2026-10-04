# Development v1 response inspection

AI-assisted inspection on 2026-09-20 of all 48 raw outputs in `qwen25_dev_v1` and `qwen35_dev_v1` (12 cases × two modes × two models), plus the saved prompts for failures. This is a development diagnosis, not a certified semantic score. The fixture lock is unchanged. These runs may differ in incidental runtime-metadata instrumentation; preserve their recorded code hashes and do not present them as the final fixed-pipeline test comparison.

## Qwen3.5:4b failures of the strict supported-case contract

| Case/mode | Deterministic result | Prompt/output inspection | Interpretation |
| --- | --- | --- | --- |
| Delaware auto 01 / retrieved | Gold page hit; answer-key and grounded-key fail | p4 ranks first but prompt contains p5 and p12 only; the $1,500 sentence is absent. Model abstains. | Context packing loses retrieved evidence. Abstention is appropriate for the actual prompt. |
| Delaware auto 04 / retrieved | Retrieval hit and answer-key fail | Gold p9 is absent from top3; prompt contains p6 and p12. Model abstains. | Retrieval miss, then appropriate abstention. |
| Delaware homeowners 04 / retrieved | Key, citation, verbatim quote pass; full-span and grounded-key fail | Answer is 2%; quote is “typically 2% of the property's value,” omitting outer words/punctuation of the gold span. | Sufficient answer/quote for the question; strict contract false negative. Do not call it an incorrect percentage. |

The other supported responses inspected have no evident contradiction to their quoted guide context. All four unsupported cases in each mode return the specified empty-string abstention object. Oracle supported contract is 8/8 on these eight cases, with Wilson lower bound only about 67.6%; it is not evidence of near-perfect population reliability.

## Qwen2.5:3b findings

| Case/mode | Observed issue | Interpretation |
| --- | --- | --- |
| Delaware auto 01 / retrieved | Same absent p4 context; abstains | Context-packing failure, not a wrong amount. |
| Delaware auto 04 / retrieved | Gives 10% from an unrelated multi-line-discount quote on p12, omits duration | Unsupported answer relative to supplied evidence and incomplete requested answer. |
| Delaware auto 04 / oracle | Gives 10% but omits three-year duration from answer (duration present in evidence) | Incomplete short answer; not the same as wrong percentage. |
| Delaware homeowners 01 / retrieved | Gives HO-4 but cites p13 and stitches p13/p6 text | Citation and quote provenance error. |
| Delaware homeowners 02 / retrieved | Correct $5,000 appears but JSON lacks closing brace | Truncated JSON; inspect `done_reason=length` and token cap before comparing content capability. |
| Delaware homeowners 02 / oracle | Answer $2,000 despite quote listing $5,000 for computer equipment | Wrong category amount. |
| Delaware homeowners 03 / oracle | Correct $500, but cites p10 instead of p9 and omits a middle sentence from the claimed verbatim quote | Citation and quote-contract error. |
| Delaware auto 03 and homeowners 01 / oracle | Adds “The guide states:” and enclosing quotation marks within evidence | Verbatim-string contract failure despite useful underlying source quote. |

All four unsupported cases in **both modes** have `abstain=true` and an empty answer. The strict abstention score is only 2/4 retrieved and 0/4 oracle because evidence/source fields contain explanatory text or a citation. Thus these development outputs support better strict-output compliance for Qwen3.5, but do **not** establish that Qwen2.5 invented private policy values on these cases.

## Proposed generic pipeline experiments (development only)

1. Preserve retrieval relevance when constructing context. The current role-bucket ordering places lower-ranked insurance roles before the top relevant page.
2. Remove duplicate/contained snippets and paragraph-like pseudo-table copies; distribute a bounded context allowance across selected pages before final truncation. Add a regression test where a top-ranked fact near the end of a page survives alongside distractors. No benchmark-specific source/page rule is appropriate.
3. Compare the existing local-hashing embedding baseline with a fixed local BGE checkpoint on the same dev queries, inspecting both hit@3 and actual evidence presence in prompts. Freeze the selected configuration before any test inference.
4. Preserve the benchmark's strict metrics and add diagnostic prompt-evidence visibility as a separate non-semantic measure. Do not change gold spans to increase observed scores.

No frozen fixture or core pipeline file was edited as part of this review.
