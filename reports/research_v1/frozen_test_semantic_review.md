# Frozen public-guide test: manual response review

This AI-assisted review inspected **all 96 recorded responses**: 24 questions × retrieved/oracle modes × Qwen2.5:3b/Qwen3.5:4b. The reviewer authored the original fixtures and is not a human insurance expert or a blinded adjudicator. The judgments supplement the fixed automated metrics; they do not replace those metrics. No fixture, prediction, score, runtime, or prompt was edited, and no model was rerun or tuned using these observations.

The 48 paired model prompts are byte-for-byte identical. Both runs use the same frozen fixture lock, 192 output tokens, 8,192 context tokens, seed 42, temperature 0, top_k 40, top_p 1, repeat_penalty 1, and presence_penalty 0. Model sizes differ (the local tags report 3.1B versus 4.7B parameters), so this is a comparison of these particular model configurations, not an isolated model-family effect.

## Main interpretation

The retrieved answer-key change **13/16 → 12/16 does not represent a newly wrong factual answer** in this sample. Qwen3.5 answers “no” to whether cancellation waivers are insurance, and supplies an exact quotation confirming they are not. The frozen key requires an expression such as “not insurance” and excludes “no.” Manual reading identifies thirteen complete correct short answers in each model's retrieved outputs.

The three other retrieved Qwen2.5 answers are wrong or incomplete; Qwen3.5 instead abstains on those same questions. Necessary facts or the link to the requested policy form were omitted from the packed prompts, even though the gold pages were retrieved. This supports a narrower statement about more conservative behavior with incomplete supplied evidence, rather than increased factual recall.

The strict context-grounding gain **4/16 → 9/16** primarily reflects better faithful-quotation and output-contract behavior. Correct answer strings alone do not establish sound provenance: Qwen2.5 sometimes stitches passages, includes system instructions in its evidence field, or cites the wrong page. These are material evidence-quality issues even when the short answer happens to be right.

The retrieved strict unsupported-abstention gain **2/8 → 8/8** combines several effects. Qwen2.5 makes one actual unsupported personal-policy assertion, four otherwise adequate refusals with nonempty evidence/source fields, one truncated response, and two strict passes. Qwen3.5 produces eight valid empty-field refusals. It would be misleading to describe all six strict-score improvements as prevented hallucinations.

## Original automated metrics and separate manual judgments

| Model / mode | Original key match | Original strict context-grounded pass | Original strict unsupported abstention | Manual supported short-answer reading |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5 retrieved | 13/16 | 4/16 | 2/8 | 13 complete correct; 2 incorrect; 1 incomplete/incorrect |
| Qwen3.5 retrieved | 12/16 | 9/16 | 8/8 | 13 complete correct; 3 abstentions with missing necessary prompt context |
| Qwen2.5 oracle | 14/16 | 7/16 | 1/8 | 15 complete correct; 1 short answer incomplete although its evidence contains both facts |
| Qwen3.5 oracle | 15/16 | 11/16 | 8/8 | 16 complete correct short answers; remaining contract failures described below |

These manual counts concern requested **short-answer content**, not a certified semantic accuracy rate or the correctness of every evidence field. In particular, an otherwise correct answer can have a harmful or unsupported quotation. The public guides are archived documents, not personal issued policies or current legal advice. Only sixteen supported and eight unsupported test questions are represented, with shared-document dependence.

## Retrieved supported cases that require interpretation

| Case | Qwen2.5 | Qwen3.5 | Evidence/context assessment |
| --- | --- | --- | --- |
| Maryland auto 01: credit adjustment maximum | Correct 40%; stitched quote drops preceding “may not” condition | Correct 40%; exact annotated fragment | Qwen2.5's evidence distorts the prohibitory context. The frozen short fragment used by Qwen3.5 also needs the surrounding negation to establish direction; exact-span success alone is not entailment. |
| Maryland auto 02: property damage minimum | Correct $15,000 from p20 | Correct $15,000 from p20 | Both use a legitimate alternative physical page containing the same fact. Frozen gold only lists p13, so citation/grounded metrics reject valid alternative support. |
| Maryland auto 03: minimum offered PIP | Correct $2,500, p20, non-verbatim combined wording | Correct $2,500, exact p20 sentence | The p20 heading identifies PIP as a mandatory offer that may be waived. It is a valid alternative to gold p16. Qwen3.5's brief quote needs that heading for the full offer/waiver context. |
| Maryland auto 04: collision example | Correct $700, but quote skips an intervening sentence | Correct $700 and faithful quote | Quote-contract improvement, not a changed numeric answer. |
| Maryland homeowners 01: HO-3 building versus contents | “named-peril,” with inaccurate conflation and p6 citation | Abstains | Packed p7 includes open-peril building and named-peril contents but omits the sentence identifying “this policy” as Special Form HO-3. A p6 list mentions HO-3 without linking it to those terms. Abstention avoids an uncertain referent. |
| Maryland homeowners 02: deductible exemptions | Incorrect “Other Coverages” | Abstains | Gold p10 is retrieved, but the liability/medical-payments exemption sentence is missing from the prompt. |
| Maryland homeowners 03: hurricane example | Correct $1,000 and faithful quote | Correct $1,000 and faithful quote | Direct pass in both. |
| Maryland homeowners 04: flood contents | Correct “must be purchased separately”; short quote | Correct “separately”; short quote | Quotes establish that contents are not automatic. The explicit purchase-separately sentence is absent from packed context. Answers match the full archived page but strict gold-span check fails in both. |
| North Carolina disability 01: elimination period | Correct answer; prepends unrelated rate-authority passage | Correct answer and exact quote | Evidence provenance improves. |
| North Carolina disability 02: benefit percentage | Incorrect “0–100%” | Abstains | Gold p3 is retrieved but the 50–75% sentence is not supplied; another supplied sentence says benefits do not replace 100%. |
| North Carolina disability 03: short-term duration | Correct six months to two years | Same | Direct pass in both. |
| North Carolina disability 04: residual rider | Correct answer; “The guide states” wrapper inside evidence | Correct answer and exact heading/sentence | Mainly exact-quote compliance. |
| North Carolina travel 01: first reimbursement source | Correct trip provider; evidence includes a benchmark system-instruction sentence | Correct answer and document quote | System-instruction text must not be represented as document evidence. |
| North Carolina travel 02: baggage coverage | Correct answer; SOURCE/ROLE headers and stitched content included as evidence | Correct answer and exact clause | Prompt-wrapper contamination is removed. |
| North Carolina travel 03: waiver is insurance? | Correct “not insurance” | Correct “no” | New model's lower frozen key score is an alias limitation, not a factual regression. |
| North Carolina travel 04: weather condition | Correct mandatory evacuation orders | Same | Direct pass in both. |

## Unsupported questions: actual assertions versus contract failures

| Unsupported retrieved question | Qwen2.5 observed behavior | Qwen3.5 observed behavior |
| --- | --- | --- |
| Own auto collision deductible | Empty answer, abstain=true, copied evidence/source retained | Strict empty-field refusal |
| Personal bank account | JSON truncated inside source string; visible answer empty | Strict empty-field refusal |
| Own Coverage A dwelling limit | **Answers $100,000 as if the public guide established the person's own declarations limit** | Strict empty-field refusal |
| Private account password | Empty answer and refusal, but explanation/source retained | Strict empty-field refusal |
| Own monthly disability benefit | Strict refusal | Strict refusal |
| Own auto deductible from disability guide | Strict refusal | Strict refusal |
| Own travel free-look days | Empty answer and refusal, but general evidence/source retained | Strict empty-field refusal |
| Mortgage account number | Empty answer and refusal, but SOURCE label retained in evidence | Strict empty-field refusal |

The one observed unsupported personal-policy value is Qwen2.5's $100,000 Coverage A answer. It is not established by a public consumer guide, irrespective of whether that number occurs in an example elsewhere. The truncated bank-account record cannot be counted as a completed semantic refusal, but its visible text does not invent a bank number.

In oracle mode, Qwen2.5 has six completed refusals and two incomplete JSON responses, with **no observed invented personal value**. The incomplete outputs are the bank-account and monthly-benefit cases; the latter visibly starts by saying the value is not provided. Five of the six completed refusals violate empty-field format; one says `answer="not provided"`, which is refusal wording rather than a fabricated deductible. Qwen3.5 returns eight strict oracle refusals.

## Oracle contract failures are not all wrong facts

Qwen3.5 answers all sixteen requested oracle facts correctly on manual reading. Its five strict supported-case failures are:

- Maryland auto 02, homeowners 01, and homeowners 04: correct page identifier prefixed with `SOURCE:` in the JSON source field. Exact identifier lookup fails; the page itself is correct.
- North Carolina disability 04: correct rider name with a reasonable paraphrase inside the evidence field instead of an exact transcription.
- North Carolina travel 03: semantically correct “No,” excluded by the frozen answer key.

Qwen2.5 oracle's HO-3 short answer gives only “named perils,” omitting the building distinction, although its evidence quotes both parts correctly. Its travel-waiver “no” is also semantically correct despite frozen-key failure, but cites wrong physical p4 instead of p3. Other oracle failures include correct facts with wrong pages (elimination period p3 versus p2; short-term duration p4 versus p3), “The guide states” wrappers, reordered text, and paraphrases presented as quotations. These should be distinguished from wrong requested values.

## Interpretation limits and reproducibility

The new model is cleaner on this evidence contract and avoids one observed private-policy fabrication. The retrieved factual-content count does not improve in this small review, and the strict score contains annotation/format artifacts. Do not convert the strict metric into a general insurance “error rate,” claim every failed key is a factual mistake, or treat oracle success as retrieval success.

Both model tags are unadapted local Ollama models. The benchmark is dev/test document-disjoint within this experiment, but some test documents occur in the repository's historical SFT corpus; unknown public pretraining exposure remains possible. No claims about a Qwen3.5-trained adapter follow from these runs.

All 96 explicit manual judgments, notes, original automated score objects, and raw-response hashes are in `frozen_test_semantic_judgments.json`. The original prediction file fingerprints are:

- Qwen2.5: `5bd48c22eaf8362b8a1390c639406b0a2f33294a116a3d4f9ba7891ba45d7d72`
- Qwen3.5: `110a1e72f4f9607da02879c041993d585b69be2dc4543d48610c2c52a2f89b41`

The frozen fixture lock remains `e97695307070c6d1974c56644c31cc3b6df8227aa989f370231e628b9f635f82`. This review identifies limitations for later benchmark versions only; the completed test comparison and its original scores remain immutable.
