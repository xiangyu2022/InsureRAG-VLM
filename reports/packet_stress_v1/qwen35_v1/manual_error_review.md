# Qwen3.5 synthetic PDF stress run v1: manual error review

Reviewed 2026-09-20, approximately 22:40 UTC. This is a separate AI-assisted inspection of every raw answer, served answer, status reason, and the relevant saved prompts. The reviewer also authored the synthetic fixtures; this is not independent human adjudication or insurance-expert certification. No inference was repeated and no original result, fixture, or pipeline file was modified for this review.

**The requested fact is supported in all sixteen answerable raw responses. Eight are suppressed by application postprocessing. All eight unsupported raw responses decline to establish the requested missing fact or controlling term; the automated raw abstention score of 5/8 misses three phrasings.** Therefore the change from raw 5/8 to served 8/8 must not be described as a reduction in model hallucinations. It is partly a difference between wording recognition and the served boolean.

These statements describe this small authored diagnostic only. They do not establish real-policy accuracy, exhaustive semantic correctness, or a production error rate. B01 and D01 include valid additional comparison facts but cite only one of the documents supporting the combined explanation; improving per-claim citation coverage remains useful.

## Unchanged automated results versus manual observations

| Cases | Original raw key/source result | Original served result | Manual observation |
| --- | --- | --- | --- |
| A01, A02, A03, A04 | Pass each | Pass each | Correct separate dwelling/personal-property/deductible/liability value and cited physical page. |
| B01 | Pass | Abstains | Correct actual collision deductible $2,500; accurately distinguishes the $500 educational example. Source parser loses the explicit declaration citation, then chooses the guide. |
| B02 | Pass | Abstains | Correct comprehensive deductible $1,400 and declaration citation. Missing lexical token `actually` triggers rejection. |
| B03 | Pass | Abstains | Correct explicitly requested example deductible $500 and guide citation. The amount regex treats evidence `$500,` differently from answer `$500`. |
| C01 | Pass | Abstains | Correct dwelling limit $265,000 and declaration citation. Missing lexical token `declared` triggers rejection. |
| C02 | Pass | Abstains | Correct claim-report policy identifier SYN-OTHER-009 and claim citation. Missing lexical token `printed` triggers rejection. |
| D01 | Pass | Abstains | Correct current water-backup limit $4,200; historical $4,600 is correctly identified as replaced. Both documents are in the prompt, but the support check examines one citation and rejects the historical amount. |
| D02 | Pass | Pass | Correct historical base limit $4,600 with base citation; question explicitly asks the historical version. |
| D03 | Pass | Abstains | Correct endorsement effective date April 1, 2026 and source. Evidence says `effective April 1, 2026`; it lacks the question token `date`. |
| E04 | Pass | Abstains | Correct version-A limit $3,900 without deciding which version controls. Rejected because `without`, `deciding`, and `printed` are absent from evidence. |
| F01, F02, F03 | Pass each | Pass each | Correct camera/laptop/jewelry item limit and corresponding physical page. The phrase “this limit ... does not cover” is scoped to the item limit, not an assertion that the whole policy excludes other items. |

| Unsupported case | Original raw abstention metric | Original served abstention | Manual interpretation of raw answer |
| --- | --- | --- | --- |
| B04 | Pass | Pass | Explicitly says annual premium is not stated; gives no invented premium. |
| C03 | Pass | Pass | Says no approved insurer payment is established, identifies absence of approval and policy-ID mismatch. |
| C04 | Pass | Pass | Says expiration date is not stated; supplies no guessed date. |
| D04 | Pass | Pass | Says earthquake limit is not established. Water-backup amounts appear only as contrasting supported information. |
| E01 | **Fail** | Pass | Says evidence “does not support a definitive answer” about the controlling equipment limit; lists the conflicting version amounts without choosing one. Phrase-matcher false negative. |
| E02 | Pass | Pass | Says controlling deductible cannot be determined; lists alternatives without resolving precedence. |
| E03 | **Fail** | Pass | Says evidence does not support an answer and it is “impossible to determine” which version was issued later. Phrase-matcher false negative. |
| F04 | **Fail** | Pass | Says “No blanket limit for all property is stated.” Phrase-matcher false negative. No aggregate is invented by the model. |

The manual columns supplement the fixed automated metrics; they do not overwrite or rescore the original summary. No expert or blinded rating was performed.

## Root causes

### 1. Citation parsing treats Markdown wrappers as part of a source

The parser's `SOURCE:\s*(.+)` match captures `** SCENARIO-B-DECLARATIONS.pdf#page=1 (Policy identifier ...)` for B01 and a trailing `*` for the B02 italic source. Exact matching against ranked sources then fails. B01's fallback chooses the guide because its explanatory words overlap the whole answer, despite the model's correct explicit declaration citation. Most other cases happen to fall back to the correct page, so passing examples do not validate this parser.

A safer parser should recognize known source identifiers inside Markdown/parentheses while preserving exact filename/page boundaries. An explicit unknown or conflicting source should produce a visible invalid-citation state; it should not silently be replaced by the most lexically similar page. Tests should include physical p1 versus p10, multiple cited pages, an unknown source, and a valid source whose filename is a prefix of another filename.

### 2. Currency regex includes punctuation commas

`\$[\d,]+` admits a trailing comma. B03's evidence has `$500, so ...`, producing token `$500,`; the answer produces `$500`. D01's historical `$4,600, but ...` similarly includes the trailing comma. Compare numeric values using a currency grammar with valid grouping and Decimal normalization, not arbitrary character-set matches. Keep $500, $5,000, and $500.50 distinct.

### 3. Every question word is treated as an evidence requirement

The subset test for `specific_terms` requires request wording such as `actually`, `declared`, `printed`, `without`, and `deciding` to appear literally in the cited text. It also requires `date` even when a document unambiguously says `effective April 1, 2026`. These are not missing answer facts. The raw amount/identifier/date and requested policy or version are present.

A generic correction should separate the requested field/entity from question scaffolding, normalize limited morphological variations, and handle date fields deliberately. Preserve meaningful discriminators: comprehensive versus collision, effective versus expiration date, camera versus laptop, policy identifiers, and explicit version A versus B. Simply relaxing the overlap threshold or deleting all remaining specific terms is unsafe.

### 4. One-citation support is insufficient for a multi-document explanation

B01 correctly contrasts actual and example deductibles; D01 contrasts the current endorsement and old base limit. Requiring all amounts from the entire answer to appear on one cited page rejects useful explanations. The additional amounts are visible in the saved context but should have claim-specific citations. Prefer a concise directly supported answer, or validate each comparison clause against its own explicit supplied source. A number appearing somewhere among retrieved pages is not enough: it could belong to another coverage, policy, or version.

### 5. Missed refusal wording can trigger an unsafe numeric repair

F04 is the most important negative control. Because “No blanket limit ... is stated” is not recognized as an explicit refusal, `answer_repaired=true`. A read-only reconstruction from the saved ranking shows `_repair_answer_from_evidence` proposes:

> The scheduled camera coverage limit is $8,000.

The later specific-term guard blocks that substitution, so the final served answer is safely empty. If that guard is relaxed without fixing refusal preservation and field matching, a correct model refusal could become a wrong blanket-limit answer. Recognize common information-absence phrasings before repair, and do not infer aggregate or controlling values from one unrelated item/version amount.

E01's “does not support a definitive answer,” E03's “impossible to determine,” and F04's “No [field] ... is stated” are concrete missing patterns. Phrase extensions should include negative controls for an otherwise supported answer that merely discusses a limitation; do not flag every sentence containing `no` or `not` as abstention.

## Regression checks recommended before a new run

1. Preserve all eight unsupported refusals, especially F04, through the full `query_structured` path after any term-guard changes.
2. Keep wrong coverage/entity values blocked even when the same number appears elsewhere in the packet.
3. Preserve explicit policy-ID mismatch; a repair estimate is not an approved insurer payment.
4. Do not choose a controlling amount from unsequenced conflicting versions. Still allow a factual question explicitly scoped to version A.
5. Reject unsupported extra numbers in an otherwise correct answer. Support comparison clauses only with appropriate cited context.
6. Keep the original run immutable and use a new run directory after changes. These visible synthetic cases are regression-driven development, not a new hidden test.

## Original artifact fingerprints

- `predictions.jsonl`: `8f98a415a41247ad7c53f5423df65e3778cb5f11fef99a254cd54fec2f7434d5`
- `summary.json`: `ad163a2e6395952a491e980617200851dcfba9ffead7d05cc0648a4209f1c450`
- `run_metadata.json`: `f72321aaf0da602ea8192494bf26713fe4ac5793a60a053a4491f2b561ad3eb9`

The immutable run metadata identifies the exact evaluated code/model/configuration. Source inspection during this review explains the saved behavior; concurrent later runtime fixes should be assessed in separate runs.

## Subsequent refusal-detector correction

After this review, narrowly scoped changes to `answer_safety.py` recognize the three missed refusal formulations. The new rules match a missing value/decision or an explicit inability to answer; they do not classify ordinary exclusions, known zero deductibles, or every occurrence of `no`, `not`, or `support` as refusal. Ten safety tests plus 28 subtests pass, including a full hybrid/legacy application-path check that a missing blanket-limit refusal never reaches numeric repair. Positive controls retain supported answers that mention exclusions or other limitations. A read-only check recognizes E01, E03, and F04 under the new detector. The original automated metric files above remain unchanged; any new run must record the changed code hash. The detector remains a wording heuristic without access to the question, not a semantic adjudicator.
