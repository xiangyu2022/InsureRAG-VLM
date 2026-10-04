# Frozen benchmark second-pass annotation review

Review completed 2026-09-20 at approximately 22:16 UTC, before test-model inference. Reviewer: the AI coding agent that authored the fixtures, performing a second pass against the archived page text. This is **not an independent human review or insurance-expert certification**. Parent-agent review provides another code/fixture check, not domain adjudication.

The fixed lock SHA256 remains `e97695307070c6d1974c56644c31cc3b6df8227aa989f370231e628b9f635f82`. No frozen question, answer key, span, document list, or corpus file was changed during this review.

## Scope and findings

All 24 supported references are established by the contents of their cited archived pages. Exact span inclusion and reference-key matching also pass programmatic validation. The 12 unsupported questions ask for personal policy or private identity values not supplied by these public guides. The ordinary missing-value cases are more relevant than the obviously unrelated identity questions, but both are simple abstention probes. They do not measure difficult adversarial abstention.

| Split/document | Available curated pages | Supported facts checked | Finding |
| --- | ---: | --- | --- |
| Dev, Delaware auto | 12 | First-offense minimum fine $1,500; PIP funeral maximum $5,000; uninsured property-damage deductible $250; initial course discount 10% for three years | Each cited page establishes the reference. The course discount is conditional on all drivers completing an approved course and applies to a portion of insurance; a broader model claim should be flagged in semantic review. |
| Dev, Delaware homeowners | 11 | Renters HO-4; typical computer category limit $5,000; worked TV replacement-cost amount $500; typical hurricane deductible 2% | Each reference is established. These are guide examples/typical terms, not an individual's policy. HO-4 appears on physical p6; the p13 renters discussion alone does not name the form. |
| Test, Maryland auto | 46 | Credit-based maximum adjustment 40%; property damage liability minimum $15,000; minimum offered PIP $2,500; worked collision payment $700 | Each reference is established. Credit adjustment gold span must be read with preceding negation; the page says an insurer may not exceed 40%. |
| Test, Maryland homeowners | 36 | HO-3 building open-peril/contents named-peril; deductible exemptions liability and medical payments; worked hurricane payment $1,000; flood contents purchased separately | Each reference is established. HO-3 gold span covers contents only, while the immediately preceding sentence supports the building answer; see limitation below. |
| Test, North Carolina disability | 5 | Elimination period; income formula range 50–75%; short-term benefit duration six months to two years; residual disability rider | Each reference is established. The guide distinguishes elimination from waiting periods; do not automatically treat them as interchangeable. |
| Test, North Carolina travel | 3 | First seek reimbursement from trip provider; baggage insurance; cancellation waivers are not insurance; weather-claim condition mandatory evacuation orders | Each reference is established as archived guide content. Free-look discussion supplies no number of days, so the personal-duration question is unsupported. |

The selected scopes contain 113 of the 201 curated pages. The 201-page number describes the source corpus, not the retrieval candidate count per question. The North Carolina travel scope has only three pages, so retrieval hit@3 there is guaranteed if all pages are returned. Report per-document results and this easy-case limitation alongside pooled retrieval metrics.

## Annotation and scoring limitations fixed before test

1. `test_md_homeowners_insurance_guide_01` requires building **open-peril** and contents **named-peril**. Its exact gold span covers only the contents clause; the full cited page contains both. The lexical answer key also accepts a response that reverses the attributes. Preserve the frozen metric and separately inspect assignment of attributes. A future benchmark version should include both clauses and structured answer slots.
2. Numeric keys check equal values with boundaries, not semantic roles. An answer containing a correct number plus contradictory qualifications could pass. The two-answer duration question can omit eligibility conditions while passing. Review entire answers and quoted context.
3. Exact citation and full-span checks can reject legitimate alternative evidence, short sufficient quotes, or paraphrases. For example, the dev hurricane response quoting only “typically 2% of the property's value” is sufficient for the asked percentage but fails the longer frozen span. Keep original metric values and label such cases as contract false negatives, not wrong insurance answers.
4. Some spans are fragments: the Maryland credit maximum requires the preceding “may not” context; the flood-content span requires its preceding subject. Full pages are supplied in oracle mode, and retrieved prompts are saved for inspection. A verbatim fragment alone is not sufficient proof of semantic grounding.
5. Archived law/coverage statements are evaluated solely as document contents. The benchmark makes no assertion that these statements describe current law or universally applicable policy terms.
6. No original PDF rendering was inspected in this second pass; the review uses the committed extracted page text and recorded physical-page citations. Layout/OCR fidelity is outside this text benchmark.

## Corpus/training provenance

The current committed `data/04_curated/sft_dataset.jsonl` contains record references to three test documents: Maryland auto (368 rows mentioning its document identifier), Maryland homeowners (355), and North Carolina disability (62). No identifier matches were found for the two Delaware documents or North Carolina travel. These are row-presence counts, not proof that every row reached a trained adapter. Therefore the benchmark is document-disjoint **between its dev and test splits**, but is not a holdout from the repository's historical SFT source data. The comparison uses unadapted local Ollama models; unknown public pretraining exposure remains possible.

The frozen Delaware auto provenance entry points to the official consumer landing page. The repository's existing source registry also identifies the exact PDF as `https://insurance.delaware.gov/wp-content/uploads/sites/15/2022/09/Auto-Insurance-Guide.pdf`. This supplementary note preserves the frozen document-list hash. Other frozen entries already identify specific official document URLs. Current URL contents may change; archived corpus hashes define the experiment.

## Post-inference inspection protocol

Read every answer, including deterministic passes. For each case record: whether the gold page and necessary span reached the prompt; whether all requested answer components are supported; whether the answer assigns numbers/attributes to the correct item; whether quoted text is faithful and sufficient; whether the citation points to that evidence; whether a guide example was mistaken for the user's own policy; and whether output was truncated or malformed. Distinguish retrieval miss, context-packing loss, semantic wrong/incomplete answer, citation/quote mismatch, strict-format-only failure, and adequate abstention.

Any manual verdict is AI-assisted and exploratory. Preserve raw results and frozen deterministic scores. Do not relabel cases or adjust keys using test outcomes. Report annotation weaknesses separately and introduce a new benchmark version for later corrections.
