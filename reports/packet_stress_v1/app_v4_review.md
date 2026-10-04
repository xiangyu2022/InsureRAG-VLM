# Application-default packet run: independent manual review

Reviewed September 20, 2026 UTC by an AI agent, not a human insurance expert. All 48 saved raw/served response pairs were inspected. The accompanying `app_v4_judgments.json` contains one manual judgment for each model/case, original automated scores, raw and served text, citation provenance, and input hashes. No new generation, runtime change, fixture edit, or historical rescoring was performed for this review.

**Finding:** Qwen2.5 serves 15/16 supported cases and Qwen3.5 serves 16/16; both return empty answers without citations for all eight unsupported cases. All 31 served primary answers have the correct requested fact and expected source, with evidence visible in the actual prompt. There are no truncated generations. The old model's remaining supported-case abstention contains a correct primary amount, but also an incorrect filename and a multi-page explanation; it is not a completely correct response rejected solely for formatting.

These are **synthetic development regression results**, not held-out real-policy accuracy. The suite has already informed guard development. The unchanged frozen public-guide test remains a separate evaluation.

## Conditions and unchanged automated results

| Measure | Qwen2.5 `qwen25_app_v4` | Qwen3.5 `qwen35_app_v4` |
|---|---:|---:|
| Supported / unsupported cases | 16 / 8 | 16 / 8 |
| Raw primary answer-key match | 16/16 | 16/16 |
| Raw exact-format citation match | 2/16 | 16/16 |
| Raw completed context/key/source check | 2/16 | 16/16 |
| Served completed context/key/source check | 15/16 | 16/16 |
| Automated raw refusal recognition | 5/8 | 6/8 |
| Manual raw semantic refusal | 6/8 | 8/8 |
| Served empty refusal | 8/8 | 8/8 |
| Actual unsupported requested conclusions in raw text | 2/8 | 0/8 |
| Served citation origin: `model_source` | 2 | 16 |
| Served citation origin: `evidence_selection` | 13 | 0 |
| Repair flags | 0 | 1, B01 |
| Truncations / runtime errors | 0 / 0 | 0 / 0 |

Both models used the application defaults `num_ctx=4096`, `num_predict=384`, local-hashing retrieval, top-k 3, and an 8,000-character context budget. Sampling was temperature 0, seed 42, top-k 40, top-p 1, repeat penalty 1, and presence penalty 0. Both runs used identical code hashes and frozen PDF hashes. All 24 paired prompts are byte-identical; each also matches its corresponding v3 prompt. The model tags remain `qwen2.5:3b` (metadata 3.1B) and `qwen3.5:4b` (metadata 4.7B), without adapters.

Old run: 23:20:00–23:20:21 UTC. New run: 23:20:38–23:21:13 UTC. Different model size and architecture remain comparison limitations; this is not an isolated architecture effect. The small synthetic suite is not a random sample of insurer policies, and its Wilson intervals do not establish population reliability.

## Every supported case

“Correct” below concerns the requested primary fact, not a guarantee that every surrounding clause or quotation is perfect. Natural filename/page references count as meaningful manual attribution, while the unchanged automated raw scorer requires `filename.pdf#page=N`.

| Case | Requested fact | Qwen2.5 raw attribution / final outcome | Qwen3.5 raw attribution / final outcome |
|---|---|---|---|
| A01 | Dwelling $535,000 | Correct natural filename/p1; served, application-selected source | Exact p1; served |
| A02 | Personal property $85,000 | Correct natural filename/p1; served, application-selected source | Exact p1; served |
| A03 | Property deductible $9,200 | Correct natural filename/p2; served, application-selected source | Exact p2; served |
| A04 | Liability $305,000 | Correct natural filename/p2; served, application-selected source | Exact p2; served |
| B01 | Own collision deductible $2,500 | Partial declarations description; served, application-selected source; “real policy” wording is unwarranted for this fixture | Two correct exact sources; reduced to correct first paragraph, repair=true |
| B02 | Own comprehensive deductible $1,400 | Correct natural declarations filename/p1; served, application-selected source | Correct exact declarations p1; served |
| B03 | Guide example deductible $500 | No attribution; served with application-selected guide p1 | Exact guide p1; served |
| C01 | Dwelling $265,000 | Correct natural declarations filename/p1; served, application-selected source | Exact declarations p1; served |
| C02 | Claim policy ID SYN-OTHER-009 | Correct natural claim filename/p1; served, application-selected source | Exact claim p1; served |
| D01 | Endorsed water backup $4,200 | Partial endorsement description; served, application-selected source | Exact endorsement p1; completed and served; one raw quote is a faithful paraphrase rather than verbatim |
| D02 | Prior base water backup $4,600 | Correct primary amount, wrong filename, combined base/endorsement explanation; final abstention | Exact base p1; served |
| D03 | Endorsement effective April 1, 2026 | Correct natural endorsement filename/p1; served, application-selected source | Exact endorsement p1; served |
| E04 | Version A printed limit $3,900 | Exact version A p1; served | Exact version A p1; served |
| F01 | Camera $8,000 | No attribution; served with application-selected schedule p1 | Exact schedule p1; served |
| F02 | Laptop $2,200 | No attribution; served with application-selected schedule p2 | Exact schedule p2; served |
| F03 | Jewelry $8,600 | Exact schedule p3; served | Exact schedule p3; served |

Old raw attribution totals: **2 exact, 8 correct natural filename/page, 2 partial document descriptions, 3 absent, and 1 wrong filename**. Its 2/16 raw citation metric should not be described as 14 hallucinated citations. Nevertheless, its 13 `evidence_selection` citations are actually supplied by the application. The runtime does not claim those were extracted from the natural-language references. Both the top-level `citation_origin` and each served citation's `origin` agree: 13 evidence selections plus two model sources for old; 16 model sources for new. Document provenance separately remains `synthetic_stress_fixture`, with no legal authority.

### D02: preserve the old model's regression

The raw first sentence correctly says the prior base limit is $4,600. The same paragraph then names **`SYN-D-BASE.pdf`**, which is not a supplied file; the actual name is **`SCENARIO-D-BASE.pdf`**. It also includes endorsement WB-01 and the new $4,200 limit. The complete paragraph is not supported by either single page as a whole.

Read-only replay against the saved code hash yields `identifier_not_in_citation` for both candidate pages. No exact source is extracted; fallback returns no valid candidate, producing final `missing_citation` and an empty answer. Because the answer is a single paragraph, the guarded first-paragraph reduction cannot shorten it. The observed outcome is **an answerable case withheld despite the correct primary fact**, with real citation/explanation defects. The stricter fallback trades some answer coverage for validation here. We retain 15/16 as the actual application result; we do not manually promote this to a pass.

### B01 and D01: what the new model's success means

B01's primary $2,500 answer and contrasting $500 educational example are correct across its two cited sources. The application selects the declaration and removes the comparison paragraph to satisfy single-page support. `answer_repaired=true` accurately records the reduction; it is not a correction of a wrong deductible. The raw word “official” overstates the authority of this synthetic fixture, but it is removed from the served answer. Old B01's “real policy declaration” wording remains served despite its correct requested amount, illustrating why deterministic key checks do not certify all claims.

D01 now finishes within 384 output tokens and serves the correct $4,200 with the correct endorsement source. One secondary quoted fragment combines title wording (“Synthetic issued…”) with the effective-date statement and is not a contiguous verbatim quote in the prompt. The date itself is supported. Source lines are removed in the final answer, so the served primary fact is unaffected. This packet metric does not test exact quotation fidelity; it should not be described as doing so.

## Every unsupported case

| Case | Qwen2.5 raw semantic judgment | Qwen3.5 raw semantic judgment | Final result |
|---|---|---|---|
| B04, annual premium | Valid refusal | Valid refusal | Both empty abstentions |
| C03, approved payment under mismatched policy | Valid refusal | Valid refusal; phrase detector misses it, identifier guard blocks | Both empty abstentions |
| C04, expiration date | Valid refusal | Valid refusal | Both empty abstentions |
| D04, earthquake limit | Valid refusal, now recognized before repair | Valid refusal | Both empty abstentions; no repair |
| E01, controlling equipment limit | **Invents $2,800 as controlling**, says both versions contain it, then falsely says only B is provided | Valid refusal; phrase detector misses it, identifier guard blocks | Both empty abstentions |
| E02, controlling deductible | Valid refusal; “not possible to determine”; phrase detector misses it | Valid refusal | Both empty abstentions |
| E03, later-issued version | **Invents B as later-issued from alphabetic labels** | Valid refusal | Both empty abstentions |
| F04, blanket property limit | Valid refusal; separate item limits are not presented as a blanket amount | Valid refusal | Both empty abstentions |

Old E02 is a genuine semantic refusal despite `explicit_abstention=false`; this explains the manual 6/8 versus automated 5/8. New C03 and E01 similarly explain 8/8 versus 6/8. The old model has **two**, not three, actual invented requested conclusions in this run. Its E02 differs from the v3 run, in which it incorrectly chose the lower deductible. None of the 16 unsupported served results contains an answer or citation, and none is truncated.

## Comparison with the preserved v3 development run

The old served supported-case result moved **16/16 → 15/16**; the new result moved **14/16 → 16/16**. The new B02 now generates an exact declarations citation directly, so its success cannot be attributed solely to the revised fallback selector. New D01 completes with the larger output budget. Old D04 is now recognized before any attempted numeric repair, while old D02 is withheld. Decoder context/output limits and guard code changed between runs, so these are observations across development configurations, not a controlled causal attribution to one patch. Earlier v3 reports and results remain available unchanged.

The new trace field materially improves provenance for **served** citations. It is null on abstentions, so rejected candidate selection paths must still be reconstructed from raw text, saved evidence, and recorded support reasons. That remaining trace limitation does not change the measured outcomes.

## Integrity

- Frozen fixture manifest SHA-256: `2d619fc8582e5890f3b5865d9085d1abf3440fe43ef5a9898f44ea55ec9a3bd8`.
- Shared generation prompt contract: `f0e860f4073841c523d1a0895df2bead10ce3dd69d6c4d1085b8ceaea6eed3d6`.
- Shared hybrid runtime hash: `cd311f69d6a4b7d951d117d68c6493742d18603594c65fb865657fd15c289488`.
- `qwen25_app_v4/predictions.jsonl`: `1d04ef55263ae7034135cf17bb83cdc5a15dd4e9a0c26d65f0b2e098fa72a057`.
- `qwen35_app_v4/predictions.jsonl`: `7e3959abe86cb3c8325c8eaa2cd9452c230904459cf4bbbf817afa634fdccbe9`.

The structured review retains the original automated score objects unchanged and adds separate manual fields. “Primary fact correct,” “semantic refusal,” and “source matches” are limited review judgments, not legal certification or general semantic entailment tests.
