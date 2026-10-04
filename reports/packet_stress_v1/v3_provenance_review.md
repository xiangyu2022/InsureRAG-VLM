# Packet stress v3: manual provenance and refusal review

Review date: 2026-09-20 UTC. Reviewer: AI-assisted/manual-agent review, not a human insurance expert. This is a review of 48 saved responses (24 cases per model) against the frozen, AI-authored synthetic PDFs. It is not real-policy accuracy, legal validation, or an independent held-out benchmark. No model was called and no fixture, historical prediction, automated score, or runtime file was changed for this review.

## Main findings

The old model's **3/16 raw citation score substantially reflects citation formatting**, rather than 13 wrong or missing attributions. Eight additional answers identify the correct filename and physical page in natural prose. Four contain a correct but incomplete document reference, and one has no attribution. The application selects the correct final page for all 16 supported answers. However, **13 of those 16 structured citations are selected by the application**, not extracted as a complete source identifier from the raw answer. `answer_repaired=false` does not distinguish that operation.

Qwen3.5 produces complete source identifiers in 15/16 supported answers, but serves 14/16. B02 contains the correct answer and names the correct declarations file, yet automatic citation selection picks its comparative guide; the guard then abstains. D01 gives the correct primary answer but reaches the 192-token limit while quoting evidence; the truncation guard correctly suppresses an incomplete generation. The saved results do not establish universal application-level superiority of the newer model.

On the eight unsupported questions, manual review finds **five valid raw refusals and three unsupported precedence claims for Qwen2.5**, versus **eight valid raw refusals for Qwen3.5**. The automated raw refusal detector reports 4/8 and 6/8 respectively because it misses several phrasings. Both applications return empty answers and no citations on all eight unsupported cases. The old model's three unsupported claims are actually blocked; the other raw-to-served changes mainly normalize refusals.

The single `answer_repaired=true` event means different things in the two runs. Old D04 is an attempted repair of a missed refusal into an unrelated water-backup amount, subsequently blocked. New B01 is a successful reduction of a correct multi-source explanation to its correct first paragraph. Neither count measures the number of automatically selected citations.

## Frozen inputs and comparability

| Item | Qwen2.5 v3 | Qwen3.5 v3 |
|---|---|---|
| Run | `qwen25_v3` | `qwen35_v3` |
| Recorded time, UTC | 23:00:25–23:00:50 | 23:02:12–23:02:52 |
| Model | `qwen2.5:3b`, metadata 3.1B, Q4_K_M | `qwen3.5:4b`, metadata 4.7B, Q4_K_M |
| Raw key presence | 16/16 | 16/16 |
| Raw exact-format citation match | 3/16 | 15/16 |
| Raw completed context/key/source check | 3/16 | 14/16 |
| Served context/key/source check | 16/16 | 14/16 |
| Automated raw refusal | 4/8 | 6/8 |
| Manual semantic refusal | 5/8 | 8/8 |
| Served empty refusal | 8/8 | 8/8 |
| Repair flag | 1: D04, ultimately blocked | 1: B01, served |

All 24 paired prompts are byte-identical. Both runs use the same recorded runtime code hashes, PDF hashes, local-hashing retrieval, top-k 3, 8,000 context characters, 8,192 context tokens, 192 output tokens, temperature 0, seed 42, top-k sampling 40, top-p 1, repeat penalty 1, and presence penalty 0. Neither model has an SFT adapter. The different parameter counts and architectures confound a pure generation-version comparison. The synthetic cases share six authored packets and have already informed application development; descriptive intervals are not population accuracy guarantees.

The raw citation scorer searches for `filename.pdf#page=N`; it **does not require a literal `SOURCE:` prefix**. Consequently, `[SCENARIO-F-SCHEDULE.pdf#page=3]` passes, while “physical page 1 of the document SCENARIO-A.pdf” fails. Some unsupported responses also have Markdown or parenthesis wrappers that the simple raw canonicalizer retains, but their citation fields do not contribute to unsupported-abstention success.

## Every supported case: raw attribution and final disposition

All requested primary facts are correct in both models' raw text. This does not certify every surrounding statement: old B01 calls the invented declaration “a real policy declaration,” and new B01 calls it “official”; neither authority characterization is warranted for the synthetic fixture. New D01 is an incomplete generation despite containing the correct primary fact.

Legend: **exact** = complete correct `filename.pdf#page=N` present; **natural** = correct filename plus physical page in prose; **partial** = correct document description/name but no complete filename-and-page reference; **none** = no attribution. “Automatic” means replay of the saved runtime's source-selection helpers, not an inference that the model supplied an exact source identifier.

| Case | Requested primary fact | Qwen2.5 raw attribution | Qwen2.5 application disposition | Qwen3.5 raw attribution | Qwen3.5 application disposition |
|---|---|---|---|---|---|
| A01 | Dwelling $535,000 | Natural: A.pdf, physical p1 | Automatic p1; text unchanged | Exact A p1 | Extracted p1; source line removed |
| A02 | Personal property $85,000 | Natural: A.pdf, physical p1 | Automatic p1; text unchanged | Exact A p1 | Extracted p1; source line removed |
| A03 | Property deductible $9,200 | Natural: A.pdf, physical p2 | Automatic p2; text unchanged | Exact A p2 | Extracted p2; source line removed |
| A04 | Liability $305,000 | Natural: A.pdf, physical p2 | Automatic p2; text unchanged | Exact A p2 | Extracted p2; source line removed |
| B01 | Own collision deductible $2,500 | Partial: synthetic declarations title/policy ID; no page | Automatic declarations p1; text unchanged | Exact declarations p1 and guide p1 | First source selected; first-paragraph reduction, repaired=true |
| B02 | Own comprehensive deductible $1,400 | Partial: declarations title and p1, no filename | Automatic declarations p1; text unchanged | Partial: declarations filename; comparative guide filename, neither page | Automatic selection chooses guide p1; blocked by identifier check |
| B03 | Educational example deductible $500 | Exact guide p1 | Extracted p1; source line removed, repaired=false | Exact guide p1 | Extracted p1; source line removed |
| C01 | Dwelling $265,000 | Natural: declarations filename and physical p1 | Automatic declarations p1; text unchanged | Exact declarations p1 | Extracted p1; source line removed |
| C02 | Claim policy ID SYN-OTHER-009 | Natural: claim filename and physical p1 | Automatic claim p1; text unchanged | Exact claim p1 | Extracted p1; source line removed |
| D01 | Endorsed water backup $4,200 | Partial: endorsement filename, no page number | Automatic endorsement p1; text unchanged | Exact endorsement p1, then incomplete quote | Blocked because generation truncated |
| D02 | Prior base water backup $4,600 | Partial: base filename, no page number | Automatic base p1; text unchanged | Exact base p1 | Extracted p1; source line removed |
| D03 | Endorsement effective April 1, 2026 | Natural: endorsement filename, first physical page | Automatic endorsement p1; text unchanged | Exact endorsement p1 | Extracted p1; source line removed |
| E04 | Version A printed limit $3,900 | Exact version A p1, inline | Extracted p1; text unchanged | Exact version A p1 | Extracted p1; source line removed |
| F01 | Camera $8,000 | Natural: schedule filename and physical p1 | Automatic schedule p1; text unchanged | Exact schedule p1 | Extracted p1; source line removed |
| F02 | Laptop $2,200 | None | Automatic schedule p2; text unchanged | Exact schedule p2 | Extracted p2; source line removed |
| F03 | Jewelry $8,600 | Exact schedule p3 in brackets | Extracted p3; text unchanged | Exact schedule p3 | Extracted p3; source line removed |

Old raw attribution totals: exact 3, natural filename/page 8, partial 4, none 1. No manually observed wrong filename/page attribution among these 16 supported raw answers. Thus 11/16 have a complete manually recognizable page attribution, versus the unchanged automated 3/16 exact-format score. The remaining partial references are not silently counted as complete page citations. All 16 old served citations point to the expected page and supplied evidence. All 14 new served citations do as well.

## Every unsupported case: semantic assessment and actual guard

| Case | Missing or unresolved fact | Qwen2.5 raw judgment | Qwen2.5 recorded support reason | Qwen3.5 raw judgment | Qwen3.5 recorded support reason |
|---|---|---|---|---|---|
| B04 | Annual premium | Valid refusal; lists actual deductibles but explicitly says premium cannot be determined | `model_reported_insufficient_evidence` | Valid refusal; premium not stated | `model_reported_insufficient_evidence` |
| C03 | Approved payment under mismatched policy ID | Valid refusal; distinguishes claim and declarations and no approved payment | `model_reported_insufficient_evidence` | Valid refusal; “This packet does not establish an approved insurer payment”; two correct source IDs | `identifier_not_in_citation`; refusal heuristic missed wording |
| C04 | Expiration date | Valid refusal; no date supplied | `model_reported_insufficient_evidence` | Valid refusal; no date supplied | `model_reported_insufficient_evidence` |
| D04 | Earthquake limit | Valid refusal: “there is no information about earthquake coverage limits”; heuristic missed; attempted repair introduces water-backup amount | `specific_question_terms_not_in_citation` after repair | Valid refusal; only water-backup amounts supplied | `model_reported_insufficient_evidence` |
| E01 | Controlling equipment limit | Unsupported claim: $2,800 controls; falsely says both versions contain it and version B is the only specific limit | `controlling_version_not_established` | Valid refusal; neither $3,900 nor $2,800 established as controlling | `identifier_not_in_citation`; refusal heuristic missed wording |
| E02 | Controlling equipment deductible | Unsupported claim: version A controls because its $2,400 deductible is lower than $9,700 | `identifier_not_in_citation` | Valid refusal; neither deductible confirmed active | `model_reported_insufficient_evidence` |
| E03 | Which version issued later | Unsupported claim: B issued later merely because labels are A/B | `controlling_version_not_established` | Valid refusal; no dates or precedence supplied | `model_reported_insufficient_evidence` |
| F04 | Blanket property limit | Valid refusal; lists separate item limits without adding them into a blanket value | `model_reported_insufficient_evidence` | Valid refusal; same distinction | `model_reported_insufficient_evidence` |

All 16 unsupported final results contain `answer=""`, `abstain=true`, and `citations=[]`. None of these model generations is truncated. The application prevented three actual unsupported precedence conclusions from the old model, not four: D04 was already a semantic refusal. Similarly, the new model's automated 6/8 raw score does not imply two hallucinations. C03 and E01 correctly decline to establish the requested facts.

## Repair and citation-selection trace

The saved runtime code matches SHA-256 `19e7ca89b5ac79648d1539dc97d9fc3caf795bf14bdafb3d310d9700e61d800a` for `src/insurerag_vlm/hybrid_pipeline.py`. Read-only replay of its deterministic helpers explains these paths without calling a model:

- `_extract_answer_source` looks for complete known source identifiers, preferring the first textual match. If an explicit source string is unknown, it preserves that unresolved string and does not silently replace it.
- When no exact source is extracted, `query_structured` calls `_choose_cited_page`, which scores answer/question word overlap, money overlap, and retrieval score. It does not parse natural-language “physical page” references. Old A01's correct natural citation therefore reaches the same automatic-selection path as uncited F02.
- `query_structured` removes standalone `Source:` presentation lines without marking a repair. Old B03 and most new supported answers consequently have different raw/final text with `answer_repaired=false`; that is presentation normalization, not a factual correction.
- Old D04's raw refusal is not recognized. The selected base page permits `_repair_answer_from_evidence` to produce **“Water backup coverage limit is $4,600.”** The later question-specific guard rejects this unrelated earthquake answer. The attempted intermediate text is not saved in the result; the exact value above is a deterministic replay, not a recorded model response. `answer_repaired=true` remains true even though the final answer is empty.
- New B01 cites both the declaration and its contrasting guide. Only the declaration is selected for structured support. The guide's $500/$2,900 comparison is therefore outside that single citation; the application keeps the independently supported first paragraph and marks a repair. The original explanation's factual distinction is correct across its two raw citations. Calling this a hallucination correction would overstate what happened.
- New B02 has a similar correct contrast but no exact source ID. Overlap selection chooses the guide, whose evidence lacks SYN-B-001, causing `identifier_not_in_citation`. The primary $1,400 answer is correct; the failure concerns source selection and application abstention.
- New D01 contains correct $4,200 and correct source ID, but `done_reason=length`/`generation_truncated=true` prevents serving it. Raw key/source checks that do not require completed generation can pass this record; the completed context check and served output correctly fail closed.

## Recommended provenance fields, not implemented here

An explicit citation-origin trace is warranted. Existing `source_origin` describes the *document's* provenance (`synthetic_stress_fixture`); it does not explain who selected the citation. A separate field such as `citation_origin` should distinguish `model_exact_source`, `automatic_evidence_match`, `unresolved_explicit_source`, and `none`. A future `model_natural_language_resolved` mode should be used only after an actual resolver matches a model-supplied filename and page, not retroactively applied to these overlap-selected citations.

For each selected citation, preserve the raw source mention (if any), selected source, selection method, and whether it was ultimately served. Record automatic selection even if the model's natural-language attribution happened to agree. Retain every model-cited source separately from the single selected application citation. This would expose both old F02's newly supplied attribution and new B02's incorrect intermediate selection.

Separate `answer_repair_attempted`, `answer_repair_kind`, and `answer_repair_served` would also prevent the old D04 flag from being mistaken for a successful correction. A presentation-normalization field can record removal of source lines without conflating it with factual repair. These are trace recommendations only; historical outputs and scores remain unchanged.

Future development-only regression checks could target semantically clear “there is no information about [requested field]” refusals before repair, and unambiguous filename/page resolution before overlap fallback, while retaining negative controls for answered questions that merely mention another limitation. Any such change should be validated prospectively and must not be used to rewrite this frozen run.

## Integrity record

- Frozen packet manifest SHA-256: `2d619fc8582e5890f3b5865d9085d1abf3440fe43ef5a9898f44ea55ec9a3bd8`.
- `qwen25_v3/predictions.jsonl`: `8d87771c784e2aa5cf3119db6a5ba2de7f30668b0e052ab25d9099179911dad4`.
- `qwen35_v3/predictions.jsonl`: `ca791210c3aee2dffd665cd28fd1fd6933355ddb21335acb6766962b10ed91f2`.
- Review sources: both runs' predictions, prompts, runtime metadata and summaries; `data/benchmarks/packet_stress_v1/packets.json`; `scripts/eval_packet_stress.py`; and the matching runtime source's `query_structured`, `_extract_answer_source`, `_choose_cited_page`, `_repair_answer_from_evidence`, and `_citation_support_details`.

Manual judgments are supplementary and do not replace the published automated metrics. The review concerns primary facts, refusal meaning, and citation provenance in a small synthetic suite; lexical key presence and page selection do not certify general semantic entailment.
