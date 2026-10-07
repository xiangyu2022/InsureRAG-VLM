# Independent insurance holdout: construction in progress

This directory records dataset construction, **not a completed 1,000-question
benchmark or a model-quality improvement result**. No final benchmark answer
inference has run. Earlier exposed holdouts remain regression-only.

`protocol.json` fixes the 1,000-test minimum and a source-separated dev minimum
of 50 (150 preferred), task coverage, source balance,
review requirements and freeze conditions. `stage1_progress.json` and later
numbered progress files are timestamped checkpoints; do not sum their counts.
Source-original FAQs, authored hypothetical cases, advisory model decisions and
Codex content-review decisions are recorded separately. None is expert or human
adjudication. Provisional per-item clearance is not dataset acceptance.

The latest item-construction checkpoint is stage103_progress.json: **669 provisional
TEST and 58 DEV records across 145 TEST document families**. Source-only reviews
continue through stage106. **Accepted counts remain zero; the dataset is not frozen.**
All 727 retained item gates pass. Another 331 independent TEST questions are needed.
All four TEST type targets remain unmet; DEV minimums are provisionally met.
Earlier reports are historical snapshots and must not be added together.

Stages94–106 reviewed 49 drafts, retained 38 TEST and held 11, with no old-item
revocations. TEST contains 430 ordinary, 94 numerical, 89 multi-evidence and 56
insufficient-evidence tasks. Remaining targets are 170/56/61/44 respectively.
DEV remains 25/10/12/11. The unchanged limits are 20 items per document family
and 20% per publisher; publisher and family split isolation both pass.

Cached TRICARE, CDI, FCAC and OPM sources supply distinct consumer coverage,
cost, enrollment and claim needs. New Zealand MBIE Consumer Protection original
Crown explanatory text was admitted under reviewed CC BY-NC4.0 conditions,
attribution and disclaimer requirements. Its five guides/directory share one
conservative family with 17 retained items; the whole publisher was assigned TEST
before authoring. Active immutable snapshot18 preserves earlier document and FAQ
bytes. Images, logos and third-party material were excluded. Source text and
individual questions remain local, including required attribution metadata.

Numerical work requires retrieving operative rules/rates and independent arithmetic;
the OPM rate example explicitly uses the October2021 table and verified HTML
columns. Two nominal multi-evidence drafts were classified ordinary because an
extra span was unnecessary. Refusals preserve missing case-specific information.
New components of held composite questions and a reformulation of a previously
revoked numerical core were held, rather than reintroduced as new independent items.
Candidate/history neighbors and changed active relationships received sequential
Codex content review, not human/expert adjudication. All original15 revocations,
later scope/family-cap exclusions, same-ID repairs, CA correction and TDI DEV
assignment remain; 88 ledger links and current vector keys/hashes were verified.

Source-only reviews do not increase the question count. Additional NHC sources
remain unadmitted because of family overlap and timing ambiguity. APRA boundary
holds remain. HIA territorial reuse applicability remains unresolved; HKIA403 was
respected. ASIC's prior permission requirement and PBGC's explicit written-
permission requirement block their use. No alternate retrieval or approval followed.
The NY surprise-bill cache has further possible exceptions requiring full item and
history review; no new questions from it were counted.

49 new vectors used four sequential local model processes and five small forward
batches; old question/history vectors were reused. Local source checks used 26
bounded HTTP requests and 1,364,565 response bytes, plus seven separately accounted
browser calls. One NHC URL guard denial occurred before any HTTP request.
There were no paid model APIs, subagents, model downloads, training, final QA
inference or test-directed tuning. Offline CPU verification passed 480 tests and
58 subtests. Global count/type gates still block freezing and final evaluation.

Prior published commit118b579 passed all five CI jobs in run37674318205. This
checkpoint's exact-commit terminal CI receipt is recorded separately. PR8 remains
draft/unmerged; its description API403 restriction remains binding.

Review ledgers are deltas: absence from the latest ledger does not release an
earlier hold. Follow their hash-linked ancestry and preserve exclusions unless an
explicit repair and completed re-audit resolve them. The stage10 local record
binds all inherited decisions and the three unchanged repaired records; it adds
no new clearance and makes no change to the pool or source registries.

Source access/reuse decisions are in `source_status.json` and
`source_approvals.json`. An approval permits the stated local research scope,
not blanket redistribution or acceptance of every document. All source bodies,
question/answer/evidence text, model responses and full audit ledgers stay in the
ignored `local/` directory. Previously denied local paths and blocked sources
remain excluded.

## Reproducible stages

1. Preflight access and rights before acquisition. Preserve response hashes and
   stop on access denials; never change hosts or methods to bypass them.
2. `extract_holdout1000_candidates.py` writes a fresh immutable source snapshot.
   `build_holdout1000_evidence_catalog.py` binds paragraphs, short rule lists and
   complete tables (including VA custom tables) to exact source offsets.
3. `screen_holdout1000_history.py` checks normalized historical text and source
   URLs. `audit_holdout1000_near_duplicates.py` screens questions and evidence
   against accessible historical strings and all prior exposed source FAQs.
   Its character-cosine approximation is deliberately conservative.
   `audit_holdout1000_containment.py` additionally scans every contiguous
   normalized evidence window inside long historical records. Default evidence
   mode requires `--documents` and source-span metadata; `--text-mode` is
   explicitly unverified generic screening. Shared windows
   require review and absence of matches does not prove independence.
4. `audit_holdout1000_semantic_questions.py` uses a frozen local BGE model and
   verifies cached history/model hashes. It reports nearest historical and
   cross-document candidates, including neighbors below the flag threshold.
   Similarity scores are review aids, not proof of independence.
5. `triage_holdout1000_candidates.py` preserves document/split/duplicate groups,
   history exclusions and unresolved source defects, including the hashed
   `document_holds.json` registry and hash-bound `document_families.json`.
   Its output is a review
   queue, never accepted data. Context repairs and authored cases need their
   own evidence/history/semantic audit after revision.
6. Record individual content, source-currency, authorship and duplicate
   dispositions. A critical source error holds the affected stratum pending
   full review. Enforce corpus-wide refusal gaps and necessary, nonredundant
   evidence for multi-evidence tasks. Recompute numeric formulas with the
   restricted Decimal evaluator and check the applicability of their rules.
   Check scenario ages/dates against a rule's effective date and transitional
   provisions; correct arithmetic cannot repair an inapplicable rate.
   A question asking whether a public rule guarantees an outcome can be ordinary
   QA even when the actual private outcome is unknown. Refusal tasks must ask for
   that unresolved outcome. For multi-evidence tasks, remove each claimed
   necessary span in turn: if another span already settles the operative answer,
   hold the task classification.
   A numerical item must require an identified rule or rate to be retrieved;
   questions supplying all rules and inputs are held as arithmetic-only.
7. `holdout_quality.verify_dataset` must pass all item, split, coverage and
   review gates with explicit source registries before freezing
   corpus/manifests/ledger/configuration hashes
   and running the fixed final benchmark. No test-directed tuning.

`review_holdout1000_candidates_local.py` produces advisory source-FAQ screening
only. `draft_holdout1000_complex_cases.py` is experimental: two small local-model
pilots produced no accepted cases and bulk drafting was stopped after unsupported
claims, scope mistakes and arithmetic failures. Do not treat either script's
output as gold labels or model benchmark results.

The audit covers accessible project history, not foundation-model pretraining.
Source prose can be obsolete or internally inconsistent even on official sites.
Outstanding source conflicts, licensing uncertainty or review gaps are grounds
to hold a candidate, not to lower the protocol targets.

The stage4 review found substantive historical overlap below automatic
similarity thresholds, including evidence contained inside a much longer
historical record. A prior provisional item was revoked after checking that
passage. Whole-record cosine and question similarity must be supplemented by
passage containment and source-family review; zero flags do not prove novelty.
Invisible format characters inside source words also require repaired display
text and audits of both original and repaired evidence, while preserving raw
source offsets. The historical normalization/cache has not silently changed.

The original12-24hour construction estimate is superseded by the estimate in
`protocol.json`. The current review queue alone is smaller than the test target;
additional suitable sources and substantive information needs remain necessary.
