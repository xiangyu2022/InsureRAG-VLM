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

The latest checkpoint is [stage210](stage210_progress.json): **854 provisional
TEST and 58 DEV across 170 TEST families**. **Accepted counts remain zero; the
dataset is not frozen.** Another 146 independent TEST are needed: 65 ordinary,
26 numeric, 31 multi-evidence and 24 refusal questions.

Stage206 completed source-only review of nine cached pages. Seven whole pages
were excluded for non-insurance content or explicit third-party reproduction
restrictions; two mixed-scope pages remain pending. No question inputs were loaded
for these decisions. Existing chunks and DEV support mappings remain unchanged.
Exact raw and normalized alias checks found no pending sources eligible for direct
review reuse. No new questions or model vectors were created by this corpus pass.

Four additional source preflights remain deferred: an unreviewed Canadian redirect,
an Australian read timeout, Quebec robots HTTP403, and unresolved CPF benchmark
reuse conditions. Nine requests transferred 894,079 known successful body bytes;
failed-response sizes are unknown. No retries, alternate routes or proxy changes
were used. No source bodies or individual QA from these leads were admitted.

Stage201 retained seven of eight New Jersey leave-insurance candidates: five
ordinary, one numeric and one refusal. One otherwise plausible candidate was held
because its answer asserted a rule outside its declared evidence. No multi-evidence
question was manufactured to meet a quota. All prior holds and historical revocations
remain. Historical counts are not additive totals.

Current robots and full linked legal and privacy policies were reviewed before
bounded source acquisition. Twelve sequential requests transferred 765,912 bytes.
Five official Labor topic pages were approved under New Jersey's express State
information reuse permission, with item-specific notices checked. Two composite
pages with unresolved internal wording and one script-dependent page without its
rendered body remain deferred. No questions were taken from those pages. All five
approved pages share one conservative TEST family. The new July 17, 2026
benefit-linked job-protection scope was checked against the dated official update.

The snapshot contains 1,388 documents and the unchanged 2,317 source FAQ.
The source-only corpus draft has 239 included, 149 excluded and 1,000 pending
sources. Its 540 chunks preserve every included primary character. The new raw
comparison table and 29 source-defined qualifier groups were reviewed. Exact page
headings and the job-protection effective date retain context. Rendered chunks meet
the 512-token and 2,400-character limits. The previous 529 chunks were reused
unchanged. All five approved bodies remain included independently of retained QA.

All 58 DEV items and 76 declared spans map by offsets. The 62 necessary units
for 47 answerable cases and 18 Spanish aliases retain their exact chunk sets;
11 refusals have gap-context diagnostics only. Global alias and whole-DEV
clearance remain unfinished. No actual DEV score, TEST inference, selection
lock or generation is claimed.

The item audit read 38 targeted historical records, 18 neighboring QA pairs,
22 numeric-template comparisons, nine additional parental-benefit QA comparisons
and two changed active neighborhoods. Only eight new questions were encoded in one
local batch; old candidate vectors and 44,581 historical vectors were reused.
There were no paid APIs, model downloads or subagents. All content review was by
a single Codex workflow, not a human or expert panel. Code is unchanged from the
534-test plus 58-subtest regression; the retained pool was freshly validated.
Sources, individual QA, annotations and weights remain local. PR8 stays draft
and unmerged; exact-commit CI is checked after publication.

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
