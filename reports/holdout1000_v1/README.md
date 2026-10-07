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

The latest construction checkpoint is stage59_progress.json:
520 provisional TEST and 48 DEV records across 134 TEST document families,
with **zero frozen/accepted records**. All 568 retained item gates pass.
Another 480 independent TEST questions are needed. TEST type targets and
DEV count/numerical/multi-evidence minima remain unmet. Publisher concentration
remains within the unchanged 20% ceiling; each document family remains at most 20.

Stages 53-59 processed 49 new drafts, retained 46, and held 3 new semantic duplicates.
One existing TEST item was also held after the NY women's/breast guidance pages
were joined to the existing healthcare-rights family: 21 items would exceed its
unchanged 20-item cap. Net growth from the 477 TEST/46 DEV checkpoint is 43 TEST+2 DEV.
An IRO/stop-loss draft was reclassified as ordinary QA because its second evidence
already repeated the employer's payment obligation. It receives no multi-evidence
credit. The TEST distribution is 353 ordinary, 73 numerical, 57 multi-evidence and
37 insufficient-evidence. DEV is 22 ordinary, 7 numerical, 9 multi-evidence and 10 refusals;
it still needs 3 numerical and 1 multi-evidence task, hence at least 52 under these counts.

All new retained questions received sequential Codex content review, including
exact support, nearest accessible historical text, active neighbor relationships,
source scope/permissions, and document-family checks. Required-document table
rows were verified against the original cached HTML. A new nine-month ward
document question was held as a duplicate of an existing numerical task;
Secondary Dependents was conservatively grouped with ID Cards and Required Documents.
Exposed travel, general surplus-lines and ALE rules were not reused to fill quotas. The original 15
revocations, six stage 43 DEV scope revocations and inherited holds remain effective.

Forty-nine new vectors used seven sequential local model processes and seven
small forward batches. Unchanged history and old candidate vectors were reused.
No new source requests, paid services, extra agents, training, final evaluation,
test-driven tuning or raw-data publication occurred. Local offline CPU tests passed:
477 tests and 58 subtests. Codex review is not human or expert adjudication;
inaccessible pretraining contamination remains unknown.

Prior commit f4fc836 passed all five CI jobs in run 37595083997. The new checkpoint
requires separate exact-commit CI verification, recorded in the local receipt.
The PR-description integration 403 restriction remains binding.

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
