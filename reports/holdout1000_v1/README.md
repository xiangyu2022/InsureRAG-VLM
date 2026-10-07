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

The latest construction checkpoint is stage93_progress.json: **631 provisional
TEST and 58 DEV records across 143 TEST document families**;
**accepted/frozen counts remain zero**. All 689 retained item gates pass.
Another 369 independent TEST questions are needed. All four TEST type
targets remain unmet; DEV minimums are provisionally met. Publisher concentration
stays within 20% and each document family at most 20. Earlier reports are historical
snapshots and must not be added together.

Stages89–93 reviewed 54 new drafts, retained 53 TEST and held 1; net
TEST growth from stage88's 578/58 is 53. TEST tasks are 407 ordinary,
91 numerical, 82 multi-evidence and 51 insufficient-evidence;
remaining targets are 193, 59, 68 and 49. DEV remains 25/10/12/11.
A proposed FEGLI assignment question was held for repeating an existing core need.
Cached OPM enrollment material remains deferred because held-source family overlap
is unresolved. Prospective ACC source terms timed out; no retry, alternative
retrieval or approval followed. Source-only stages90 and92 add no questions.

NHC official explanatory HTML was acquired under verified CC BY-NC4.0 terms,
robots restrictions and exact linked URL guards. Sources retain written attribution,
license and disclaimer references; images, external forms and third-party content
were excluded. Whole-publisher TEST assignment preceded any development. The NHC
coverage overview and its substantively overlapping new-claim page share one family
at its unchanged 20-item cap. Complaint/dispute/rights pages and home claim-assignment
pages form two further reviewed families; common navigation does not create a
substantive family link. Immutable snapshots15 and16 preserve all earlier document
and original FAQ bytes. Source text and individual questions remain local.

Selected dates distinguish 2022 cap phase-in, damage-based 2024 statutory regimes,
and interaction-based insured-person rights. Numerical results were independently
recomputed; multi-evidence tasks require each cited rule; refusals preserve missing
personal facts or unavailable applicable procedures. Candidate/history neighbors,
changed active relations, evidence correspondence, source permission and family
isolation received sequential Codex content review, not human/expert adjudication.
All earlier holds/revocations, same-ID repairs, CA task correction and TDI DEV
assignment remain. A copied private ledger display label was corrected by an
append-only erratum; the actual vector counts and public totals were already correct.

54 new vectors used 3 sequential local model processes and
5 small forward batches; old candidate/history vectors were reused.
Local source acquisition used 10 bounded HTTP requests and 777,695 response bytes.
There were no paid services, subagents, model downloads, training, final QA inference,
test-directed tuning or raw-data publication. Offline CPU verification passed
480 tests and 58 subtests. Global count/type gates still block freezing.

Prior commit1e28a0f passed all five CI jobs in run37649728442. This checkpoint's
exact-commit terminal CI receipt is recorded separately. PR8 stays draft/unmerged;
the PR-description API403 restriction remains binding.

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
