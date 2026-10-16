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

The latest checkpoint is [stage237](stage237_progress.json): **918 provisional
TEST and 58 DEV across 175 TEST families**. **Accepted counts remain zero; the
dataset is not frozen.** Stage235 retained fourteen of fifteen drafts: nine
ordinary, two numeric, two multi-evidence and one refusal. One tax-residency
draft was held for insufficient independence from historical tasks. A renewal
document question was classified as ordinary because its operative rule needs
only one source passage; no multi-evidence credit was claimed.

Another 82 independent TEST are needed: 30 ordinary, 18 numeric, 17 multi-evidence
and 17 refusal. Original quotas, family cap and all prior holds remain, including
the original 15 historical revocations and two stage215 scope revocations.
Historical checkpoints are not additive.

The source-only corpus has 272 included, 158 excluded and 981 pending sources
in a 1,411-document snapshot, with 636 partial chunks. Eight complete cached
ALDOI sources added twelve chunks; eleven reviewed sources remain pending for
vintage, tax-scope or missing structural labels. Five complete CDCP pages added
eighteen chunks in one conservative family. All 606 previously published chunks
are unchanged. Source text, complete tables and relevant qualifiers are preserved.
Whole-source dispositions, global aliases and whole-DEV clearance remain
unfinished. All fourteen new retained questions map to included bodies. The 58
DEV items, 76 declared spans, 62 necessary support units and 18 reviewed aliases
retain their exact mappings. No actual DEV score, TEST inference or selection lock.

This batch embedded fifteen new questions in one local process and one forward
batch, reusing 3,373 candidate and 44,581 historical vectors. Review read 71 short
historical records in full; one 650,269-character Medicare manual was examined
through targeted windows and its relevant calculation section, not read in full.
It also covered 33 neighboring QAs, ten changed active relations, 37 corpus gap
windows and eleven template counterpart QAs. Automatic scores do not override
content review. CDCP acquisition used five GETs (153,359 body bytes) and one web
discovery. Separate Alberta/ICBC policy preflights used six GETs (632,949 bytes)
and one web discovery; rights restrictions stopped both before business pages.
No paid calls, downloads or subagents. Remaining quota is not observable.

Usage corrections preserved: stage217's 19 vectors used one process and two
forward batches (16 plus 3). Stage234 inherited an incorrect discovery count
from a template; its actual web discovery count was zero. Original records are
preserved with local errata. No model rerun was needed for either correction.

Code is unchanged since the 534 tests plus 58 subtests regression run. Fresh
item validation and support mapping passed; global count and task quotas still
fail. Content review is Codex review, not human or expert adjudication.
Raw sources, individual QA, annotations and weights remain local.

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
