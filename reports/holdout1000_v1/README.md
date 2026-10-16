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

The latest checkpoint is [stage385](stage385_progress.json): **1,000 provisional
TEST and 57 DEV across 184 TEST families**. **Accepted counts remain zero; the
dataset is not frozen.** After a global classification check, TEST has **604
ordinary, 150 numerical, 146 multi-evidence and 100 insufficient-evidence**
questions. Four genuine multi-evidence replacements are needed to restore the
registered quota. The minimum count alone does not authorize freezing.

All 77 unique QAs involved in 63 active semantic-similarity edges were read.
That review exposed early NHC questions using a split paragraph, a comparison
foil, or redundant evidence as multiple citations. The entire affected NHC
multi-evidence stratum of 15 items was checked against its source spans and HTML
structure. Four were reclassified as ordinary; their questions, answers and
evidence are unchanged. Eleven retained distinct necessary evidence blocks.
The complete 142-ledger hash chain was verified. None of the 416 effective
inherited item holds appears in the retained pool. No old hold was restored.

The source corpus has **607 included, 361 excluded and 458 pending** across
1,426 documents, with 1,126 partial chunks. Ten complete active-source bodies
(32,441 characters) produced eight inclusions and two pending decisions; all
prior 1,110 chunks are unchanged. TRICARE Prime Remote's active-duty applicability
of point-of-service fees and the student-support threshold remain unresolved
against other cached guidance. Original prose was preserved, not silently fixed.
There are **108 pending sources used by current TEST**, down from 116. TEST
declared-span mapping is **586 complete and 414 pending**, up from 567 complete.
All 57 DEV texts, 61 necessary units and 19 reviewed Spanish support alternatives
are unchanged and rebound to the current corpus and corrected pool.

Global source dispositions, equivalent aliases, family/history review and legacy
disposition reconciliation remain unfinished. The readiness gate still blocks
DEV retrieval before model loading. No actual DEV/TEST run, model selection or
immutable selection lock exists. Four private user-requested examples were shown
read-only and logged locally; future development use must isolate them from TEST.

The 82 focused tests recorded at stage379 remain applicable because this update
changes no code, tests, protocol or family registry. Exact-commit CI for
`240bfb7f556136344f7ce3bbaed7a696c7fa953d` passed all five jobs. This checkpoint
used no model forwards, source network requests or external model API calls.
Review is sequential Codex content review, not human or expert adjudication.
Raw sources, individual QA, annotations and weights remain local. The Agent
scaffold is unchanged. PR8 stays draft and unmerged. Global review continues.

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
`protocol.json`. The provisional count now meets the minimum; unresolved global quality
findings may still require exclusions and independently reviewed replacements.
