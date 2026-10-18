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

The latest checkpoint is [stage586](stage586_progress.json): **1,000 provisional
TEST and 57 DEV across 184 TEST families**, retaining **600 ordinary, 150 numerical,
150 multi-evidence and 100 insufficient-evidence TEST tasks**.
**Accepted counts remain zero; the dataset is not frozen.**

A whole-source payment-scope hold withdrew one active ordinary task, temporarily
reducing TEST to 999. One new independently reviewed ordinary task restored the
quota under the original seed 42 and caps. Three previous ordinary reserves remain
blocked by the family cap; no held item was restored or task reclassified to fill
the vacancy. AB-31's five former active items and one reserve remain withdrawn.

The source manifest has **790 included, 519 excluded and 117 pending** documents.
Every source has a manifest review record; pending records are not admission.
The register has **53 open issues and 13 resolved by source exclusion**. Two whole
source chunks were appended; all prior 3,918 chunks are unchanged. Current partial
corpus size is **3,920 chunks**. Declared TEST mappings are **877 complete and 123
pending**, which does not establish necessary-support or semantic-alias clearance.

All 154 ledger links and 437 effective item holds were verified, with none active.
The 61 active semantic pairs retain their prior reviewed identities. DEV content,
61 necessary units and 19 reviewed aliases retain their chunksets; global corpus-wide
alias/family/history clearance remains unfinished.

The private PDF literal order passes the real renderer for all 35 pages and 13,377
words, with 624 provenance groups. Source context dependencies, licensed-range chunks
and an original unit ambiguity still block admission. No canonical snapshot changed.

**94 synthetic tests passed again**. Code and tests are unchanged from
`30f844e03542fb70c764548fedcd576125b7ad9f`. All five CI jobs passed on the preceding
exact commit `77ca65a718f0e3eab6e63f96cbf5d81bad7832e0`; the new commit needs its own
CI receipt. DEV readiness remains false.

This batch used one web-search call containing two queries and three official page
opens. No reference page entered the corpus. One local embedding batch encoded one
new question, reusing 3,533 prior candidate vectors and 44,581 history question vectors.
No paid calls, model downloads or actual DEV/TEST inference occurred. The registered
generation runtime was unavailable at its last check. No selection lock, TEST freeze
or final evaluation has occurred. Reviews are sequential Codex content reviews,
not human or expert reviews. Raw sources, QA, annotations and weights remain local.
PR8 stays draft and unmerged.

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
