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

The latest item-construction checkpoint is stage114_progress.json, consolidated in
stage115_progress.json: **722 provisional TEST and 58 DEV records across 148 TEST
document families**. **Accepted counts remain zero; the dataset is not frozen.**
All 780 retained item gates pass. Another 278 independent TEST questions are needed.
All four TEST type targets remain unmet; DEV minimums are provisionally met.
Earlier reports are historical snapshots and must not be added together.

Stages107–115 reviewed 66 drafts, retained 58 TEST and held eight. Five previously
retained NY surprise-bill questions were revoked after conservative source-family
reconciliation, for a net gain of 53 TEST from published stage106. The complete NY
rights family remains capped at20. TEST contains453 ordinary,106 numerical,101
multi-evidence and62 insufficient-evidence tasks; remaining targets are147/44/49/38.
DEV remains25/10/12/11. Publisher and document-family split isolation pass, with
unchanged limits of20% per publisher and20 items per family. All94 ledger links,
prior revocations, three same-ID repairs and current vector keys/hashes were checked.

Cached OPM/TRICARE sources were supplemented by one dated35-page OPM FEGLI booklet,
four EU consumer pages and two RRB booklets. The FEGLI booklet and13 program-page
documents form one conservative family with20 TEST items. Four EU pages form two
families, and each RRB booklet remains a complete family. Snapshot21 preserves all
previous document bytes and2317 original FAQ candidates and now has1334 documents.
OPM agency prose uses a reviewed exact-PDF scope; EU-owned prose uses CC-BY4.0 with
attribution and adaptation notice; RRB agency text uses its explicit educational/
informational reuse permission and attribution. Third-party text/assets and logos
are excluded. Raw sources, attribution details and individual questions remain local.

The review held historical or retained-core duplicates, unnecessary retrieval,
ambiguous provider scope and unresolved Medicare entitlement wording. One nominal
multi-evidence question was reclassified ordinary because an extra span was not
necessary. Numerical tasks independently verify operative caps, offsets, rounding
and payment rules. Refusals preserve missing case-specific facts. Historical and
candidate neighbors, including changed active relationships, received sequential
Codex content review, not human/expert review. No final QA inference or tuning ran.

PBGC dependencies were checked both before and after this batch: zero retained
items depend on PBGC. Its written-permission hold remains, without alternate
retrieval. IRDAI's robots prohibition was respected before a consumer-page request;
Maryland permission remains unresolved, Minnesota retains its cross-host hold,
and FCA copyright terms remained inaccessible. The EU planned-care page and RRB
Medicare/UB11 materials remain unadmitted. These are source-specific holds, not a
claim that all compliant construction sources are exhausted.

Only66 new vectors were encoded in five sequential local model processes and seven
small forward batches; old vectors were reused. This checkpoint used24 bounded
HTTP requests and3,319,841 response bytes, plus five separately accounted browser
calls. A robots guard denial occurred before HTTP. No paid model API, model download,
subagent, training or final QA inference was used. Offline CPU verification passed
480 tests and58 subtests. Exact-commit CI is verified and recorded separately.
PR8 remains draft/unmerged; its description API403 restriction remains binding.

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
