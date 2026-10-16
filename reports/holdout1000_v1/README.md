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

The latest checkpoint is [stage128](stage128_progress.json): **761 provisional
TEST and 58 DEV across 153 TEST families**. **Accepted counts remain zero; the
dataset is not frozen.** All 819 retained item gates pass. Another 239 independent
TEST are needed: 127 ordinary, 35 numeric, 42 multi-evidence and 35 refusal tasks.
DEV remains 25/10/12/11 by those types. Historical checkpoints are not additive.

The cached batch reviewed six candidates: five retained and one held as a duplicate
of an existing initial-claims-paid fraud question. New retained tasks comprise one
ordinary, one numeric, two multi-evidence and one refusal. All 99 ledger links,
original 15 revocations, later holds, three same-ID repairs, source/split/family
caps and vector hashes were checked. No retained PBGC dependency was found.

All 1,344 cached raw artifacts and normalized texts passed hash checks. The complete
source-only inventory has 42 exact-text alias groups. Its draft inclusion manifest
contains one prior pilot inclusion, 115 existing-hold exclusions and 1,228 pending
source/structure dispositions. Pending does not mean newly usable or unreviewed in
every previous workflow. Existing evidence must be bound to explicit corpus review.
The whole-manifest gate rejects omitted/pending sources before tokenization and
keeps reviewed distractors searchable. It does not prove permissions by itself.
Global mapping and semantic alternatives remain unfinished; the prior one-source
pilot mapped four items/five spans only.

The six registered ranking arms share deterministic ID ties and bounded per-arm
dense100/positive-BM25-100 unions. Identical cross-encoder pairs are reused without
changing either pool. An offline CUDA smoke used four synthetic sources and one
synthetic query with the three hash-pinned inherited retrieval models. All six
arms ran in about 6 seconds without truncation, generation or real DEV/TEST inputs.
This establishes runtime wiring, not retrieval quality or a DEV selection. Full
corpus review, mapping and DEV orchestration remain to be completed under the
[prospective protocol](evaluation_protocol_v1.json). Retrieval does not require the
unresolved Qwen generation service.

Six new candidate vectors were encoded in one curation forward batch. The separate
synthetic smoke used three encoder batches and one four-pair reranker batch.
Stages127/128 made no network requests. Earlier stage126 Indiana terms preflight
used two requests/65,338 bytes and stopped without approval or consumer acquisition;
all source restrictions remain. No paid API, download, new training, subagent, DEV
scoring or TEST inference ran. Offline checks: 510 tests plus 58 subtests passed.
Raw sources, individual QA and weights stay private. PR8 remains draft/unmerged;
exact-commit CI is checked after publication. Work continues.

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
