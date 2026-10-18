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

The latest checkpoint is [stage618](stage618_progress.json): **1,000 provisional
TEST and57 DEV across184 TEST families**, retaining **600 ordinary,150 numerical,
150 multi-evidence and100 insufficient-evidence TEST tasks**.
**Accepted counts remain zero; the dataset is not frozen.**

An explicit reviewed deletion helper preserves historical struck-out words as
labeled source metadata, rather than treating them as active instructions. It
rejects partial selections, ambiguous markers and unsupported structures without
changing the DOM. One cached template now preserves all three original deletions.
Removing only the marker wrappers reproduces the previous source text exactly.
Each substantive chunk carries the original revision notice; this is a historical
template, not verified current-law guidance. Snapshot36 changes one source; the
other1,425 document lines and FAQ candidates are unchanged.

Six whole-source exclusions address documented scope conflicts, incomplete
definitions and a duplicated coverage-list explanation. No page was rewritten
or selectively trimmed to retain useful answers. Four current-corpus issues were
resolved by exclusion. A prior review description was corrected: the original
list has five HTML elements but only four distinct named coverages.

The manifest has **795 included,532 excluded and99 pending** documents, with
**47 open issues and19 resolved by source exclusion**. There are **3,970 partial
chunks**, including two new chunks; all previous3,968 chunks are byte-identical.
Declared TEST mappings remain **878 complete and122 pending across30 sources**.
All1,426 per-source hashes and the complete snapshot root digest were checked.
DEV content,61 necessary units and19 aliases keep their previous chunk sets.

**52 targeted tests passed**, including11 new revision cases and one nontext
table case. Tables now reject image/nontext cells; the earlier real168-cell
table transcription remains identical. All five CI jobs passed on preceding
commit `020df40921a37668390a97995f77118133e86318`; this commit needs its own receipt.
Pool154,ledger154,437 item holds and61 reviewed active semantic-pair identities
remain bound. No withdrawn task was restored. Global support,alias,family,
history and legacy clearance remains unfinished.

One official-source open attempt returned no statutory text and did not resolve
the dependent-support boundary. No reference entered the corpus. No model
forwards,new vectors,paid calls or model downloads were used. No actual DEV/TEST
inference,selection lock or freeze occurred. Reviews are sequential Codex content
reviews,not human or expert reviews. Raw sources,QA,annotations and weights stay
local. PR8 remains draft and unmerged.

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
