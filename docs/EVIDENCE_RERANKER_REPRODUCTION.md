# Evidence reranker: reproduction and audit

This iteration keeps the selected query adaptation and original BGE document
encoder fixed, refreshes training hard negatives, and trains the MiniLM reranker.
It also independently tunes the old reranker's fusion weight, so an inference
configuration gain is not automatically attributed to gradient training.

## Run the selected research retriever

From `work/InsureRAG-VLM`, use the CUDA environment
`../.venv-qwen-smoke/Scripts/python.exe` in place of `python` below:

```powershell
python scripts/query_evidence_reranker.py --device cuda --corpus insuranceqa --question "What happens when a term life insurance policy expires?"
python scripts/query_evidence_reranker.py --device cuda --corpus condition --question "When can a health plan impose cost sharing on preventive services?"
python scripts/query_evidence_reranker.py --device cuda --corpus fiqa --question "How are term life and whole life insurance different?"
python scripts/query_evidence_reranker.py --device cuda --corpus finqa --report ADI/2009 --question "What is the interest expense sensitivity to a 100 basis point change in LIBOR?"
```

The fourth route requires an explicit company/year report identifier. It searches
all released pages from that report, not an oracle gold page. FinQA originally
supplies a report page with each question; this project defines a harder
report-scoped evidence retrieval variant. It is not a directly comparable FinQA
leaderboard execution/program accuracy result and is not open-corpus company
identification. Raw question text remains unchanged in the benchmark.

The CLI verifies the selected model, frozen validation-scoring fingerprint,
document index and source text. It uses two query encoder passes, retrieves
dense100 + positive-BM25 top100, reranks at most 200 candidates, and returns
original text/row evidence. It does not call Qwen or execute arithmetic programs.
CPU inference is supported by the code; recorded smoke checks use CUDA. Cold
loading timings are not steady-state production latency measurements.
The recorded laptop GPU was shared with other desktop workloads during part
of training. Runtime and peak allocated-memory records describe this run;
they are not a dedicated-hardware throughput benchmark.

## Data and model locations

Keep these sibling directories:

```text
work/
  InsureRAG-VLM/
    data/benchmarks/finqa_evidence_v1/
    data/training/evidence_reranker_v1/
    reports/evidence_reranker_v1/
  models/
    bge-small-en-v1.5/
    insurerag-query-adaptation-v1-seed-42/epoch-2/
    insurerag-condition-listwise-v1-seed-123/epoch-1/
    insurerag-evidence-reranker-v1-seed-42/{epoch-1,epoch-2}/
    insurerag-evidence-reranker-v1-seed-123/{epoch-1,epoch-2}/
  finqa_download/
```

The new package is incremental over Domain_Trained, Retention, Condition and
QueryAdaptation deliveries. The local workspace already has those files.
Public base-model downloads and Python/CUDA runtimes are external dependencies.
Source/model/index hashes, actual package versions and GPU identity are retained
in the experiment records and package manifest.

## Source verification and isolation

FinQA is pinned to upstream commit
`0f16e2867befa6840783e58be38c9efb9229d742`, with the repository's README, MIT
license, original split files and corrected table formatter preserved. The
[official repository](https://github.com/czyssrs/FinQA) documents earlier label
leakage caused by inconsistent positive/negative table formatting. Our formatter
uses only the source `filename`, `pre_text`, `post_text` and `table` fields. It
does not inspect questions, gold indices, supplied retrieved passages, model
inputs or reasoning answers. All gold support texts are checked against the
uniformly generated source rows before use.

Original train/dev/test have distinct pages but overlapping annual reports.
This fixture excludes training questions from any development/test annual
report, and excludes development questions from test annual reports. The
resulting fixture contains 2,351 training questions, 496 validation questions
and all 1,147 original public test questions. One invalid training reference
(`text_-1`) is excluded instead of silently repairing its label. No test
questions are removed. Training is further filtered for duplicate questions,
held-out evidence text and near-question overlap, then for adequate distinct
negative candidates; the actual new financial-report training count is 2,075.

The combined reranker training set has **33,426 unique questions** and **367,338
mined negative pairs**: 11,629 insurance FAQ, 259 government, 13,999 general,
5,464 FiQA and 2,075 financial-report questions. FiQA is newly used for this
reranker's supervised training, but already trained the previous query encoder;
do not count its 5,464 questions again as new project-wide data. Relative to the
previous query experiment, 2,075 unique questions are newly added. Relative to
the old reranker, 100 previously included questions are removed and 7,539
questions are added, for a net increase of 7,439.

The 192,231-entry combined cache is a collection of source passages/rows, not
192,231 clean expert-labelled examples. It includes 86,421 raw FinQA rows;
punctuation-only extraction noise and duplicate content are retained in the
source corpus and quantified separately. Known positive text aliases cannot
be sampled as negatives, and held-out reports and matching held-out texts are
excluded from both positive and negative training pairs. Historical InsuranceQA
shared answer labels remain disclosed. Annual-report isolation does not imply
unseen companies: 86 companies occur in both actual training and test years
(the pre-filter training fixture shares 87 companies with test).
The original financial-report test retains nine duplicate report/question
groups (1,138 distinct groups among 1,147 rows). Post-test sensitivity checks
give each group equal weight and also inspect identical evidence-text aliases;
neither changes the original primary labels. Original questions may rely on
page context, so expanding search to all released report pages can introduce
ambiguity and alternative correct evidence that the source labels do not cover.

## Recorded sequence

Existing completed outputs are immutable. Do not rerun initialization over this
delivery. Use an independent workspace with the inherited inputs and empty
versioned output directories if repeating training.

```powershell
python scripts/prepare_evidence_reranker_data.py
python scripts/mine_evidence_reranker_training.py
python scripts/run_evidence_reranker_experiment.py
```

The raw source downloader is supplied as `inspect_finqa_source.py` beside the
repository. For exact recorded inputs, prefer the pinned delivered archive
files/provenance over resolving a new upstream branch head. The preparation
script does not execute downloaded source code.

The experiment runner waits for successful mining, registers the selection
protocol, scores the old-model validation control, trains each seed for two
epochs, scores both epoch checkpoints on validation, chooses a configuration,
then evaluates only the selected new model and old-model controls on test. It
does not tune on test scores. Model training and evaluation implementations,
source fixtures and initial weights are hashed before gradients.

Three fusion weights (.5, .75, 1.0) are considered for each of four trained
checkpoints and the old reranker: 15 configurations. Query blend, candidate
budget and lexical weight stay fixed. The validation utility weights FAQ MRR
.5, FiQA MRR .15, government MRR .1, general MRR .1, and financial-report
complete-evidence@5 .15. Guards constrain domain Hit@10, forbid a FAQ Hit@1
drop from the old default, and bound financial-report complete-evidence loss.
Promotion requires exceeding the best eligible old-model fusion setting.
The full fallback and tie rules are in `selection_protocol.json`.

Training combines listwise cross entropy, a head-sensitive pairwise margin
term, and KL retention from the old reranker. Pair weights use detached
reciprocal-rank swap differences. Each presentation includes one cyclically
sampled original positive and five distinct negatives; all other known
positives are excluded from negative pools. Non-FinQA negatives scoring close
to the positive under the teacher receive lower weight, without becoming
invented positive labels. This combines several changes; it is not an isolated
claim for a novel loss or a causal estimate of the new data's effect.

## Verify and inspect

```powershell
python scripts/audit_evidence_training_data.py
python scripts/profile_evidence_training_errors.py
python scripts/verify_evidence_experiment.py --require-test
python -m pytest -q
```

Use the full CUDA/Python environment for model/data audits; use the CPU test
environment for pytest. Final verification checks actual parameter changes,
update and pair counts, unchanged query weights, data masks, every scored
checkpoint's complete inference-file fingerprint and the selection recomputed
from validation. Verification/report commands write audit artifacts; they do
not retrain models or rescore tests. Hashes naturally change if audit timestamps
are regenerated, so preserve the delivered records when checking the package.

Two pre-training mining fixes are documented in `preflight_mining/`: unnecessary
full random-candidate materialization, and lexical IDs discarded by text-alias
deduplication but still referenced during teacher-score lookup. The failed run
produced no trained weights or test scores. Source data and labels were not
changed by these fixes. A completed verified embedding cache was safely reused.

## Evidence and limits

- `head_failure_diagnosis.json`: previous historical FAQ first-rank regressions;
  diagnosis only, not training examples.
- `training_error_profile.json`: confusing negatives on training questions,
  with original labels and no fabricated expert review.
- `data_verification.json`: original support facts, report/source separation,
  positive/negative masks, truncation lengths and overlap qualifications.
- `train_seed_*/`: actual gradient steps, loss logs and epoch weight hashes.
- `{valid,test}_candidates/`: fixed query/corpus-specific candidate lists;
  gold evidence is never injected into evaluation candidates.
- `{valid,test}_scores/`: raw reranker scores and checkpoint fingerprints.
- `{valid,test}_evaluation/`: per-question predictions and paired comparisons.
- `selection.lock.json`, `verification.json`, `delivery_checks.json`: selection
  chronology, deep training checks and final runnable-artifact checks.

The new test is public; base-model pretraining exposure is unknown. It is new
to this project's evaluation, not guaranteed unseen by every pretrained
component. Historical regression sets and validation sets have been reused.
Intervals are clustered by shared labels, source or annual report, with an
additional company-cluster sensitivity analysis for financial reports; they
are not adjusted for multiple comparisons. Complete evidence retrieval still
does not establish correct arithmetic, present-day insurance advice, Qwen
answer quality, GraphRAG performance or production readiness.
The inherited FiQA near-question exposure (`fiqa_q_3446`) remains disclosed;
the source training question is excluded now, but inherited weights cannot
undo historical exposure. The analysis reports the original 648-row test and
the 647-row sensitivity subset without changing model selection.
Complete-evidence metrics have their intended supporting-fact meaning on
FinQA. Multiple FAQ/FiQA relevance labels can instead be alternative answers;
the same raw fields for those corpora should not be interpreted as facts that
must all be included in a single answer.
