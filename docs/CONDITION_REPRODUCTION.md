# Condition-oriented reranker experiment

This is a local retrieval research experiment. It fine-tunes the 22.7M-parameter
MiniLM cross-encoder from the previous retention checkpoint. BGE remains fixed;
this does not train Qwen or validate generated insurance advice.

## Inputs and isolation

The final training fixture contains 25,987 unique questions: 11,729 historical
InsuranceQA questions, 259 publisher-written government questions, and 13,999
original SQuAD crowdworker questions. The 6,000 added SQuAD questions come from
previously unused training-positive paragraphs and emphasize numbers, dates,
time boundaries, and conditions. They are not 6,000 additional insurance questions.

The additional government sources include 16 HTML documents downloaded from the
[DOL ACA FAQ index](https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/affordable-care-act/for-employers-and-advisers/aca-implementation-faqs)
(Parts 56–71) and archived HICRIC government text. `dol_additional_v1/manifest.json`
and its records retain the source URLs, retrieval timestamps, and HTML/text hashes.
Not every downloaded document yields an accepted training pair. Original source
dates still apply; this corpus is not a determination of current law.

The fresh government test contains 157 publisher Q/A pairs from 28 previously
unused URLs (27 families after collapsing ACA proposed/final versions).
It searches 48,172 answer candidates. All paragraphs with held-out source
metadata, plus duplicate normalized answer texts, are excluded from positive
and negative training pairs. Fresh test positives are also checked against all
previous training positive and negative texts. The historical InsuranceQA split
retains its known shared labels and must not be described as document-disjoint.

An initial format audit rejected 22 test pairs and one training pair because a
later question could be embedded inside the extracted answer. The aborted
preflight snapshot, initial protocol, partial caches, and amendment report remain
available. It was stopped during candidate retrieval, before student scoring,
training, validation, or fresh test scoring. Final source-span verification
checks all 203 government pairs and all 6,000 original SQuAD labels. Mechanical
verification is not expert semantic adjudication; headers, footnotes and some
context-dependent questions remain limitations.

## Registered experiment

`reports/condition_v1/selection_protocol.json` freezes the recipe and selection
rule before training. Two seeds (42, 123) each start from the identical previous
retention epoch-2 checkpoint. Each trains for one epoch at learning rate 3e-6
using FP32 parameters and BF16 autocast. FAQ/government/general repeat factors
are 2/8/1; repetitions count as presentations, never new unique questions.

The previous trained model mines difficult candidates from BGE, BM25, previous
negatives, and same-document government answers. Each presentation samples one
original positive, two student-mined negatives, two lexical negatives, and one
random negative. Pairwise margin loss uses confidence 0.25 when the public
teacher scores a negative within one logit of or above the sampled positive,
otherwise 1.0. These remain unlabeled negatives, not invented positive labels.
Public-teacher KL retention is also applied (temperature 2; weight 0.15 FAQ,
0.3 government/general).

The conditional follow-up is recorded in `reports/condition_v1/followup_intent.json`.
It is triggered only if both first-recipe seeds fail validation promotion, before
opening the fresh test. It preserves the same questions, positive/negative IDs,
initial model, learning rate, repeat factors and validation search. Its supervised
loss is listwise cross entropy with 0.02 label smoothing and `log(confidence)`
added to negative logits. Public-model scores still determine negative confidence,
but the KL target is now the frozen previous **insurance specialist**, weighted
0.1 for FAQ and 0.2 for other domains. This avoids explicitly pulling the better
insurance model toward the weaker public insurance baseline. This explanation
is a hypothesis: loss and retention target change together, so the experiment
does not isolate their individual causal contributions.

The follow-up has its own frozen plan, source-preserving anchor-score fixture,
two independent training runs and selection in `reports/condition_listwise_v1/`.
The rejected parent checkpoints and validation scores remain available. Validation
is reused adaptively across the two recipes; only the final selection is evaluated
on the new government test.

Validation compares both seeds and the previous model using exactly the same
six configurations: hybrid100/union200, lexical weight 0.2, cross weights
0.35/0.5/0.65. The objective is 0.6 FAQ + 0.2 government + 0.2 general Hit@10,
with per-domain regression guards and weighted MRR/Hit@1 tie-breaks. A new
checkpoint must beat the best eligible previous-model configuration to be
promoted. Otherwise the previous weights are retained. Seed variability is
reported on validation only. No fresh-test tuning or relabeling is allowed.

Tests report BGE and both public/previous rerankers at the old and selected
configurations. This separates training changes from candidate/fusion changes.
Existing tests (2,000 FAQ, 416 government, 81 government + 600 general) are now
historical regression checks, not new blind tests. The 157-question government
cohort is the only new held-out test this iteration.

## Commands

Run from the repository directory. Use Python with torch, transformers, numpy,
scipy, scikit-learn, requests and threadpoolctl. CUDA training was run locally on
an RTX 4070 Laptop GPU. Public model download revisions and prior data setup
remain in `RETENTION_REPRODUCTION.md`. Checkpoints, indexes and upstream SQuAD
JSON files are sibling/large artifacts, not Git source files.

Outputs deliberately refuse overwrite. Use a fresh reproduction checkout with
the experiment output directories absent; preserve delivered evidence separately.
Do not delete evidence to rerun a command in the active workspace.
The test wrapper can resume already-completed score artifacts only after checking
their data/model/code/selection hashes. An initial relative-path failure before
historical-government scoring was fixed in the orchestration wrapper; frozen
scoring/evaluation code and completed FAQ/mixed results were preserved. See
`operational_recovery.json` for the record.

```powershell
python scripts/fetch_additional_dol.py
python scripts/prepare_condition_data.py
python scripts/audit_condition_fixture.py
python scripts/eval_condition_v1.py plan
python scripts/prepare_condition_training.py
python scripts/run_condition_validation.py
python scripts/run_condition_followup.py
# If the parent recipe was rejected and the follow-up completed:
python scripts/run_condition_listwise_tests.py
# Otherwise evaluate the parent selection with scripts/run_condition_tests.py.
```

For exact source reproduction, use the delivered pinned government records and
raw cache rather than fetching live pages again. `fetch_additional_dol.py` is a
discovery/snapshot command; live government pages may change. The local model
paths default to the same sibling `models` layout as the prior delivery.

The selected configuration can be queried without generation:

```powershell
python scripts/query_condition_listwise.py --device cuda --corpus condition --question "When can a health plan impose cost sharing on preventive services?"
python scripts/query_condition_listwise.py --device cuda --corpus insuranceqa --question "What does renters liability insurance cover?"
python scripts/query_condition_listwise.py --device cuda --corpus hicric --question "How does COBRA coverage coordinate with Medicare?"
```

`condition` searches 48,172 answer passages; `insuranceqa` searches 27,413 FAQ
answers; `hicric` searches 78,830 archived snippets. Only the first two relevant
test corpora have benchmark claims. A snippet-query smoke test verifies execution,
not answer quality over all archived snippets. Both hybrid100 and union200
evaluation artifacts score the full union before selecting a ranking pool;
this experiment does not establish a latency improvement.

All result claims must be tied to `test_evaluation/summary.json`, the frozen
selection, training logs, and code/data/model hashes. Hit@10 measures whether
an original labeled passage appears among the first ten results, not generated
answer correctness, calibration, legal validity, or production readiness.

## Post-test extraction repair

`extract_government_qa_boundaries_v2.py` is a separate, conservative parser for
future explicit-Q/A fixtures. It fixes a wrapped `Q-A5.)` cross-reference being
mistaken for a question boundary. It does not replace the frozen v1 extractor,
alter any delivered benchmark labels, or produce a rescored result. Six
regression tests and the original-source diagnostic are provided:

```powershell
python -m pytest tests/test_government_qa_boundaries_v2.py -q
python scripts/audit_condition_boundary_repair.py
```

The diagnostic restores a 31-word partial answer to a 293-word source passage,
with exact character spans. The repair was designed after seeing this test case;
it therefore cannot turn the existing test into a new blind evaluation. This
parser deliberately does not support implicit-answer or paragraph-only formats.
Broader source review and new held-out sources are needed before a future run.
