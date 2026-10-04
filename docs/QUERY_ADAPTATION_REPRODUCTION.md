# Query adaptation: inference, audit and reproduction

This is an asymmetric retrieval research experiment. The selected encoder must
encode **queries only**. It is not interchangeable with the original BGE document
encoder. Results and limitations are in [the Chinese report](QUERY_ADAPTATION_UPDATE_ZH.md).

## Existing workspace: run a query

Run commands from `work/InsureRAG-VLM`. The existing CUDA environment is
`../.venv-qwen-smoke/Scripts/python.exe`; the following uses `python` as shorthand.

```powershell
python scripts/query_adapted_retrieval.py --device cuda --corpus insuranceqa --question "What does renters liability insurance cover?"
python scripts/query_adapted_retrieval.py --device cuda --corpus condition --question "When can a health plan impose cost sharing on preventive services?"
python scripts/query_adapted_retrieval.py --device cuda --corpus fiqa --question "How are term life and whole life insurance different?"
```

`--device cpu` is also supported by the implementation but the recorded three-route
smoke test used CUDA. Loading the models and rebuilding BM25 take time; the saved
smoke timings include cold start and do not measure production latency.

The CLI verifies the selection contract, checkpoint inference files, index and
corpus identities. It retrieves and reranks original source passages, returns
three by default, and makes zero generation calls. The PDF/Qwen service is a
separate path; this CLI does not silently replace its model or index.

## Artifact layout

Keep these sibling directories when applying the incremental delivery:

```text
work/
  InsureRAG-VLM/
    data/benchmarks/fiqa_v1/
    data/training/query_adaptation_v1/
    reports/query_adaptation_v1/
  models/
    bge-small-en-v1.5/
    insurerag-condition-listwise-v1-seed-123/epoch-1/
    insurerag-query-adaptation-v1-seed-42/{epoch-1,epoch-2}/
    insurerag-query-adaptation-v1-seed-123/{epoch-1,epoch-2}/
    insurerag-query-ablation-v1-seed-42/{epoch-1,epoch-2}/
    insurerag-query-ablation-v1-seed-123/{epoch-1,epoch-2}/
  fiqa_download/
```

The zip is a delta over the earlier Domain_Trained, Retention and Condition
deliveries. The local workspace already includes those layers. The manifest
lists every inherited archive and hash. Public base models and Python/CUDA are
external dependencies, not a bundled environment image. Use the pinned public
BGE download from earlier reproduction instructions and verify its weight hash:

- Original BGE: `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`.
- Fixed insurance reranker: `5c2a25cc8bc6adae9711646df96d659f4a1f1321a7c2080b8c7acc50c817d723`.
- Selected query weights: `e067a9a14428aabb32dbd2f0e84a7af31bcb8290ffc6fe422faed906d4d5b857`.

The document cache has 105,810 original-BGE vectors; each evaluation searches its
documented corpus subset. Historical government-only documents absent from that
cache are encoded with the original document encoder and retained separately.
The new training denominator has 100,465 allowed documents after isolation.

## Verification and report generation

```powershell
python scripts/audit_query_adaptation_data.py
python scripts/verify_query_adaptation.py --require-ablation --require-test
python scripts/analyze_query_adaptation_results.py
python scripts/audit_query_control_numerics.py
```

The data audit compares imported FiQA questions, corpus and qrels against the
publisher archive; it also checks positive/negative isolation. The deep verifier
checks all eight actual weights, changed parameter values, optimizer counts,
selection chronology, inference file hashes and evaluation scores. The raw BGE
checkpoint has a deterministic `embeddings.position_ids` buffer omitted by
Transformers 5 serialization; the verifier checks it equals `arange(512)` before
excluding it from learned-parameter comparison.

Use an environment with matplotlib for `python scripts/report_query_adaptation.py`.
It regenerates the report, CSV, PNG/SVG and parent model cards from recorded
results; it never changes the selection or reruns test inference. Frozen
experiment scripts reject unexpected code/data/model hashes.

`python -m pytest -q` ran in the full CPU test environment: 302 tests and 58
subtests passed. The minimal environment passed 275 tests and 58 subtests, with
13 optional-dependency skips. The rerun of the three evaluation tests passed
after adding the missing optional scikit-learn import guard. Logs and hashes are
recorded with the delivery checks.

## Reproduce training in an independent copy

Preserve the delivered experiment. Training/evaluation scripts use fixed versioned
paths and intentionally refuse to overwrite completed directories. Do not run
initialization commands over the delivered evidence. In a separate reproduction
workspace, copy the frozen data, index, protocol and upstream dependencies first;
reserve empty output directories for each new run. The dated protocol metadata
and hardware-dependent floating-point results are not promised to be byte-identical.

The primary sequence was:

```powershell
python scripts/prepare_query_adaptation_data.py
python scripts/cache_query_adaptation_vectors.py
python scripts/register_query_adaptation_plan.py
python scripts/run_query_adaptation_training.py
python scripts/eval_query_adaptation.py freeze
python scripts/eval_query_adaptation.py valid
```

These initialization steps are for a fresh reproduction workspace with the
required prior experiment inputs. The supplied raw archive, frozen manifests,
preflight encoding amendment and query/document caches are the preferred way to
replay the exact recorded inputs. A fresh network fetch can return newer metadata;
the delivered provenance pins the dataset-card revision and all source hashes.

Primary training is two seeds × two epochs, full-corpus multi-positive softmax
plus KL retention, 3,142 updates and 100,514 repeated query presentations per seed.
The original document encoder and fixed reranker are never optimized. Selection
compares nine configurations, including the original-query pipeline. FAQ,
finance, government and general validation Hit@10 receive weights .6/.2/.1/.1;
all guards and tie rules are in `selection_protocol.json`.

The matched-budget labelled-data control was registered **after** primary
validation and before the new test. Its protocol, implementation hashes,
generated source diffs and generators are retained. Run its two seeds with:

```powershell
python scripts/train_query_data_ablation.py --seed 42
python scripts/train_query_data_ablation.py --seed 123
python scripts/eval_query_data_ablation.py valid
```

The ablation evaluator deliberately has no selection or test mode. It substitutes
deterministic old-question/label/teacher-anchor triples for finance slots while
holding document vectors, slot weights, order, update count and schedule fixed.
Unlabelled FiQA documents remain in both arms. This is a validation-only
supervision contrast, not an independent test-based ablation.

Finally, the once-selected primary test was run through:

```powershell
python scripts/run_query_adaptation_tests.py
```

This checks completion and unchanged selection, verifies all weights, evaluates
only the frozen selected model and controls, then verifies the resulting scores.
It refuses to overwrite the test directory. Do not repeatedly optimize against
these now-inspected historical/new-to-project test results.

## Key evidence files

- `selection_protocol.json`, `evaluation_implementation.lock.json`, `selection.lock.json`:
  pre-training design, pre-validation evaluator lock, validation-only decision.
- `train_seed_*/`, `ablation_train_seed_*/`: losses, actual steps and checkpoint hashes.
- `valid/`, `test/`, `data_ablation/valid/`: raw model scores, embeddings, per-question
  predictions and all metrics; reranker score reuse is input/model/hash checked.
- `data_verification.json`, `preflight_encoding/`: unchanged original labels and
  equal-object Unicode serialization repair before embedding computation.
- `failure_cases.json`, `failure_analysis.json`, `data_ablation_results.json`:
  post-test diagnosis, shared-label/source-cluster uncertainty and sensitivity.
- `control_numerics.json`: small dense-score drift and near-tie rank changes;
  the prior pipeline's per-question retrieval metrics still match exactly.
- `checkpoint_files.lock.json`, `verification.json`, `delivery_checks.json`:
  file identities, actual training evidence and final executable checks.

Preserve InsuranceQA, SQuAD, government and FiQA source attribution/terms from
their data cards and earlier releases. FiQA is historical forum content with
incomplete relevance labels and unknown base-model pretraining exposure. Retrieval
metrics are not proof of current insurance/legal correctness, answer-generation
accuracy, graph-reasoning ability or production readiness.
