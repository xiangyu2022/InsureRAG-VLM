# Source-heldout exploratory evaluation

**No promotion.** On the frozen 37-question test, publisher-macro complete-answer
retention increases from 44.29% to 49.76%, but both arms retain 20/37 answers.
The paired two-publisher 95% interval is [-3.33, +14.29] percentage points.
Served answerable coverage declines from 13/37 to 10/37. This is not a stable
or end-to-end quality improvement, and the preregistered three-publisher minimum
was not met. See the [full Chinese report](../../docs/SOURCE_HOLDOUT_EVAL_ZH.md).

## Artifact map

- `preregistration.json`, `candidate_registry.json`, `selection.lock.json`: protocol,
  all three candidates, guardrail failures and frozen inference checksums.
- `sources.json`, `data_flow.json`, `acquisition_registry.json`: public sources,
  terms/attribution, snapshot hashes and 218 original QA -> 84 evaluated questions
  versus 219 indexed answer chunks.
- `history_audit.json`, `publisher_name_audit.json`: initial audit gap, corrective
  3,552-file audit, exclusions, source-name probes and known access limitations.
- `split_manifest.json`: fixed 47/37 question split by publisher and sealed hashes.
- `dev_comparison.json`, `test_comparison.json`: all denominators, separate Recall/Hit,
  complete evidence, packing losses, raw/served metrics and publisher-cluster CIs.
- `diagnosis.json`: explicitly non-blinded source inspection, not expert labels.
- `offline_smoke.json`: no-key CLI execution on existing public PDF snapshots.

`local/` is ignored. It contains source snapshots, extracted pairs, historical
audit strings, sealed labels, raw prompts/outputs, cached vectors, timings and
original/corrected summaries. Do not publish those files or model weights.

## Reproduction boundaries

The public checkout does not contain the separately trained query-adaptation or
cross-encoder weights, the complete historical comparison inventory, or the new
publisher text. Numeric reproduction requires the pinned local inputs. Fresh
web downloads may differ; check hashes and stop if they do. This is not advertised
as a one-command, public-data-complete benchmark.

The run used Python 3.12.10, torch 2.11.0+cu128, transformers 5.18.0, NumPy 2.5.2,
scikit-learn 1.9.1 and Ollama 0.34.2 on Windows/RTX 4070 Laptop. Optional acquisition
and audit dependencies are in `requirements-source-eval.txt`. CPU tests need only
`requirements-dev.txt`; GPU packages and model artifacts are separate.

The current harness expects sibling directories `models/` and
`insurerag-main-improvement/` next to this checkout. The latter holds the pinned
192,231-answer JSONL and BGE index. Its corpus, index and weight hashes are checked
before inference. Model names/hashes and generation limits are recorded in the
selection lock and local run protocols. Ollama must be a user-controlled loopback
server on port 11437 with the pinned `qwen3.5:4b` digest; no API key is needed.

For a **new, separately preregistered run in a fresh workspace**, the workflow is:

1. Review each source's current terms. Only then use
   `python scripts/acquire_source_holdout.py reports/source_holdout_v1/acquisition_registry.json`.
   The registry enumerates explicit pages, not a crawler. Do not fetch excluded
   sources or reuse this run's exposed questions as a fresh holdout.
2. `python scripts/prepare_source_faq.py` mechanically extracts original FAQ
   question/answer pairs. Inspect extraction independently of model results.
3. Run `inventory_source_history.py --roots <local-roots.json> --output <fresh-dir>`.
   The roots file is a list of objects with `alias`, `path`, and optional
   `exclude_relative`. Only explicit project roots are read. Record inaccessible
   paths and all actual train/dev/test/report coverage; do not silently drop gaps.
4. Run `audit_source_overlap.py --historical <historical_texts.jsonl> --output <fresh-audit.json>`.
   It scans all candidates against all supplied historical strings with bounded
   sparse batches. Existing audit outputs are never overwritten.
5. `seal_source_holdout.py` expects the reviewed local audit/inventory at the documented
   `local/` filenames, freezes publisher splits, and refuses an existing sealed
   directory. It is run **before** model development. It does not turn missing
   publisher counts into a valid confirmatory study.
6. `retrieve_source_holdout.py --split dev --arm baseline` performs label-free
   retrieval. Allowed arms are baseline, cross_only, original_query, cross_quarter.
   A distinct `--run-suffix timing` preserves independent timing reruns.
   `summarize_source_retrieval.py` scores fixed denominators separately.
7. `generate_source_holdout.py --split dev --arm <arm>` preserves both raw and
   postprocessed answers and paired empty-context controls. It refuses an existing
   generation directory. `summarize_source_generation.py` removes citation fields
   from content metrics and requires a fresh `--output-name` when correcting metrics.
8. Freeze the decision before any test inspection. The included
   `freeze_source_selection.py` records this run's **failed-candidate rejection**;
   it is not a generic automatic selection policy. Test retrieval/generation verifies
   the lock, allowed comparators, sealed test hash and every frozen inference file.
9. Run only frozen test comparators, then `summarize_source_comparison.py --split test`.
   Keep every original question, errors and refusals in the reported denominators.

For offline verification without model downloads or keys:

```text
python -m pytest -q -p no:cacheprovider
python scripts/smoke_source_holdout_offline.py
```

The smoke command requires a fresh local output directory and uses existing public
PDF snapshots copied into it. Synthetic QA smoke labels are not evaluation evidence.
The research-only `source_scope_audit` function can be tested without any source data;
it is not connected to serving or the frozen experiment.

## Corrections and limitations

The preseal historical inventory covered data roots plus the prior diagnostic,
not every historical report. The expanded audit was completed after the selection
lock; no frozen question was flagged by either audit. Its changed IDF moved one
nonselected exclusion in and one out; the initial exclusions were never restored.
Ten denied demo session directories and their containing upload tree remain out
of scope, as do separately re-extracted raw historical PDF bytes.

The broader Oregon state domain was present in earlier tax/Medicaid references.
This is not a jurisdiction-disjoint or model-pretraining-unseen claim. Publisher
FAQ pairs and exact answer retention are mechanical relevance proxies. Empty
contexts are synthetic controls. Token F1, literal numbers and valid IDs do not
establish semantic accuracy. Only one dev question has multiple gold chunks and
none do in test. Two heldout publisher clusters are insufficient for stable inference.
Positive labels mean an original publisher answer exists, not that every retrieved
context is sufficient. Refusal precision is descriptive for this paired design;
not every refusal on an original question is an unjustified refusal.

The fixed-seed GPU generator was not byte deterministic: 13/47 identical dev empty
prompts returned identical strings across arms. No additional run was selected to
improve the scores. The final summary corrects SOURCE-field contamination of served
content/numeric metrics; original raw outputs and selection-lock evidence remain local.
