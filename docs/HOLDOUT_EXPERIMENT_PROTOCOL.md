# Independent holdout experiment preparation

As of October 16, 2026, construction has reached 761 provisional TEST and 58 DEV
items. Accepted counts are zero; the benchmark is not frozen. The current
[aggregate checkpoint](../reports/holdout1000_v1/stage128_progress.json) is the
source of truth. Historical exposed results are regression evidence only.

The [prospective protocol](../reports/holdout1000_v1/evaluation_protocol_v1.json)
registers BM25, original dense, original/adapted hybrid and original/adapted
reranked retrieval. All share source-only documents and a fixed encoder. A
reviewed inclusion manifest, deterministic chunking and support-unit mapping
must be completed before DEV scoring. Candidate generation must not use gold
document filters, reference answers or evidence-only chunks.

DEV selects an arm by family-macro complete-evidence@5, with the registered
tie-breaks. Freeze the DEV decision, model/tokenizer hashes, corpus, code,
context packing, prompt and review rubric before any TEST inference. At least
1000 independent TEST and every existing quality gate must pass before freezing
TEST. A design document is not a selection lock or benchmark acceptance.

The holdout_experiment module implements annotation-based metrics and uncertainty:

- score_retrieval reports chunk Recall@5, Hit@5, chunk completeness and nDCG@10.
  Chunk completeness is diagnostic; it is not the primary support-unit measure.
  Refusal tasks have no answer-retrieval score. Short lists do not receive padding.
- evidence_unit_metrics requires each necessary support unit. Equivalent alias
  chunks are alternatives, while a span crossing boundaries can require several
  chunks. It does not perform semantic judgment or build the annotation mapping.
- paired_family_delta preserves paired questions and resamples whole document
  families. Its confidence interval is for the question-macro difference; it also
  reports the equal-family point estimate. It does not yet implement ratio-metric
  intervals such as refusal precision. A single family gets no fabricated CI.

The remaining runner must account for intended denominators, failures, valid
citation IDs versus claim support, refusal precision/recall, complete evidence
after packing, measured timings and actual tokens. A preregistered 100-case
paired semantic-review subset must be labelled Codex content review, not expert
review or whole-test semantic accuracy. Undefined ratios remain null. The final
generator receives only the question and retrieved evidence, never task labels,
reference answers, calculation annotations or gold spans.

The inherited baseline, trained query encoder and trained reranker weights have
verified hashes. Real Qwen3.5-4B integration remains unverified: the usual Ollama
installation/model manifest was absent and both historical local endpoints timed
out. The current runtime location is needed. No download, replacement model,
new DEV comparison or TEST inference was performed. This checkpoint includes
protocol and metric preparation, not completed end-to-end execution.

Validation: 510 tests and 58 subtests passed offline, including denominator,
short-ranking, failure, evidence-alias and family-pairing cases. Individual QA,
raw sources, runtime traces and weights stay local; only aggregates and code are
published. No source holds or count targets were relaxed.

The holdout_corpus helper now validates reviewed source ranges, exact hashes,
encoder token limits and protected atomic spans before constructing chunks. It
maps gold support offsets only after source-only chunking. Missing coverage,
fabricated IDs and stale content fail closed. One full-source pilot mapped four
items/five spans across two chunks, without inference. Global inclusion, semantic
alternatives and DEV selection remain pending. Loopback port 11435 also refused the
read-only runtime probe; the unresolved runtime is not silently substituted.

The whole-source manifest gate requires every snapshot document to have an explicit
include/exclude disposition before tokenization. The inventory verifies 1,344
raw/text hashes but still has 1,228 pending source/structure reviews. Existing review
evidence can be reused; pending is not a claim all these sources are new or suitable
for independent questions. Exclusions and aliases need review without gold filters.

The six-arm fusion core and pinned local retrieval models passed a four-source,
one-query synthetic CUDA wiring run. No real DEV/TEST question was used. Each
candidate union stays within 200, with stable score/ID ties and reuse of identical
cross-encoder pairs. Full DEV orchestration and relevance mapping remain pending;
unavailable generation does not prevent retrieval.


The stage131 implementation adds `scripts/run_holdout_dev_retrieval.py`. Supply
the complete source snapshot, source-only inclusion manifest, label-free DEV
request JSON and separately reviewed DEV clearance. `--check-only` reports all
readiness blockers without model loading or generation. Actual inference also
requires local pinned models and a private output directory. Never mark a draft
clearance ready merely to run the command.

The CLI preserves failed intended requests, fingerprints code/model/tokenizer and
corpus inputs, rejects changed resumes and reuses exact saved results. A completed
resume loads no models. The real-model integration validation used only synthetic
fixtures; actual DEV readiness remains blocked by corpus and support review.
Current partial mapping covers 14 spans and 11 complete declared-evidence items;
it does not approve semantic alternatives. Shared execution timings are recorded
but do not establish standalone per-arm p95 for the selection tie-break. Scoring
and the final selection lock remain separate unfinished work.
