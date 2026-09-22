# Retrieval evaluation contract

## Main metric: Recall@5

This is a retrieval + LLM system. **Macro-averaged page Recall@5 is the primary
retrieval metric**: the fraction of each query's labeled relevant evidence pages
found in the first five results, averaged equally across queries. Choose models
on validation Recall@5; report held-out test results only after selection. Report
ties as ties and inspect nDCG@10, MRR@10, latency and answer quality as secondary
diagnostics. Do not average these differently scaled metrics into an ad hoc score.

The default retriever returns five pages. The answer packer currently uses at most
`max_answer_pages=3`, reorders by insurance evidence roles, and applies a character
budget. Recall@5 therefore measures **retrieval availability**, not proof that all
five pages reached the LLM. Check packed-context evidence, answer EM/F1, citation
precision, abstention and latency alongside it. Revisit the cutoff if the serving
retrieval budget changes. The main metric alone is not an end-to-end release gate.

For one query with gold page set G and ranked results:

- Recall@k = unique gold pages found in the first k / |G|, for k=1,5.
- Hit@5 = 1 if any relevant page appears in the first five, otherwise 0.
- MRR@10 averages 1/r for the first relevant page at rank r<=10, otherwise 0.
- Binary nDCG@10 = sum(rel_i / log2(i+1), i=1..10) / IDCG@10.
  IDCG sums the same discounts for min(|G|,10) ideal relevant results. All labeled
  positives currently have relevance 1; graded relevance is not implemented.

Repeated predictions consume positions but cannot earn multiple gains. Gold page
aliases are deduplicated. Source URLs, internal page keys and visual page IDs for
one gold page must be aligned in the legacy lists or represented explicitly in
`gold_evidence`. Preserve document identity; two files with the same basename are
not interchangeable. Answerable rows without gold evidence are rejected.
Unsupported queries are excluded from retrieval averages and evaluated separately.

All three entry points (QA, text benchmark, visual benchmark) use the same scorer.
They require `top_k>=10` to honestly report the @10 diagnostics; Recall@5 still
looks only at positions 1..5. An exhausted corpus may return fewer than ten items.
Rank is the returned position; equal scores do not receive fractional ranks.
Initial text ties use stable corpus order; final hybrid page ties use source ID.

## Version 2 manifests and historical results

Earlier synthetic manifests repeated vague questions such as "What coverage
concept is explained by this source?" with different gold pages. Those rows are
not independent source-identifying queries. The old evaluator also called Hit@5
"Recall@5" and used only the first hit for nDCG. These formulas coincide with the
standard metrics only for a single gold page. Old ablations could request fewer
than ten results while reporting @10 metric names.

The historical files remain unchanged for provenance. Reports at
`reports/retrieval_eval/*.md` are explicitly marked historical and their JSON
counterparts carry an evaluation-status note. Do not present their numbers as
results from the corrected protocol. No new trained-model scores are inferred
from old aggregate metrics.

Run `python scripts/prepare_retrieval_eval_v2.py` to reproduce committed v2
manifests under `reports/retrieval_eval/v2/`. The transformation:

1. Makes the document ID an explicit part of the question, without adding the
   answer or gold page/chunk number.
2. Consolidates identical questions within that document into one query, taking
   the union of already labeled positive pages. It invents no relevance labels.
3. Preserves original row IDs, questions and evidence in provenance, and preserves
   document-disjoint validation/test splits. Multiple pages can now be relevant.

These are **document-scoped synthetic retrieval tests**, not a replacement for a
human-reviewed benchmark of unrestricted insurance questions. Document context
makes the task easier and changes the task definition. Existing positives can
still be incomplete or noisy; unjudged relevant pages may be scored as misses.
The manifest audit reports query/document counts. Further human review is needed
before using these scores to claim generalization or production answer quality.
Repeated questions generate a warning in generic evaluators; the v2 builders
strictly require unique queries.

The clean/targeted/external evaluation builders now write v2 manifests and report
paths by default. Rerun the BGE experiments in an environment containing the trained
model, indexes and full external PDF corpus. The checked-in v2 audit contains no
new model scores. Example using an existing trained curated index:

```bash
python main.py retrieval-metrics data/04_curated \
  reports/retrieval_eval/v2/expanded_targeted/test.jsonl \
  --index-dir reports/training_data_dense/index \
  --retrieval-model models/retrieval/bge-base-insurerag \
  --retrieval-mode hybrid_multimodal --corpus-source curated \
  --disable-image-signal --top-k 10
```

## What actually ranks pages

The hybrid path combines dense/sparse lists using RRF, then applies metadata,
insurance, graph and optional image signals and rolls snippets up to pages.
RRF is a retrieval fusion method; it is unrelated to computing evaluation MRR.

The corrected `dense_only` path uses the maximum raw cosine similarity of its
page/snippet candidates per page. It bypasses metadata reprioritization, sparse
table retrieval, RRF-based ordering, graph expansion, rule reranking, image signals
and page/snippet bonuses. It still uses the same corpus, embedding model, candidate
budgets and page rollup as the hybrid path. Older reports' `dense_only` included
some of these non-dense signals and must not be compared directly with this version.
