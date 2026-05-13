# Insurance RAG Roadmap And Inference-Time Impact

## Goal

Move the repository from a generic citation-grounded hybrid retriever toward a stronger
insurance-structure reasoning system without making the default online path unnecessarily slow.

The guiding principle is:

```text
offline structure building + cheap online routing first
slow multimodal or tool-heavy reasoning only when gated
```

## Current Delivered Scope

These pieces are now implemented in the default `hybrid_multimodal` path:

- Dense text retrieval + sparse text retrieval.
- Rule-based query understanding.
- Lightweight table-record extraction and table sparse retrieval.
- Lightweight document typing and clause typing.
- Graph-edge construction across declarations, endorsements, exclusions, and definitions.
- Graph expansion from retrieved evidence pages.
- Lightweight page-image auxiliary scoring.
- Insurance-logic page packing with page-role metadata.
- Structured cited output with `query_understanding`, citation roles, and caveats.

These changes were chosen because they raise insurance specificity while keeping most latency in
offline indexing rather than at answer time.

## Phase 1: Low-Latency Insurance Structure

Status: implemented

Scope:
- `document_type`, `clause_type`, and `coverage_tags` heuristics
- query understanding
- snippet/page/table corpora
- table sparse retrieval
- graph expansion
- insurance-logic context packing

Inference-time impact:
- Low
- Most cost is moved to `build-index`
- Online query cost increases only slightly because:
  - query understanding is rule-based
  - table retrieval is a small sparse branch
  - graph expansion only touches nearby page keys

Why this phase comes first:
- It moves the system from plain semantic retrieval toward insurance-aware evidence assembly.
- It improves endorsement, declarations, and numeric-field handling without invoking a heavier
  model.

## Phase 2: Structured Insurance Reasoning

Status: partially implemented, not complete

Recommended next additions:
- stronger section parsing
- stronger table normalization
- explicit override/conflict resolver
- metadata filters by `document_type`, `coverage_tags`, and `clause_type`
- structured answer fields such as normalized `coverage`, `limit`, and `conflict_notes`

Inference-time impact:
- Low to medium
- Rule-based conflict resolution is still much cheaper than adding another model call
- Better table normalization may add a small CPU cost but should remain below LLM latency

Expected benefit:
- More stable answers for:
  - declarations-page limit questions
  - deductible vs premium confusion
  - exclusion vs endorsement override cases

## Phase 3: Gated Slow Path

Status: planned

Recommended additions:
- table reasoning tool for field normalization and numeric disambiguation
- VLM escalation for scanned pages or visually complex schedules
- graph-aware conflict resolver for base-policy vs endorsement disagreements

Inference-time impact:
- Medium to high
- These should not be part of the unconditional path

Serving recommendation:
- default fast path:
  - dense+sparse retrieval
  - table branch
  - graph expansion
  - lightweight rerank
  - text answer
- medium path:
  - default path + table reasoning tool
- slow path:
  - medium path + VLM escalation on 1-2 targeted pages only

This gating is important because insurance RAG often needs structured reasoning, but it does not
need full multimodal reasoning on every question.

## Phase 4: Learned Reranking And Deeper Multimodal Reasoning

Status: planned

Recommended additions:
- learned cross-encoder reranker on top candidates
- LLM-based query understanding fallback
- full page-image reasoning for scanned declarations, schedules, and forms

Inference-time impact:
- High
- These are the most likely modules to dominate online latency

Guardrails:
- rerank only the top 10-20 candidates
- only invoke learned query understanding when rule-based understanding is uncertain
- only send a very small page set to the VLM

## Practical Benchmark Plan

The next benchmark pass should report both quality and latency for these configurations:

1. `hybrid_text`
2. `hybrid_multimodal --disable-image-signal`
3. `hybrid_multimodal`
4. `hybrid_multimodal` plus future table-reasoning or VLM escalation when added

Latency reporting should include:
- p50 query latency
- p95 query latency
- candidate-generation time
- rerank time
- answer-generation time

Error reporting should break down:
- text-only hit
- table-assisted hit
- graph-expansion rescue
- image-assisted hit
- override/conflict miss
- image-noise false positive

## Recommendation

The repository should continue to prioritize:

1. better offline structure building
2. cheap online routing
3. strict gating for expensive reasoning

That path fits insurance QA better than a page-image-first system because most questions are driven
by declarations, endorsements, exclusions, definitions, and numeric evidence, which are better
served by structured retrieval than by unconditional visual inference.
