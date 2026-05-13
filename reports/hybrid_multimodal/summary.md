# Hybrid Multimodal Migration Summary

## Purpose

This report documents the post-migration default retrieval architecture for InsureRAG-VLM.
The repository no longer treats page-image retrieval as the primary production path. The
default query stack is now:

```text
User Question
  -> Dense Text Retrieval
  -> Sparse Text Retrieval
  -> Query Understanding
  -> Table Retrieval
  -> Candidate Merge (RRF)
  -> Graph Expansion
  -> Lightweight Page-Image Auxiliary Scoring
  -> Multimodal Reranker
  -> Snippet-to-Page Rollup
  -> Insurance-Logic Context Packing
  -> LLM Answer
  -> Tool Router Stub
```

## What Changed

- `hybrid_multimodal` is the default `retrieval_mode`.
- Curated `rag_snippets.jsonl` and `rag_pages.jsonl` are now the preferred retrieval corpus when available.
- Dense and sparse text retrieval form the main candidate set.
- Rule-based query understanding now detects coverage, limit, definition, exclusion, declaration, endorsement, and diff-oriented questions.
- Lightweight section parsing now adds `section_titles`, `section_path`, `section_anchor`, and endorsement form codes to page and snippet metadata.
- Lightweight table extraction now writes normalized table records and a table sparse index for deductible, limit, premium, and schedule-like evidence.
- Lightweight document-structure heuristics now add `document_type`, `clause_type`, `coverage_tags`, and graph edges used for expansion across declarations, endorsements, exclusions, definitions, and section-aware endorsement links.
- Lightweight page-image embeddings are used as an auxiliary page prior rather than as a standalone answer selector.
- Final answers remain page-cited, snippet-grounded, abstention-aware, and now include richer citation-role metadata plus structured conflict summaries for declarations-style numeric evidence and endorsement-override review.

## New Index Artifacts

The default `build-index` flow now writes these retrieval artifacts under `index_dir`:

```text
hybrid_snippets_dense.npy
hybrid_snippets_sparse.json
hybrid_snippets.jsonl
hybrid_pages_dense.npy
hybrid_pages_sparse.json
hybrid_pages.jsonl
hybrid_tables_sparse.json
hybrid_tables.jsonl
hybrid_graph.jsonl
hybrid_page_image.npy
hybrid_page_image_pages.jsonl
```

The legacy `index.npy` and `index_meta.json` files are still emitted for compatibility with
older scripts, but they should be treated as compatibility artifacts rather than the canonical
hybrid index representation.

## Relationship To Older Reports

The existing reports under:

- `reports/ablation/`
- `reports/ablation_real_pdfs/`
- `reports/research_proof_local/`

contain valid historical benchmark numbers, but they were produced before the hybrid multimodal
migration. They should be interpreted as pre-migration baselines for:

- `local_text`
- `visual_stub`
- `local_image`

They do not represent the current default product path.

## Validation Status

Implementation validation completed so far:

- Syntax checks passed for the updated retrieval modules and CLI.
- Targeted unit tests passed for:
  - lightweight page-image embedding generation
  - backward-compatible `rank_pages()` output shape
  - image-signal disable path
  - image-assisted rerank behavior
  - query-understanding detection for insurance questions
  - table-record extraction for numeric insurance fields
  - table/graph artifact generation during `build-index`
  - graph expansion surfacing endorsement-related pages
  - structured query output including `query_understanding` and `caveats`
  - default retrieval-mode configuration

Pending post-migration validation:

- retrieval metrics for `hybrid_multimodal`
- retrieval metrics for `hybrid_text`
- answer/citation metrics after the new reranker
- latency comparison for:
  - base hybrid text retrieval
  - table-aware retrieval
  - graph expansion
  - image-signal-on vs image-signal-off
- error-group analysis including:
  - text-only hit
  - image-assisted hit
  - multimodal-rerank rescue
  - image-noise false positive

For the phased roadmap and expected inference-time tradeoffs, see
`reports/hybrid_multimodal/roadmap.md`.

## Recommended Next Report

The next benchmark report should compare at least:

- `hybrid_text`
- `hybrid_multimodal`
- `local_image`
- optional `colqwen2_*` or `colpali_*` research backends

and should be written as a new post-migration benchmark summary rather than overwriting the
historical baseline numbers.
