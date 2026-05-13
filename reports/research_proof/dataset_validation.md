# Curated Dataset Validation

This curated dataset is now more than a validation artifact: when available, `rag_pages.jsonl`
and `rag_snippets.jsonl` are also the preferred default retrieval corpus for the repository's
`hybrid_multimodal` query pipeline.

The current hybrid pipeline also derives lightweight table records and document-graph edges from
this curated corpus during `build-index`, so the curated pages now feed:

- dense and sparse snippet retrieval
- dense and sparse page retrieval
- lightweight table retrieval
- graph expansion across declarations, endorsements, exclusions, and definitions

- Status: PASSED
- Dataset directory: `data\04_curated`
- RAG pages: 169
- RAG snippets: 1090
- RAG corpus records: 1259
- SFT records: 3600
- SFT answerable records: 3200
- SFT unsupported records: 400
- Errors: 0
- Warnings: 6

## Quality Checks

- Empty answerable SFT evidence: 0
- Duplicate SFT prompt groups: 81
- Duplicate RAG citation groups: 37
- Split leakage status: not_applicable

## Warnings

- SFT records do not contain split metadata; source-level leakage cannot be checked for SFT training splits.
- Low-value cached source text should remain excluded or manifest-only: data\04_curated\source_cache\mn_doi_auto_basics.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data\04_curated\source_cache\mn_doi_auto_shopping.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data\04_curated\source_cache\mn_doi_home_claims.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data\04_curated\source_cache\mn_doi_homeowner_guide.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data\04_curated\source_cache\mn_doi_property_coverage.txt (190 bytes)
