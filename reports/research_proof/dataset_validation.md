# Curated Dataset Validation

- Status: PASSED
- Dataset directory: `data/04_curated`
- RAG pages: 201
- RAG snippets: 1180
- RAG corpus records: 1381
- SFT records: 3850
- SFT answerable records: 3400
- SFT unsupported records: 450
- Errors: 0
- Warnings: 6

## Quality Checks

- Empty answerable SFT evidence: 0
- Duplicate SFT prompt groups: 87
- Duplicate RAG citation groups: 37
- Split leakage status: not_applicable

## Warnings

- SFT records do not contain split metadata; source-level leakage cannot be checked for SFT training splits.
- Low-value cached source text should remain excluded or manifest-only: data/04_curated/source_cache/mn_doi_auto_basics.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data/04_curated/source_cache/mn_doi_auto_shopping.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data/04_curated/source_cache/mn_doi_home_claims.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data/04_curated/source_cache/mn_doi_homeowner_guide.txt (190 bytes)
- Low-value cached source text should remain excluded or manifest-only: data/04_curated/source_cache/mn_doi_property_coverage.txt (190 bytes)
