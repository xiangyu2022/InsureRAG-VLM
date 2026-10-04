# Real-source page navigation diagnostic

171 distinct directed page references in eight archived OPM plan brochures:
17 development cases across APWU/GEHA families and 154 test cases across
SAMBA/NALC/MHBP/Compass Rose/Foreign Service families. All variants within a plan
family stay in the same split. The brochures share an OPM format, and many
references within one plan are correlated. SAMBA contributes 83 test cases.

Inputs supply an anchor passage and the known plan. The task is to include the
referenced target page in a three-page context. These are **structural integration
cases**, not 171 natural insurance questions, claim decisions, or independently
adjudicated answers. The labels and graph links share the literal extraction
process; 100% graph extraction accuracy cannot be inferred from this fixture.

The benchmark uses only singular numeric page references resolved against
unique printed footer labels. It excludes ranges, lists, missing/ambiguous
targets and self-links. Source/target quotes, PDF hashes, printed-to-physical
maps and rejections are archived. References are deduplicated by source-target
page pair. The original 120 synthetic cases are retained separately.

The final run in `reports/opm_reference_navigation_v1/final_verification` compares
anchor + BM25, anchor + bounded graph + BM25, and literal printed-page lookup.
The lookup arm parses the input passage without using target labels. It performs
better than generic traversal here, so this test does not establish a need for
GraphRAG over simple explicit-reference resolution.

The first development run is marked invalid: its lookup arm accidentally used
the gold target as an oracle. Raw outputs remain for traceability. The correction
occurred after viewing test outputs, so the final comparison is exploratory,
not an untouched confirmatory holdout claim.
