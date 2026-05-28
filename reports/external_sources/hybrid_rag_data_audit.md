# Hybrid RAG Data and Source Audit

Audit date: 2026-05-28

## Current Active Curated Corpus

The curated hybrid RAG corpus in `data/04_curated` now contains:

- `201` page records
- `1,180` snippet records
- `1,381` total RAG records
- `56` source documents
- `3,850` SFT records, including `3,400` answerable records and `450` unsupported counterexamples

This is enough for the current hybrid RAG research loop around consumer insurance coverage, declarations pages, deductibles, limits, endorsements, replacement cost, actual cash value, and policy-review abstention behavior.

## Supplemental Data Added

Recent curation runs pulled two previously local-but-not-curated Delaware DOI PDF guides into the active curated RAG corpus:

- `de_auto_insurance_guide.pdf`
- `de_homeowners_guide.pdf`

The latest expansion also added nine high-signal official DOI/OCI sources:

- Nebraska DOI policy forms, coverages and limits, homeowners terms, auto coverage, and auto shopping pages
- Wisconsin OCI homeowners guide, homeowners FAQ, and homeowners savings / ACV-vs-replacement-cost page
- Pennsylvania DOI homeowners insurance guide PDF

Together, these updates raised the active RAG corpus from `45` to `56` source documents and from `1,259` to `1,381` records.

## Source Mix

The active corpus is grounded in regulator or quasi-regulator sources:

- NAIC glossary and topic-style consumer material
- State DOI / insurance regulator material from CA, DE, FL, MD, NC, NE, NY, PA, SC, TX, WA, and WI
- Local official PDF guides and declarations-page explainers

Coverage by page-level domain:

- homeowners: `110`
- auto: `72`
- insurance-general: `11`
- disability: `5`
- travel: `3`
- renters: `3`
- flood: `1`

High-value content types now include:

- declarations-page explainers
- policy-form explainers
- coverage explainers
- claims guides
- deductible guides
- settlement explainers
- official glossary content

## Exclusions and Failed Sources

The curation process intentionally excludes weak or risky sources:

- `iii_ho3_sample_policy` remains excluded because redistribution may be restricted.
- Minnesota DOI pages were fetched but remained low-signal after parsing, so they should stay excluded or manifest-only.
- Massachusetts DOI pages returned `403` during scripted fetch; keep them as candidate sources, but do not silently mix blocked pages into the active corpus.
- The old Washington homeowner guide PDF URL returned `404`; the surrounding Washington OIC web explainers are still included.

## Remaining Gaps

The corpus is sufficient for general hybrid RAG experiments, but not yet a complete production policy-packet benchmark.

Remaining useful additions:

- more real local policy packets with `packet_manifest.json`
- carrier-issued declarations, base forms, endorsements, and schedules that can be used locally without redistribution risk
- more flood, renters, umbrella, cyber, and commercial insurance packet examples
- explicit train / validation / test split metadata for SFT records, so source-level leakage can be checked directly

## Synthetic Packet Examples

The repository also includes synthetic packet examples in `examples/policy_packets/`.

These examples are not part of the official curated regulator corpus. They are redistribution-safe local demo packets for testing packet-aware retrieval behavior:

- `synthetic-homeowners-001`: declarations, base homeowners policy, water backup endorsement, scheduled property
- `synthetic-auto-001`: declarations, base auto policy, rideshare endorsement, vehicle schedule
- `synthetic-renters-001`: declarations, base renters policy, identity theft endorsement, property schedule

They are useful for exercising cross-document logic before private or carrier-issued packet data is available.

## Validation

Validation command:

```bash
.venv_py313/bin/python main.py validate-curated-data --dataset-dir data/04_curated --output-dir reports/research_proof --min-unsupported 400 --min-sft-records 3600 --min-rag-records 1300 --no-update-summary
```

Result:

- status: passed
- empty answerable SFT evidence: `0`
- unsupported records: `450`
- RAG corpus records: `1,381`

Smoke retrieval also built a temporary curated index successfully at `/tmp/insurerag_curated_index`.
