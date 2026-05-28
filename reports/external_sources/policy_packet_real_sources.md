# Policy Packet Graph And Real Data Sources

## Why This Exists

The repository now supports packet-aware local policy ingestion so that real policy documents can be grouped across:

- declarations pages
- base policy forms
- endorsements
- schedules

This is intended to reduce the common failure mode where retrieval finds a base-policy exclusion but misses the later endorsement that changes the result.

## Primary Data Source For Policy Interpretation

The preferred source for packet-aware policy interpretation is **local real policy data** supplied outside Git and described with `packet_manifest.json`.

Expected source classes:

- carrier-issued declarations pages
- carrier-issued base policy forms
- carrier-issued endorsements / riders
- policy schedules and numeric coverage summaries
- redacted or internal specimen packets approved for local use

These documents are treated as:

- `source_origin = local_real_policy_packet`
- higher-priority evidence than consumer explainers
- local-only artifacts that should not be redistributed by default

Example manifest: `examples/packet_manifest.example.json`

## Existing Official Public Sources Already In The Repo

The repository already contains real public official insurance documents used for retrieval development and explainer coverage. These are still useful, but they should remain **supplemental** to real policy packets.

### Official public document collections

- `public_docs`
  - manifest in `src/insurerag_vlm/qa.py`
  - downloaded under `data/00_raw/external/public_docs`
- `state_doi_docs`
  - curated state-regulator pointers and a small downloaded subset
  - manifest in `src/insurerag_vlm/state_doi_sources.py`
- `EXTERNAL_WEB_SOURCES`
  - curated official web/PDF sources used to build `data/04_curated`
  - manifest in `scripts/curate_insurance_datasets.py`

### Representative official sources already enumerated in code

- National Association of Insurance Commissioners
- Maryland Insurance Administration
- California Department of Insurance
- Texas Department of Insurance
- New York Department of Financial Services
- Washington Office of the Insurance Commissioner
- South Carolina Department of Insurance
- Delaware Department of Insurance
- Florida Department of Financial Services
- North Carolina Department of Insurance

These are real official materials, but most are:

- consumer guides
- declarations explainers
- glossary pages
- endorsements explainers

They are valuable for terminology, coverage explanations, and supplemental grounding, but not a replacement for carrier-issued packet evidence.

## Recommended Evidence Hierarchy

For production policy interpretation, use this order:

1. Local real policy packet documents described by `packet_manifest.json`
2. Local approved specimen packets or local-only redacted policy samples
3. Official public declarations explainers and policy-form explainers
4. Official consumer guides and glossaries

## Current Expansion Strategy

The new packet-aware graph implementation assumes that any **expanded policy reasoning corpus** should come from one of these provenance classes:

- `local_real_policy_packet`
- `official_public_document`
- `curated_real_official_document`

This keeps packet reasoning tied to real documents rather than synthetic text.
