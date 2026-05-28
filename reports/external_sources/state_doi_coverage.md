# State DOI Coverage

The `state_doi_docs` dataset still tracks all `50` states, but it is now curated for policy interpretation rather than breadth-only crawling.

## Downloaded Explainer Subset

Only `3` state DOI sources remain as downloaded content in `data/00_raw/external/state_doi_docs/`:

- Louisiana Consumer's Guide to Auto Insurance
- Mississippi Consumer Quick Guide to Home Insurance
- Vermont Consumer Advisory on Rising Insurance Premiums

These were kept because they contain consumer-facing insurance explanation content that is closer to the project goal of interpreting policies and explaining coverage concepts.

## Manifest Only Official Pointers

The remaining `47` state entries are preserved as official pointers rather than downloaded files.

This includes:
- high-value but currently pointer-only sources such as Arizona and Massachusetts
- regulator landing pages, complaint portals, navigation hubs, and similar low-signal pages from the other states

## Why The Dataset Was Pruned

Most state DOI homepage downloads were not strong evidence sources for:

- policy interpretation
- declarations / endorsement reasoning
- coverage / exclusion explanation grounded in policy text

Instead, many of them were dominated by:

- navigation menus
- regulator homepages
- complaint workflows
- administrative forms

Those files were removed from the downloaded subset to reduce retrieval noise and lower hallucination risk in the hybrid RAG system.
