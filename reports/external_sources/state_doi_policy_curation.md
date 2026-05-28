# State DOI Policy Curation

## Goal

Reduce hallucination risk by keeping only official state-regulator documents that materially help with:

- explaining common policy concepts
- clarifying consumer-facing insurance terms
- supplementing policy-grounded answers with clearly labeled general guidance

## Keep As Downloaded Content

These sources are retained as actual downloaded files because they are close to the project goal of explaining insurance coverage and policy concepts:

- Louisiana Consumer's Guide to Auto Insurance
- Mississippi Consumer Quick Guide to Home Insurance
- Vermont Consumer Advisory on Rising Insurance Premiums

## Keep As Official Pointers Only

These remain in the dataset as official source pointers, but are not downloaded into the active explainer subset:

- Arizona Insurance and Storms or Disasters Guide
- Massachusetts Homeowners Insurance Page

They are still potentially useful, but were not stable enough for direct download or are better treated as optional expansion material.

## Downgraded To Reference Only

All remaining state DOI pages are treated as `reference_only` because they are mostly:

- regulator homepages
- navigation portals
- complaint or consumer-service forms
- general agency landing pages

These sources are tracked for provenance, but excluded from the main policy-interpretation subset to avoid retrieval noise and unsupported policy-style answers.

## Recommended Use

### RAG

- Do **not** mix `reference_only` state DOI pages into the main policy evidence index.
- Use retained downloaded state DOI guides only as `supplemental_explainer` evidence.
- Policy / declarations / endorsements / tables must remain higher priority than regulator guidance.

### SFT

- Do **not** use regulator portal pages for answer supervision.
- Only use the retained guide-style documents for:
  - general concept explanation
  - caveated answers when policy text is unavailable
- Keep external official benchmark documents separate from training whenever possible.
