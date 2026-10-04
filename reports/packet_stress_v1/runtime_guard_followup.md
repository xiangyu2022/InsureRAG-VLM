# Runtime follow-up after the frozen public-guide comparison

The frozen public-guide runs use the source snapshot in `reports/research_v1/frozen_source_v1.zip`. The changes below were made afterward. Do not relabel the earlier predictions as outputs from this revised runtime.

## Answer postprocessing

- Source labels accept Markdown while retaining exact file and page identity. An unknown explicit citation is rejected instead of silently replaced with another retrieved page.
- When the model supplies no exact page citation, postprocessing considers only pages that independently support the complete served claim (or an eligible verified first paragraph). `citation_origin` distinguishes `model_source`, `evidence_selection`, and no served citation. A selected citation must not be counted as raw model citation correctness.
- A trailing comma is not part of a currency amount. Distinct values such as $750, $7,500, and $750.50 remain distinct.
- Citation validation uses the text actually packed for the cited source when that context is available. Lookup wording is separated from policy IDs, claim IDs, version labels, dates, and coverage/item constraints.
- Document-reference wording such as “using the uploaded guide” is validated against source identity separately from the requested fact. Basic plural/verb variants and “maximum”/“up to” are recognized without equating minimum with maximum or accepting a different named guide.
- When a supported primary paragraph is followed by an unsupported comparison amount, only the independently validated primary paragraph may be served. The raw response is retained, and `answer_repaired=true` identifies this deterministic reduction. Multi-part or ambiguous list/comparison questions are excluded from this reduction; they must not lose a requested second fact.
- Explicit refusals remain refusals. Missing blanket limits, unapproved payments, and unresolved controlling versions cannot be repaired into nearby numeric examples.

These checks are **heuristics, not semantic entailment**. In particular, numeric membership does not establish the mapping from a field to its value: an answer can swap two values printed on the same page while both values remain present in the evidence. The tests cover wrong amounts, wrong sources, wrong IDs, missing fields, and several explicit contradictions; they do not establish general insurance correctness or calibrated confidence. A reviewer should inspect the cited field/value association.

## PDF upload and text availability

Normal RAG generation receives extracted text. The image-QA API and optional learned page-image retrievers are separate paths. Enabling the lightweight image signal does not make normal generation read page pixels.

The revised upload path:

1. Builds an unpublished candidate corpus and index at stable paths.
2. Skips clearly blank rendered pages without requiring OCR.
3. Records unavailable OCR and unreadable pages explicitly, and excludes those pages from text retrieval.
4. Rejects an uploaded PDF with no readable text using HTTP422, including when an older readable document is already active.
5. Activates the candidate only after successful indexing. Failed candidates are deleted; the active corpus and index remain intact.

Partial PDFs can be accepted when they contain readable pages; the UI/API reports readable, blank, and unreadable page counts. Missing Tesseract is not a substitute for the separate image-QA API. In text-only mode, a page without selectable text cannot be distinguished from a blank page without rendering; it is reported as unreadable rather than silently counted as successfully read.

CPU synthetic upload checks on this host verified rejection/recovery with and without image rendering, text-plus-blank handling, partial scanned PDFs, and retention of the previous active document when an unreadable replacement fails. These are ingestion regressions, not a scanned-document OCR accuracy benchmark.
