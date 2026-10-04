# Local application integration checks

These are development integration records, separate from the frozen public test.
The test documents are a public Delaware auto guide and fictional PDFs. No private
customer policy or paid cloud API was used.

## Upload failure and recovery

`ingestion_before.json` records the observed missing-Tesseract failure, blank-page
failure, zero-text upload accepted in text-only mode, and retained failed upload
blocking the next valid document. `ingestion_after.json` records the actual handler
retake in image-enabled and text-only configurations:

- A pure scan without working OCR returns HTTP422 and does not activate or retain
  the candidate upload.
- A following readable PDF with a blank page returns HTTP200; its one readable and
  one blank page are counted, and the extractive answer gives the fictional `$735`.
- A mixed readable/scanned PDF returns HTTP200 with an explicit unreadable-page
  warning. Partial ingestion is visible and is not a guarantee that omitted pages
  contain no relevant terms.

The tests in `tests/test_ingestion_upload.py` reproduce these contracts without
external models. They do not establish transcription quality or visual QA accuracy.

## Public guide and real Qwen3.5 inference

Question: “Using the uploaded Delaware auto guide, what maximum funeral expense
benefit is included in PIP coverage?”

`public_guide_v3.json` preserves an actual raw answer of `$5,000` with exact page5
citation which was incorrectly rejected by the lexical support guard. The general
wording-normalization repair preserves the distinction between minimum and maximum
and checks document identity; the earlier report remains unchanged.

`public_guide_v4.json` is a separate actual structured call after repair. It returns
the correct primary amount, `abstain=false`, `answer_repaired=false`, and
`citation_origin=model_source`. The report includes the public PDF hash, source hash,
exact Ollama digest, actual prompt evidence, and completed-generation metadata.

The browser flow was also exercised on the local HTTP server: upload18/18 readable
pages, ask the question, expand the page5 preview and retrieval trace. Asking “What
is my collision deductible under my policy?” against that uploaded public guide
returns an abstention; the guide does not establish a personal policy deductible.

A final presentation follow-up exposes the original generator output in a collapsed
“unvalidated” panel, renders it as plain text, shows model-versus-system citation
origin, and labels a shortened/replaced answer accurately. The `$5,000` example was
checked again in the actual browser and displays the original Markdown source line
separately from the validated answer. The UI presentation change does not change the
saved v4 generation or its scores.

The local service is a single-user demo. These checks are not production concurrency,
security, clinical/legal/insurance decision validation, or general policy accuracy.
