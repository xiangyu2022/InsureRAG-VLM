# Incremental page ingestion v2

This extends the existing loaders, hybrid retriever, Chroma store, MCP tools and
review Agent. Historical indexes remain schema 1; no original corpus or TEST/DEV
labels are rewritten. New web uploads select `ingestion_policy=page_v2`.
Historical `build-index` replay keeps its existing default and can explicitly
select `--ingestion-policy page_v2`. For the complete strict Chroma path use:

```powershell
python main.py ingest-pages private/policy.pdf --doc-id policy-stable-id --work-dir .knowledge/policy-v2-run1 --store .knowledge/policy-v2-db
python main.py knowledge-search "your question" --store .knowledge/policy-v2-db --collection page_clauses_v2 --record-schema 2
python -m src.insurerag_vlm.clause_mcp --store .knowledge/policy-v2-db --collection page_clauses_v2 --record-schema 2
python -m src.insurerag_vlm.agent_mcp_cli start --thread-id policy-v2-1 --question "your question" --store .knowledge/policy-v2-db --collection page_clauses_v2 --record-schema 2
```

The Agent defaults to a deterministic mock excerpt for workflow validation,
not a model-quality result. Its approval, edit/review, retrieve and reject flow
is documented in [AGENT_MCP_LOCAL.md](AGENT_MCP_LOCAL.md). Source text and full
metadata persist in the checkpoint; only the controller prompt uses a bounded
metadata projection. Approval still requires current evidence-version validation.

## Page routing and failure behavior

Each physical PDF page is inspected independently. A usable text layer with no
detected raster images, vector paths, annotations or ruled tables uses native
PyMuPDF extraction. Scans, images, charts and mixed pages use a compressed JPEG
and structured multimodal recognition. **A text layer never overrides a detected
visual signal.** Detection is conservative: decorative vectors may also require
vision. Complex unruled text layouts remain a detection limitation and require
review; this is not a trained page classifier or a guarantee of all layout cases.

The defaults are 144 DPI, JPEG quality 85, no chroma subsampling, and maximum
edge 2,000 pixels; allowed quality is 75–95 and edge 800–3,000. Rendering is scaled
before allocating a large PDF raster. Original files are copied unchanged to
hash-named private originals. Metadata records source and compressed-image hashes,
original/rendered dimensions, DPI, quality, filter, parser/converter versions and
original-page normalized coordinates. Figure bounding boxes identify regions of
that original page, not invented crop files.

Native text below 10 rendered pixels fails the readability guard. Scans have no
reliable native font measurement; the backend must declare readability, which
is recorded as **backend_declared_readable**, not independent quality acceptance.
Uncertain/unreadable results remain pending. More demanding pages need explicit
regional rendering/review in a later iteration; this version does not silently
shrink them until unreadable or claim that compression always preserves quality.

Without an explicitly configured, already-present local vision model, visual
pages remain `pending`. OCR is **never** their silent final fallback. The CLI
returns exit code 2 and does not index any partial document. New web uploads
remain inactive if any page is pending; the prior active corpus is unchanged.
Hash-named originals, compressed images and reports are retained under the
private `pending-page-v2` directory before the inactive upload candidate is removed.

The optional CLI flags `--vision-model ollama:EXISTING_MODEL --vision-digest DIGEST`
use the existing `VLMClient.generate_with_images` adapter, check the local model's
identity/capability, and request typed structured JSON. No default connection,
download, paid API, server startup or fallback is added. The prior socket 10013
blocker was not retried in this implementation. Mock visual tests do not validate
real recognition, chart summarization or table extraction quality.

## Page-first structural semantic chunks

The implementation iterates PDF pages (or raster frames / explicit text form-feed
pages) **before** constructing blocks or chunks. A plain text file without form
feeds is one logical page, with physical/printed page fields explicitly marked
not applicable. It is never assigned fictional PDF page numbers.

Native PDF text blocks, paragraph breaks, Markdown headings and existing
`insurance_structure.extract_section_metadata`/clause rules provide structural
boundaries. Visual pages supply typed paragraph/clause/heading/table/figure
regions. Each paragraph/clause is an independent semantic unit. Long units split
only at sentence, list or paragraph boundaries; adjacent sentences are packed
within a soft character budget. A single oversized indivisible sentence/row
group remains intact with `oversized_unsplit=true`, rather than being truncated.
The character budget is a packing limit, not the semantic boundary algorithm.

This is deterministic structural/linguistic segmentation, **not** embedding-based
semantic clustering. It cannot guarantee optimal discourse boundaries or resolve
every ambiguous heading. Local-hashing is a 512-dimensional lexical baseline,
not BGE or a high-quality semantic embedding model. Existing 384-dimensional
vectors are never reused for these contextual records.

Chunk IDs derive from document ID, original page, ordered block and continuation
number. Every fragment retains its source unit ID and continuation index.
Locations refer to normalized original-page boxes and ranges within the stored
page-local structured extraction, not raw PDF byte offsets. Complete ordered
blocks make that extraction reconstructible. No chunk contains another page's
body. Previous-page heading context may carry forward with `inherited=true`, the
original page ID and original heading location; it is not represented as text
printed on the new page. A failed page clears continuity context.

## Typed metadata and content types

`page_schema.py` defines nested, validated models. Missing required provenance,
conflicting identities, invalid boxes/cells, nonexistent blocks, wrong page
ranges or stale contextual text fail validation before Chroma writes.

| Layer | Preserved fields |
| --- | --- |
| Record | schema version, stable document/chunk IDs, source filename/hash, original page number, page ID, chunk sequence, content type and evidence origin |
| Processing | native/visual route, parser version, explicit model version when visual, image conversion/compression provenance and review state |
| Page | physical and printed page values; headers, footers, headings, section structure, captions, paragraphs, tables, figures and complete ordered blocks |
| Block | stable ID, parent ID, order, type, text, location/range, heading level |
| Chunk | contributing block IDs, locations, semantic boundary, continuation, overflow state, table/figure identity and structured details |
| Observation | observed/not_found/unreadable/not_applicable/unknown_legacy; typed value, extraction source, optional confidence and needs_review |

Non-observed fields cannot carry invented values. Missing confidence remains
null; confidence is not a calibrated assurance score. Native header/footer
classification is positional and marked for review. Header/footer text is retained
in page metadata and blocks, even if a consumer later suppresses it in body display.
Ambiguous heading levels are marked unreadable, not assigned invented chapter
numbers. Recognizer observation provenance is stamped with the actual configured
model version and review-required state, not blindly accepted from its output.

Chroma's flat metadata stores the full record as reversible JSON (`record_json`).
MCP restores it completely, and the Agent keeps it in evidence references and the
approval payload. A bounded controller view does not replace or delete that record.

## Tables and figures

Tables are separate `table_rows` records. Their structures retain table ID/number,
caption, all header-row levels, cells with row/column indexes and merged spans,
units, footnotes and page boxes. Large tables split by row groups. A connected
merged-cell row span is indivisible. Each group retains the full original table
structure and selected original row indexes; its retrieval text repeats headers
without pretending those headers were printed again. Ordinary paragraphs are
not concatenated into table chunks. The table-level character location covers
the source table unit; selected rows and cell boxes locate the fragment precisely.

Matching explicit table numbers produce a **candidate** cross-page association
ID with review required (confidence 0.5), not a proven continuation. Missing table
numbers are not guessed. Every table chunk stays on its original page. Explicit
continuation information is retained separately from the recognizer.

Images/charts produce separate `figure_summary` retrieval records, bound to
figure ID, original page/image hash, region, caption and model version. Structured
fields preserve observed axes, legend, units and numeric uncertainty. The prompt
forbids guessed numbers and restricts trends to visible evidence. Summaries have
`evidence_origin=generated_summary`, `source_text=null`, and `summary_text` set.
They are retrieval aids. Agent claim/calculate validation refuses to use them as
verbatim source quotations or numeric operands. Original image review remains
necessary; the mock-based workflow does not establish summary truthfulness.

## Context prefix without citation contamination

Every v2 chunk has separate `context_prefix`, `context_metadata`, `source_text`
(or `summary_text` for figures), and `retrieval_text`. The latter is exactly:

```text
context_prefix + "\n\n" + text
```

Chroma dense and BM25 retrieval, and strict hybrid snippet indexing, use
`retrieval_text`; citation checks use only the original `text`/source transcript.
MCP read also exposes `source_text` and `generated_context` separately.

`structure-prefix-v1` is a deterministic template, not an LLM call. It names the
verified document identity, page, observed heading path and inheritance, content
type, a literal caption/table label or leading source excerpt as a topic anchor,
and an explicitly stated heading question when available. Otherwise the purpose
is `unknown; not stated`. It never infers an unstated chapter or problem solved.
The prefix records its sources, method/version, absent model version, completion
state and source-text hash. Validator recomputation prevents a stale/arbitrary
prefix or retrieval text from being written with changed content. The record
version hash binds the full structure and prefix, so updates re-index changed
context even when the clause body itself did not change.

Example (fictional):

```text
Document: synthetic-example (synthetic-layout.pdf). Page: 2.
Observed heading context (hierarchy may need review): Section 2 Fictional benefits.
Content type: table_rows. Topic anchor (source label/excerpt, not inferred): Fictional benefits.
Explicit question/purpose: unknown; not stated.
```

See the [complete synthetic metadata example](examples/page-v2-metadata.synthetic.json).
It is generated from a mock fixture, contains no private source content, and is
explicitly not a verified extraction from a real policy.

## Compatibility, migration and checks

Use a separate store/collection with `--record-schema 2`. Collection contracts
bind both schema and contextual-template version. Schema 1 remains readable
without rewriting its hashes; layout/context omissions are returned as
`unknown_legacy`. Old subset-based Agent references remain valid only for unchanged
schema-1 record versions. Missing key provenance still fails. There is no automatic
upgrade inventing absent layout. Rebuild v2 from the authorized originals into a
new collection; unbound legacy vectors are rejected for contextual v2 imports.

Targeted tests cover pure-text/scanned/chart/mixed pages; pending visual pages;
page boundaries and sentence/list packing; inherited headings; required metadata;
table/header/figure separation; prefix/version updates; full real MCP round trips;
Agent quotation restrictions; and preservation/rollback of new uploads. Historical
hybrid and Agent regression suites are also run. Synthetic/mock tests are distinct
from the small existing DOL-text smoke and from still-unavailable real multimodal
quality evaluation. The 1,000-question holdout remains provisional and unfrozen.
