# Local Chroma clause database and read-only MCP

The clause service uses **persistent local Chroma**, plus the project's existing
BM25 implementation and reciprocal rank fusion. It runs over **stdio only** and
exports exactly `search_clauses` and `read_evidence`. It does not start an HTTP
listener, register a remote client, request credentials, fetch source URLs, or
generate answers. Existing PDF/UI and historical NumPy evaluation entry points
remain compatible; they are not silently redirected to a different corpus.

This is a new executable knowledge-service entry point, not a claim that an
external MCP host has already been connected. Protocol tests launch real local
MCP client/server processes. Their fictional plumbing fixtures establish software
behavior, not insurance answer quality or real-model performance.

## Install and run

Run these commands from the repository root in a project virtual environment:

```powershell
python -m pip install -r requirements.txt -r requirements-knowledge.txt
python -m src.insurerag_vlm.clause_store_cli --records private/clauses.jsonl --store .knowledge/chroma
python -m src.insurerag_vlm.clause_mcp --store .knowledge/chroma
```

`private/clauses.jsonl` is an operator-supplied, provenance-reviewed input, not a
file distributed by this change. The importer prints counts and the embedding
contract. The server reserves stdout for MCP messages and waits for a stdio
client. An empty/missing database is an error; it is never populated implicitly.
The default collection is `insurance_clauses`; `--collection` selects another
operator-configured collection. `.knowledge/` is ignored by Git. Keep databases,
raw sources, vectors, and per-clause files local and outside published artifacts.

Dependencies are pinned to `chromadb==1.5.9` and the official maintained MCP v1
SDK `mcp==1.30.0`. MCP v2 is now stable; this implementation deliberately pins the
supported v1 low-level API and tests the same installed version, rather than
mixing v2 examples with v1 imports. See the [official v1 branch](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x).
Installation downloads SDK dependencies, not embedding/generation model weights.

## Embeddings and retrieval

The default is the explicit **local-hashing baseline**, 512 dimensions, SHA-256
token counts and L2 normalization. **It is not BGE.** Chroma is always opened with
`embedding_function=None`; all embeddings are provided explicitly. Telemetry is
disabled in the local Chroma settings. There is no cloud fallback or automatic
model download.

An already-present local transformer checkpoint can be selected using `--model
<directory>`, with `--pooling`, `--query-instruction` and `--max-length` as needed.
Use exactly the same options for import and serving. The existing embedding
implementation records checkpoint hashes, dimension, pooling, normalization and
query instruction. Collection metadata binds that full contract; a mismatched
backend/configuration fails on reopen. Query instructions apply only to queries.
Unsupported/zero vectors fail explicitly; the hashing baseline's tokenizer is
primarily English/alphanumeric and is not a multilingual semantic model.

Chroma uses cosine distance. BM25 reuses `SparseRetriever.build`, an in-memory
version of the existing file-backed BM25 builder. Equality filters apply to both
channels. Candidate sets use up to `max(4 * top_k, 20)` results per channel and
equal-weight RRF with k=60. RRF scores are ranking scores, not confidence or a
coverage decision. A cached sparse representation is reused for the same filter.
This local baseline reads the selected collection into memory for BM25; it is
not a distributed or large-scale sparse index.

## Input and evidence identity

Each JSONL record requires:

| Field | Meaning |
| --- | --- |
| `record_id` | Globally unique stable snippet ID, preserved verbatim |
| `doc_id` | Existing document ID, preserved verbatim |
| `text` | Clause evidence text, unchanged |
| `page_number` | Nonnegative integer; 0 explicitly means unspecified |
| `citation` or `source` | Existing human-readable evidence locator |
| `source_sha256` or `source_hash` | Lowercase SHA-256 of the original source artifact, **not** its snippet |

`page_id`, `parent_page_id` or `page_key` is preserved as the page identity. If
absent, an explicit document/page key is constructed; no offset is invented.
Existing `char_start`/`char_end`, source URL, jurisdiction, publisher, effective
dates, bounding boxes, and other JSON metadata are retained. Offsets must be a
valid ordered integer pair when supplied. The operator remains responsible for
verifying that the supplied source hash, scope and quoted evidence match the
original artifact; import validation does not replace source-quality review.

`evidence_id = SHA256(record_id UTF-8)` is the opaque database key. `text_sha256`
binds the imported text and `record_sha256` binds the full normalized record.
Supplied text hashes are checked; conflicting stable IDs across documents are
rejected. Source edits preserve the evidence ID and change its version hash.
Clients can pass the search result's `record_sha256` into `read_evidence` to
detect a changed version rather than silently using stale evidence.

Only the operator import command writes records. Unchanged records are skipped
without embedding work. Changed records are upserted under the stable ID;
omitted IDs are retained, **not deleted**. All input rows and vectors are checked
before writes. Large imports use batches of 200; an interrupted multi-batch
import can leave a partial update, so rerun the same input before serving. It is
idempotent, not a cross-batch transaction. Chroma's [upsert semantics](https://docs.trychroma.com/docs/collections/update-data)
support updating existing IDs and adding new IDs.

A process lock permits one store owner at a time. Stop the stdio process before
updating its database, then restart it. This prevents readers from serving an
in-progress import and keeps the BM25 cache coherent. OS process exit releases
the lock. There is no multi-user authorization layer; use a private directory and
a trusted local operator/client. MCP read-only annotations are backed by the
actual absence of write/file-access tools, not treated as access enforcement.

## Tool contracts

`search_clauses`:

```json
{"query":"waiting period for dental crowns","top_k":5,"filters":{"doc_id":"your-document-id","page_number":2}}
```

`query` is a nonblank string of at most 2,000 characters. `top_k` is an integer
from 1 to 20, not a string or boolean. Optional equality filters are restricted
to `doc_id`, `page_number`, `source_sha256`, `jurisdiction`, and `publisher`.
Multiple fields are ANDed. Arbitrary Chroma operators and unknown keys are
rejected. The implementation uses Chroma's documented [metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).
Results contain full stored evidence/provenance, dense/BM25 ranks and RRF score.
No matches return `{"status":"ok","results":[],...}`.

`read_evidence`:

```json
{"evidence_id":"<64 lowercase hex characters from search>","expected_sha256":"<record_sha256 from search>"}
```

It returns `status=found` with the stored record, `status=not_found` for a valid
missing ID, or `status=stale` for a version mismatch. Invalid parameters produce
an MCP tool error. IDs are never resolved as paths or URLs; traversal strings,
`file://` values, client-supplied paths, and unknown arguments are rejected.

## Migrate an existing index without re-embedding

The existing hybrid pipeline's JSONL records and NumPy vectors can be reused:

```powershell
python -m src.insurerag_vlm.clause_store_cli --records data/hybrid_snippets.jsonl --vectors data/hybrid_snippets_dense.npy --manifest data/hybrid_embedding_manifest.json --provenance private/source_provenance.jsonl --store .knowledge/chroma
```

The optional provenance sidecar contains records keyed by the existing
`record_id`, adding missing source hashes/locators. It cannot overwrite existing
fields and rejects duplicate or unknown IDs. The vectors must have exactly the
same row order and length as the metadata, finite values, and the configured
dimension. The old manifest's embedding fingerprint must match exactly. With
these checks, vectors are passed directly to Chroma without re-embedding. Use
the appropriate `--model` options for a pre-existing BGE/local-transformer index;
never relabel it as hashing. Migration does not modify the old files.

The older `index.npy` + `index_meta.json` format may omit text, source hashes and
stable snippet IDs. It is **not sufficient on its own**. Supply verified,
aligned enriched records and the corresponding embedding manifest, or rebuild
from the authorized source using the existing ingestion pipeline. Missing
provenance is an explicit import error, not fabricated by migration.

The Python `ChromaRetrieverAdapter(store)` implements the existing Agent
retriever interface. It returns provenance-bound evidence and can be injected
into `ReviewAgent`; use a corpus-version-specific `runtime_id`. Mandatory human
approval and its SQLite checkpoint flow are unchanged. The fixture-based Agent
demo is still explicitly synthetic; no generation service is started here.

## Validation and remaining boundaries

```powershell
python -m pytest -q tests/test_chroma_knowledge.py tests/test_hybrid_pipeline.py tests/test_embedding_backend.py tests/test_agent_graph.py
```

Tests cover a separate importer process, two separately restarted MCP stdio
server processes, protocol initialization/tool discovery/calls, strict parameter
rejection, missing/stale evidence, filters, idempotent/updated writes, embedding
contract errors, BM25 compatibility, and migration of actual existing-pipeline
output. Fixtures are fictional and temporary. CI installs the knowledge SDKs
through `requirements-dev.txt` so these tests do not silently skip.

No external MCP application has been registered or granted access. Client-specific
registration remains a separate step; its configuration must launch the project
Python with `-m src.insurerag_vlm.clause_mcp`, repository-root working directory,
and an explicitly chosen local store. No network listener is needed.

Formal TEST/DEV/training files and source-review checkpoints are untouched.
The 1,000-question holdout remains provisional, accepted=0 and unfrozen. Source
clearance blockers are not solved by putting data into Chroma. No synthetic
customer-policy module, paid API call or model download is part of this change.
