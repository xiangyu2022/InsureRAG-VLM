# Content-bound citation preview regression

Two synthetic PDFs both named `policy.pdf` contain collision deductibles of $735
and $1,820. The old preview implementation resolved a filename against the current
active upload. After replacement, an old chat citation therefore displayed the
new PDF page.

The corrected API returns a session-local opaque `preview_url` and
`preview_sha256` with each served citation. Its registry retains the answer's
captured corpus folder, page number, and PDF hash. Rendering requires the recorded
path to remain within that snapshot and an approved document root, verifies the
bytes even on image-cache hits, and renders those same checked bytes. It does not
fall back to the current upload. Tokens expire with the server session; deleted
or changed files have no preview. A source-only request now returns HTTP 400.

The [reproduction script](run.py) performs actual local HTTP uploads, chat dispatch,
and PNG requests. Answer construction is an explicit controlled fixture, with no
model inference or answer-quality claim. Its before check executes only the two
original preview methods extracted from the unchanged frozen source archive.
The [initial attempt](run_01/attempt.md) documents why the local-extractive answer
path was unsuitable for isolating this preview regression.

The current [recorded result](run_03/results.json) confirms:

- The original unbound lookup changes the old citation to the new page pixels.
- The bound old citation retains the old pixels, including registration after
  the active upload changes; the new citation has different PDF and pixel hashes.
- An unbound request returns 400; a mismatched PDF hash returns 404 even when an
  image was already cached. No opaque token is included in the saved report.

The six [unit/integration regressions](../../tests/test_citation_previews.py) also
cover an upload completing during answer generation, changed file bytes,
cross-snapshot/root paths, malformed or unknown tokens, and registry isolation.
The preview contract establishes byte provenance, not whether an answer is
supported by those bytes.

Reproduce in a fresh output directory from the repository root:

```powershell
python reports/citation_preview_v1/run.py --output-dir reports/citation_preview_v1/retake
python -m pytest tests/test_citation_previews.py -q
```
