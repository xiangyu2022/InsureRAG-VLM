# Archived public insurance coverage and regulatory corpus

Source: [HICRIC Data](https://huggingface.co/datasets/Persius/hicric), maintained by
[TPAFS/HICRIC](https://github.com/TPAFS/hicric). Pinned dataset revision:
`e9304975feff9ccaaf15cf547f698590d00d78c3`.

The two downloaded configurations contain 4,771 records: 3,661 coverage rules and
medical policies, plus 1,110 regulatory guidance records. Exact normalized-text
deduplication merges 373 records, leaving **4,398 records and 1,115 unique source
URLs**. These are not 4,398 independent PDFs. CMS ZIP bundles can contain many
source records. The resulting categories contain 3,292 coverage records and 1,106
regulatory records. Source access dates range from January 17 to January 25, 2024.

`records.jsonl` preserves original extracted text and upstream provenance.
`rag_pages.jsonl` uses one logical record per source record; its `page: 1` is
**not a physical PDF page**. `rag_snippets.jsonl` contains **78,830** query-independent
180-word windows with a 120-word stride and source word offsets. No question or
answer label was used to select these windows. No graph edges or independent
question labels are implied by the snippet count.

Changes from upstream: normalized-text deduplication, preservation of merged
provenance, deterministic identifiers, word-window splitting, and source-bound Q/A
extraction into the separately versioned government benchmark. The upstream data
card is retained as `UPSTREAM_README.md`, and `manifest.json` records file hashes
and downloaded Parquet hashes. Historical text, missing symbols, page footers and
other extraction artifacts can remain. This is an archived research corpus, not a
representation of current coverage rules.

## Attribution and license

The curated HICRIC dataset is distributed under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This derived corpus retains that license, attribution and the original source URLs.
Original government sources retain their applicable terms. The repository's code
license does not replace the dataset's license.

## Rebuild

Install `requirements-domain-training.txt`, then run from the repository root using
fresh output directories:

```powershell
python scripts/prepare_hicric_extension.py --cache ../hicric_download --corpus data/research_corpus/hicric_public_v1 --benchmark data/benchmarks/hicric_government_qa_v1
python scripts/index_research_corpus.py --corpus data/research_corpus/hicric_public_v1 --model ../models/bge-small-en-v1.5 --device cuda --output reports/hicric_public_v1/bge_index
```

The builders refuse to overwrite artifacts. When this folder already contains the
delivered manifests, rebuild to new folders and compare content hashes. The index
uses exact cosine search. Corpus size alone is not a reason to claim an HNSW or
GraphRAG quality improvement.

The delivered index has been built: 78,830 vectors × 384 dimensions, taking about
222 seconds on the recorded GPU. Of the snippets, 543 exceed the encoder's 512-token
limit and are truncated for embedding only; their full snippet text is preserved.
Both query profiles have been exercised against this index with source-linked
results. These runs verify operability, not a labelled snippet-search accuracy.
