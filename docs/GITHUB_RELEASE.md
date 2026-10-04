# October 3, 2026 research release

Code, small fixtures, compact reports and tests are versioned in Git. Large data,
per-query score/prediction files, trained weights and vector indexes are published
as [GitHub Release assets](https://github.com/xiangyu2022/InsureRAG-VLM/releases/tag/research-prototype-2026-10-03).
They retain the exact bytes of the previously verified local research deliveries.

## Restore the full research artifacts

Clone this repository into a dedicated parent directory. From the clone, run:

```sh
python scripts/restore_research_assets.py
```

This standard-library-only script downloads approximately 3.73 GB in seven ZIP
files, validates their SHA-256 checksums, and applies them in the recorded order.
It preserves all Git-tracked files, so historical archives cannot replace current
code or documentation. Missing data files go under the repository; model weights,
source downloads and archived helper scripts go in sibling paths in its parent.
Use a dedicated parent directory with sufficient space for downloads and extracted
artifacts. Existing generated artifacts at these paths are restored to their
recorded versions. Base BGE weights, Ollama models and Python/CUDA runtimes are
separate downloads documented in the reproduction guides.

You can reuse already downloaded archives:

```sh
python scripts/restore_research_assets.py --archives-dir ../research-release-downloads --offline
```

`--dry-run --offline` validates the archives and previews extraction without writing
research artifacts. A normal clone is sufficient for offline unit tests; restoring
all assets is required for the complete trained retrieval/reproduction workflow.

The archive order and hashes are in [research-assets.json](releases/2026-10-03/research-assets.json).
The manifests alongside it retain per-file hashes, source terms and the selected
checkpoints. Earlier manifests saying `github_pushed: false` record their original
creation-time state; they are not rewritten during publication. Archived README
hashes refer to the archived README, not this newer GitHub presentation.

## Data and model terms

Preserve each source's original attribution and license. The repository's code
license does not relicense third-party data or pretrained models. FinQA's pinned
source declares MIT; FiQA attribution and CC-BY-SA-4.0 terms, InsuranceQA and SQuAD
source terms, and individual government/policy-document notices remain applicable.
Source registries, upstream identifiers and provenance records accompany the data.

## Results and limitations

- [Current retrieval results and failures](EVIDENCE_RERANKER_UPDATE_ZH.md)
- [Full reproduction procedure](EVIDENCE_RERANKER_REPRODUCTION.md)
- [Tasks and primary-metric recommendations](TASKS_AND_METRICS_ZH.md)

This release includes research retrieval checkpoints and a separate Qwen3.5 serving
prototype. It does not establish improved end-to-end generated-answer accuracy,
production readiness, or a general GraphRAG/HNSW advantage.
