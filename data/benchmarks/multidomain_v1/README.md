# Source-separated mixed-domain retrieval fixture

Frozen before model inference on 2026-10-01. Checksums and upstream identities are in `manifest.lock.json`; extraction/filter decisions are in `audit.json`.

| Domain | Train questions | Validation | New test | Source grouping |
|---|---:|---:|---:|---|
| Government Q/A | 213 | 127 | 81 | 28 / 14 / 14 canonical URLs |
| General SQuAD paragraphs | 7,999 | 400 | 600 | 442 / 24 / 24 Wikipedia titles |

The answer file contains 20,556 candidate paragraphs/answers, including extra unlabelled distractors. The experiment combines these with 27,413 InsuranceQA answers. The test population is **not 681 insurance questions**: it comprises 81 government and 600 general questions.

Government pairs retain source record IDs, original URLs and character offsets into the pinned HICRIC archive. All canonical URLs from the earlier 416-question government test are excluded. Publisher Q/A extraction and exact alignment do not replace expert annotation; context and footnote quality flags are retained in the experiment report.

SQuAD questions are the original human-written questions, at most one per paragraph. Gold retrieval paragraphs are derived from mechanically verified original answer spans. Original offsets belong to the original SQuAD paragraph; two training questions map to a punctuation-normalized duplicate paragraph with different canonical offsets, documented in the source audit. This task does not report official SQuAD span exact match or F1.

The final training pipeline additionally excludes all paragraphs from new held-out URLs/titles from **both positive and negative** pairs (`data/training/retention_v2`). The evaluation index still contains all candidates. The initial v1 training-negative fixture is not eligible for model selection.

Sources: [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) and [HICRIC](https://huggingface.co/datasets/Persius/hicric), CC BY-SA 4.0 with original source attribution. Derived records preserve Wikipedia/government links; this README does not replace the upstream terms. The combined InsuranceQA portion retains its separate research-only restrictions. Historical government content is not current policy advice. Public checkpoint pretraining exposure is unknown.
