# Publisher-written government FAQ transfer test

This benchmark contains **416 questions, 416 candidate answer texts and 37 source
records/URLs** from the archived HICRIC regulatory guidance configuration. CMS,
Medicaid and Department of Labor publishers wrote the Q/A content. The build script
extracts matching numbered `Qn`/`An` sections; it does not generate questions with an
LLM. Every included question and answer has exact source-character offsets.

All 416 examples are evaluation-only. They were frozen before trained model
inference and are not used for training, checkpoint selection or fusion tuning.
The trained checkpoint is selected on InsuranceQA validation first. Transfer search
uses **27,829 answer candidates**: all 27,413 InsuranceQA answers plus all 416
government answers, shared by every arm. Gold answers are not inserted into query
candidate lists; BGE and BM25 must retrieve them normally.

The labels mean original publisher Q/A pairing, not independent expert adjudication
of every potentially relevant alternative. The extraction filter rejects unmatched
numbering, ambiguous duplicate questions, some explicitly context-dependent
questions, and answers outside 25–350 words. Abbreviations, historical references,
footers and context dependence can remain. The rejected-section audit is included.

All source spans were checked. The final 11,729 training questions have zero exact
question overlaps and zero near overlaps under the registered character TF-IDF
threshold of 0.92; the highest similarity is approximately 0.680. This check cannot
exclude semantic overlap or exposure during the base models' public pretraining.

Report the source-URL-cluster bootstrap interval as the primary paired uncertainty:
416 questions from 37 sources are not 416 independent source documents. This is
archived FAQ answer retrieval, not a medical/legal advice or multi-page policy
reasoning benchmark. No evaluation result should be described as Qwen accuracy.

Attribution: [HICRIC Data](https://huggingface.co/datasets/Persius/hicric), revision
`e9304975feff9ccaaf15cf547f698590d00d78c3`, derived from original publisher materials
identified in every record. This extracted dataset retains
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and original-source
attribution; see the [corpus data card](../../research_corpus/hicric_public_v1/README.md).
