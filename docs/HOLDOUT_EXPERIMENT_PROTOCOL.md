# Independent holdout experiment preparation

As of October 16, 2026, construction has reached 958 provisional TEST and 58 DEV
items across 178 TEST document families. Accepted counts are zero; the benchmark
is not frozen. The [aggregate checkpoint](../reports/holdout1000_v1/stage258_progress.json)
records the current counts, validation and remaining work. Historical exposed
results are regression evidence only.

The [prospective protocol](../reports/holdout1000_v1/evaluation_protocol_v1.json)
registers BM25, original dense, original/adapted hybrid and original/adapted
reranked retrieval. All share source-only documents and a fixed encoder. A
reviewed inclusion manifest, deterministic chunking and support-unit mapping
must be completed before DEV scoring. Candidate generation must not use gold
document filters, reference answers or evidence-only chunks.

DEV selects an arm by family-macro complete-evidence@5, with the registered
tie-breaks. Freeze the DEV decision, model/tokenizer hashes, corpus, code,
context packing, prompt and review rubric before any TEST inference. At least
1,000 independent TEST and every existing quality gate must pass before freezing
TEST. A design document is not a selection lock or benchmark acceptance.

The holdout_experiment module implements annotation-based metrics and uncertainty:

- score_retrieval reports chunk Recall@5, Hit@5, chunk completeness and nDCG@10.
  Chunk completeness is diagnostic; necessary support units define the primary
  completeness measure. Refusals have no answer-retrieval score. No padding.
- evidence_unit_metrics requires each necessary support unit. Equivalent alias
  chunks are alternatives; crossing a chunk boundary can require multiple chunks.
  The function does not make semantic judgments or construct the annotations.
- paired_family_delta resamples whole document families with paired questions.
  It reports question-macro uncertainty and an equal-family point estimate.
  Ratio-metric intervals, such as refusal precision, still need orchestration.

The whole-source manifest gate requires an explicit include/exclude review for
every snapshot document before tokenization. Raw/text hashes for all 1,426 sources
were verified. Current dispositions are 301 included, 198 excluded after source review
and 927 pending. Review order prioritized DEV sources after a source-only
inventory of all 291 TDI documents. It does not authorize a gold-filtered corpus.
Existing reviews can be reused when their content hashes and scope still apply.

holdout_corpus validates source ranges, hashes, token limits and reviewed atomic
spans. The 697 partial chunks preserve all included characters and have at most
512 tokens and 2,400 characters. Source headings, lists, table headers and
qualifiers must survive chunking. Oversized atomic spans fail instead of being
silently truncated. Source structure, never an answer key, defines boundaries.

For a reviewed large table, complete row atoms may carry copied source header
spans. Headers remain exact, hash-bound source text and count toward both encoder
and character limits. The primary text partition and evidence offsets remain
unchanged; retrieval and reranking consume the header-enriched representation.
This source-only relationship needs explicit review. It does not establish gold
unit necessity or automatically clear repeated headers as support alternatives.

All 58 DEV items and 76 declared evidence spans now map by exact offsets. The
reviewed draft has 62 necessary units for 47 answerable cases. Eleven refusal
cases retain gap-context diagnostics only. Exact text matches in another source
are leads; they do not establish equivalent support across product scope or
license restrictions. Eighteen Spanish support aliases have passed unit-specific semantic review; one
compound guaranty-membership alias was rejected. Global alias review and
whole-DEV clearance are unfinished.
The real readiness check therefore still stops before model loading.

scripts/run_holdout_dev_retrieval.py accepts the complete source snapshot,
source-only manifest, label-free DEV request JSON and separately reviewed DEV
clearance. --check-only reports blockers without loading models. Actual inference
requires pinned local models and a private output directory. Exact-hash resumes
reuse saved successes and failures, preserving intended denominators. A completed
resume loads no models. Failed queries remain in later scoring denominators.

The six-arm CLI passed a real-model synthetic integration using four invented
sources and one invented query, plus a no-model resume. Candidate unions remain
within 200 per arm with stable score/ID ties and shared cross-encoder pairs.
This was not a real DEV/TEST run. Shared execution timings do not establish
standalone per-arm p95 for selection. Scoring and selection-lock orchestration
remain unfinished.

Generation is separate from retrieval. The inherited baseline, trained query
encoder and trained reranker have verified hashes. The requested Qwen3.5-4B runtime
remains unavailable at the checked local paths/endpoints; its current location is
needed. No download, replacement model, real DEV comparison or TEST inference has
been performed. Generation receives only questions and retrieved evidence, never
reference answers, task labels, formulas or gold spans.

Final evaluation must retain failures, distinguish citation-ID validity from
claim support, measure evidence after packing, report actual timings/tokens, and
use the preregistered paired 100-case semantic-review subset. Label that review
Codex content review, not expert review or whole-test semantic accuracy. Undefined
ratios stay null. Bootstrap whole families and ratio numerators/denominators as
registered, rather than assuming independent questions.

Validation passed 534 tests and 58 subtests over 56 explicitly selected top-level
files. Individual QA, source text, annotations, traces and weights stay local;
only aggregates, source-family metadata, code and tests are published. No source
hold, task quota or acceptance threshold was relaxed. PR8 remains draft.
