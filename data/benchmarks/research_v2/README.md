# Expanded public-document benchmark

This version contains 300 source-inspected questions: 60 development and 240 test.
The test split contains 180 supported questions and 60 questions whose answers
cannot be established from the selected document. Each document contributes nine
distinct supported fact targets and three evidence-near unsupported questions.
The 24-question `research_v1` test and its results remain unchanged.

The source registry distinguishes physical PDF pages from archived HTML text
records. Nine newly acquired agency PDFs add auto, renters, health, long-term
care, property claims, flood, and life-insurance material. Sixteen archived HTML
documents supply five development and eleven test documents. A web record with
`page=1` is not a physical PDF page. See `documents.json` for provenance, family
assignments, source URLs and exact PDF hashes.

Development and test documents and declared document families do not overlap.
The six `research_v1` documents are excluded. Several related test documents
share a family, so the number of independent families is smaller than the
number of test documents. Source spans and reference/answer-key consistency were
checked before test inference. This is AI-assisted
annotation, without human insurance-expert adjudication or a probability sample
of real customer questions. Public pretraining exposure is unknown; the legacy
HTML sources were present in the project's earlier SFT corpus. The evaluated
Ollama models are unadapted and do not load that SFT adapter.

Questions concern the **archived source as written**, including dated examples,
not a representation that every statement remains current law or describes a
reader's own issued policy. Source errors, stale guidance and extraction defects
can remain. Answer and evidence keys are deterministic contract checks; they do
not prove semantic correctness or establish a production error rate.

The corpus uses a fixed 180-word window with a 120-word stride, without selecting
chunks using gold labels. Retrieval searches the entire named document. For
HTML, source/page hit is largely trivial; the primary retrieval diagnostic is
whether the annotated evidence span is actually present under its source in the
packed prompt. The primary answer diagnostic also requires a completed JSON
response, the expected answer keys, a valid source, and a verbatim quote covering
the annotated evidence in the supplied context. Valid shorter quotations and
unlisted paraphrases can fail this strict contract.

The October 2026 graph revision evaluates Qwen3.5-4B with temperature 0, seed
42, context 8,192 tokens and a 384-token output budget. The runner also supports
matched model comparisons, but a new Qwen2.5 comparison is not part of this revision. The larger output budget
was chosen before inference to accommodate the expanded source-quotation task.
Report strict supported success, answer coverage, conditional contract accuracy,
unsupported abstention, malformed/truncated outputs and transport failures
separately. All 240 scheduled test questions stay in their relevant denominators.
Confidence intervals resample complete documents; family-cluster intervals are
also supplied as a sensitivity analysis. They are descriptive for this authored
fixture and do not make it representative of production traffic.

Use `scripts/prepare_research_benchmark_v2.py --validate-only` to verify the
fixture, then prepare prompts once before generation. Exact commands and model
digests are recorded with the results in `reports/research_v2`.
