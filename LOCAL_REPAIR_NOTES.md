# Local research prototype upgrade

Base commit: `2923b32e728387bfa788788eb731ef83c6e7e60d`
Local branch: `codex/evaluation-integrity`
Date: 2026-09-20. No commit or GitHub push was made.

The source archive and binary patch include the implementation, frozen fixtures,
raw runs, failed development runs, and interpretation notes. Git history, Python
environments, generated indexes, uploaded files, and model weights are excluded.
The optional two-step compatibility adapter is packaged separately.

## What actually ran

- Real local Qwen3.5-4B Q4_K_M inference through Ollama 0.34.2 on an RTX 4070 Laptop
  with 8,188 MiB VRAM. Exact model digests and request options are saved per run.
- Frozen public test: 24 questions × retrieved/oracle evidence × two unadapted
  models = 96 responses. All 48 paired prompts and decoding settings match.
  Qwen2.5-3B versus Qwen3.5-4B strict context-grounded retrieved passes were 4/16
  versus 9/16; manual short-answer correctness was 13/16 for both. This is not a
  general error-rate reduction or comparison against the old trained 7B adapter.
- Final application-default synthetic packet regression: 24 cases per model,
  4096 context / 384 output tokens, with raw and served answers kept separately.
  See `reports/packet_stress_v1/app_v4_review.md` for semantic review and caveats.
- Image-only counterfactual diagnostic: 9/9 supported answers and 3/3 refusals
  across three clean invented declaration images. No OCR or answer text in prompts.
- Actual Qwen3.5-0.8B LoRA compatibility smoke: two invented records, two optimizer
  steps, 2,705,664 trainable parameters, 1.93 GiB peak allocated CUDA memory,
  exact save/reload forward parity. This is not useful task fine-tuning.
- Independent CPU audit: 372 serialized adapter tensors match loaded tensors;
  all 10 local base/tokenizer files match official pinned Hugging Face metadata.
- Actual public-PDF upload and browser question: 18/18 readable pages; funeral
  benefit `$5,000` with model-provided page 5 citation. Earlier false abstention
  and the corrected trace remain under `reports/application_integration/`.

## Implemented repairs

- Explicit model/provider selection, no silent model or embedding substitution,
  native non-thinking/schema options, truncation detection, image provenance.
- Corpus-bound, content-hashed index identities; document folders cannot silently
  query an unrelated snapshot. Staged uploads activate only after indexing succeeds.
- Missing OCR and blank/scanned pages produce visible coverage counts and warnings;
  a zero-text upload is rejected without poisoning later uploads.
- Citation previews bind to the answer-time document snapshot and SHA256. Same-name
  replacements cannot redirect old answers to new page images; unbound requests fail.
- Relevance-preserving context packing; exact citations; conservative numeric,
  refusal, source identity and conflict checks. `citation_origin` distinguishes
  model citations from evidence-selected citations. Raw answers remain available.
- Consistent retrieval labels/aliases, comparable evaluation manifests, cumulative
  SFT record/prompt/document/parent lineage, explicit evaluation-row rejection,
  prompt-only truncation rejection, and document-overlap gates for held-out claims.
- Separate key/quote/page/context/refusal metrics with denominators and descriptive
  Wilson intervals. Empty groups are null. Token F1 is never called an error rate.

## Reproduce and interpret

Install `requirements-dev.txt` in a separate Python 3.11+ environment and run
`python -m pytest -q` with `INSURERAG_USE_OLLAMA=0` for CPU tests. The locally
verified suite currently has 210 tests plus 52 subtests. Remote CI has not run.
See `docs/REPRODUCIBILITY.md` for exact commands and tested dependency snapshots.

The public test protocol/code were frozen before first test inference; the archive
`reports/research_v1/frozen_source_v1.zip` preserves that source. Later application
guard repairs are evaluated as development regression and do not rewrite the public
test. Model size/artifact, small sample size, shared documents, AI-assisted labels,
unknown pretraining exposure, and heuristic semantic checks limit the findings.
Read `docs/KNOWN_LIMITATIONS.md` before making performance claims.
