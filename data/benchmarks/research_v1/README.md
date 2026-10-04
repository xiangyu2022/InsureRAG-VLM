# Frozen document-scoped research benchmark v1

This small research benchmark asks questions about archived **public consumer guides**, not issued customer policies. Author: **AI-assisted/manual-agent-verified; no human-expert adjudication**. The annotating agent read the selected page text and recorded exact spans; tests verify span inclusion and answer keys. The benchmark was frozen before model inference using `manifest.lock.json`.

## Composition and evidence

- Development: 12 questions from the Delaware auto and homeowners guides (8 answerable, 4 unsupported).
- Test: 24 questions from the Maryland auto/homeowners and North Carolina disability/travel guides (16 answerable, 8 unsupported).
- Each document contributes four answerable questions, one ordinary missing policy-specific value, and one unrelated private fact. Synthetic questions contain no real customer information.
- `documents.json` enumerates every available curated page in each scope. Citation page numbers are physical PDF page numbers, which may differ from printed page labels.
- The data snapshot is the existing 201-page public curated corpus. All scoped pages and snippets remain available to retrieval; filtering never uses the gold page or answer.
- Development and test documents do not overlap. However, these public documents occur in the old SFT data. This is **not** a holdout from that training history. Comparisons must use unadapted Ollama models, with possible unknown public pretraining exposure acknowledged.
- Archived statements are evaluated as document contents, not as current law or advice.

Earlier template questions such as “What does the evidence say about exclusions?” are underdetermined across the full corpus: many documents and pages are legitimate answers, while an exact-source metric arbitrarily treats just one as correct. Merely generating 39 such questions does not establish 39 unique source-grounded targets. Here the named-document scope and a specific fact identify the target; distractor pages inside the document remain. Questions about “my policy” deliberately cannot be answered by substituting a public worked example.

## Separate experimental paths

`retrieved` uses the repository's actual `hybrid_pipeline.DocumentRetrievalPipeline.query_with_ranking` method and its context packing. A benchmark-only JSON generation contract returns `answer`, exact `evidence`, exact `source`, and `abstain`. It does **not** test the application's `query_structured` repair or UI postprocessing.

`oracle` is an explicitly separate generation diagnostic: supported cases receive full gold pages. Unsupported cases receive the first three scoped pages, chosen without gold labels; those consumer guides cannot establish private customer values. Oracle scores do not count as retrieval success. Oracle prompt length may differ from retrieved context and is recorded; this is diagnostic, not an otherwise identical end-to-end treatment.

Both models receive the same system prompt, JSON schema, per-mode evidence, temperature 0, seed 42, context window, and generation token cap. Thinking is disabled and any returned thinking length is logged. The runner requires an exact model tag/digest and an explicit loopback endpoint; no paid API or silent fallback is allowed. Local-hashing is the explicitly labeled default embedding baseline; a local embedding checkpoint path is also supported.

## Metrics and limits

- Retrieval hit@k: a supported gold page occurs among the retrieved pages. Document scoping does not itself count as a hit.
- Answer-key match: all required boundary-aware strings/numbers occur in the short answer. $2,500 is distinct from $25,000; $2,500.00 is equivalent.
- Citation match, verbatim quote inclusion in the cited page, and inclusion of the annotated gold span are separate checks.
- Grounded-key pass requires the four checks jointly. This is a conservative deterministic contract score, **not semantic correctness or a production insurance error rate**. Valid paraphrases or shorter valid quotes can fail; a matching key can coexist with a semantic mistake. Inspect recorded responses.
- Strict unsupported abstention requires `abstain=true` and empty answer/evidence/source. Malformed JSON or backend failures do not pass.
- Wilson 95% intervals report small-sample uncertainty. They treat questions as independent; shared documents violate that simplification. These intervals do not establish population generalization.
- Latency includes the local request/model load and inference; per-document index construction is excluded from query latency and recorded separately. Cold and warm calls are not interchangeable.
- No human expert ratings, production workloads, cost-savings claims, or real claim/coverage decisions are represented.

The 16 answerable and 8 unsupported test cases are too few to justify precise reliability claims. Keep test fixtures frozen; iterate only on development. If evaluation reveals an annotation issue, preserve the original result and issue a new benchmark version before making any new test comparison.

## Run

```powershell
python scripts/eval_research_benchmark.py --validate-only --split test --output reports/research_v1/validation
python scripts/eval_research_benchmark.py --mode retrieval-only --split dev --output reports/research_v1/retrieval_dev
python scripts/eval_research_benchmark.py --endpoint http://localhost:11435 --model qwen2.5:3b --split dev --mode both --num-predict 192 --timeout 600 --output reports/research_v1/qwen25_dev
python scripts/eval_research_benchmark.py --endpoint http://localhost:11435 --model qwen3.5:4b --split dev --mode both --num-predict 192 --timeout 600 --output reports/research_v1/qwen35_dev
```

After development and an independent fixture review, freeze the final pipeline/config and run the same commands with `--split test` and new output directories. Record CPU/GPU/hardware and run order separately. No inference is executed by fixture construction or validation.

Outputs include request-level predictions, prompts and sources, inference token/duration metadata, model digest, code/prompt/data hashes, summary JSON, and index-build durations. Partial `--limit` runs are allowed only on development; existing prediction directories cannot be overwritten.
