# Expanded frozen document benchmark v2

V2 is a separate experiment. The v1 fixtures, scripts, raw results, and scoring history remain unchanged. The target is 60 development questions and 240 test questions, with one document scope per question and document-family separation between splits. Corpus size is not independent benchmark size; questions remain clustered within their source documents.

## Authored input contract

`data/benchmarks/research_v2/` contains `dev.jsonl`, `test.jsonl`, a list in `documents.json`, the frozen `manifest.lock.json`, and a separate `corpus/` with `rag_pages.jsonl` and `rag_snippets.jsonl`. Original PDFs listed in `source_files` are also checked against their exact recorded hashes. Archived HTML records may have a single pseudo-page; that is not a physical PDF page.

Each case preserves the v1 fields `id`, `split`, `document_scope`, `question`, `answerable`, `reference_answer`, `answer_keys`, and `gold`, and additionally requires `category`, globally unique `fact_id`, and `document_family_id`. `document_scope` contains exactly one `doc_id`. A supported case needs a reference satisfying all answer-key groups and an actual source/span in that scoped document. Unsupported cases have empty gold and answer keys. The document registry identifies `doc_id`, `document_family_id`, `split`, `title`, and source provenance.

Validation rejects duplicate case IDs, normalized exact questions, or authored fact IDs; document families crossing the split; duplicate normalized evidence pages of at least 80 characters crossing the split; missing sources; altered corpus/source hashes; invalid gold spans; and incorrect declared split counts. Same-split duplicate pages are reported. Declared families and exact-page checks do not prove the absence of paraphrased duplicates or base-model pretraining exposure.

## Prepare once, then run matched models

Preparation performs the actual hybrid retrieval and evidence packing without model generation. It saves each exact prompt, retrieval ranking, case annotation, source identity, and source/code/decoding hashes. Only the frozen prompt goes to the model; reference answers and gold labels are never appended to a generation request. Both arms consume the same prepared folder, including the same inputs for failed calls.

The primary path is **retrieved only**. Oracle mode is optional and deliberately supplies gold pages; it is a separate generation diagnostic. The shared v2 defaults are an 8,192-token context, a 384-token output limit, temperature 0, seed 42, retrieval top-3, and an 8,000-character packed-context budget. The 384-token limit was selected before v2 test inference because complete supporting quotes can exceed the earlier v1 192-token budget. It applies equally to both models and both splits; it is not an observed v2 improvement. Every run records its actual settings.

```powershell
.venv/Scripts/python scripts/prepare_research_benchmark_v2.py --validate-only
.venv/Scripts/python scripts/prepare_research_benchmark_v2.py --split dev --output reports/research_v2/prepared_dev
.venv/Scripts/python scripts/prepare_research_benchmark_v2.py --split test --output reports/research_v2/prepared_test
.venv/Scripts/python scripts/eval_research_benchmark_v2.py --prepared reports/research_v2/prepared_test --model qwen3.5:4b --expected-digest 2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd --endpoint http://127.0.0.1:11435 --output reports/research_v2/my_qwen35_test
```

Use a new output folder. To run the comparison arm, pass its exact locally installed model tag and verified digest with the same prepared folder. Explicit loopback Ollama is the only inference route; the runner rejects a different installed digest. It never substitutes another model, provider, or extractive answer. Local hashing is the declared preparation baseline; a real local embedding directory can be selected explicitly during preparation and its fingerprint is frozen.

Preparation can use `--limit` only on development data. Test schedules must be complete. `--resume` appends the missing suffix of an interrupted matching run; it checks the existing prediction checksum and every prior prompt/annotation identity and never retries or drops completed error rows. A completed run cannot be overwritten or resumed. Freeze preparation only after code and annotations are ready: changing evaluation source files invalidates that protocol.

```powershell
.venv/Scripts/python scripts/compare_research_runs_v2.py --left reports/research_v2/my_qwen25_test --right reports/research_v2/my_qwen35_test --output reports/research_v2/my_comparison
```

The comparator requires full matching test schedules, fixture/protocol/code hashes, labels, ranking, actual prompt text, and decoding options. Errors retain their intended prompts and remain in the comparison. Model architecture, size, and quantization may differ; a paired deployment difference is not a causal estimate of a model-version effect.

## Metrics and uncertainty

The primary retrieval diagnostic is `gold_evidence_in_context`: a complete annotated span must occur under its own SOURCE identifier in the actual packed prompt. Page hit@k is secondary, especially when an HTML document has one pseudo-page.

The answer metric remains a strict deterministic contract: matching key, correct gold citation, verbatim quotation covering an annotated span, that quote/source actually supplied in context, and completed generation. It is not expert semantic correctness; wrong roles, negation, or extra contradictions can evade lexical checks, and adequate short quotes can fail full-span matching.

Reported separately:

- **Supported contract success:** strict successes divided by all supported questions.
- **Answer coverage:** completed, valid JSON with `abstain=false` and a nonempty answer, divided by all scheduled questions; supported-only coverage is also reported.
- **Conditional supported contract accuracy:** strict successes among the supported questions the model answered. Zero answered questions gives null, not 0% or 100%.
- **Conditional answer contract precision:** supported strict successes among all answered questions, including unsupported-question answers in the denominator.
- **Unsupported behavior:** strict completed empty-field abstention, valid non-abstaining answers, and failure to strictly abstain are separate rates. Invalid JSON, truncation, and transport failure are not counted as safe refusals.
- **Operational outcomes:** schema validity, completion, and request-error rate, with no failures dropped.

`summary.json` includes overall, per-document, per-category, and document-family sensitivity tables. Confidence intervals use 2,000 seeded percentile-bootstrap draws, resampling whole documents with replacement and recomputing question-weighted numerator/denominator ratios. Conditional metrics recompute their denominators in every draw; undefined draws are counted. Fewer than two eligible clusters yields no interval. Fewer than ten is explicitly flagged as unstable. Per-document tables therefore report counts/rates without pretending one document supplies a cluster interval.

Paired differences use the same sampled documents in both model arms. Their conditional-accuracy denominators can differ because the models answer different subsets; that is not accuracy at matched coverage. Family-cluster sensitivity is also reported when documents are related. No cluster adjustment makes this authored, non-random fixture representative of a production document population, and subgroup intervals are not corrected for multiple comparisons.
