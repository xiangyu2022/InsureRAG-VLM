# Observed graph research results — 2026-10-01

## Public-source Qwen3.5 test

240 completed requests, zero transport errors, 180 supported / 60 unsupported questions, 20 test documents and 13 declared document families. Each question searches its named source. Model: unadapted Qwen3.5-4B Q4_K_M, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`; BGE-small-en-v1.5 local checkpoint, exact vector search plus BM25. Actual code/model/data/options are in [run metadata](../research_v2/qwen35_final_test/run_metadata.json).

| Metric | Count | Rate | Descriptive document-bootstrap 95% interval |
| --- | ---: | ---: | --- |
| gold_evidence_in_context | 131/180 | 72.78% | 61.11%–82.78% |
| supported_context_contract_success | 72/180 | 40.00% | 31.11%–48.33% |
| supported_answer_coverage | 146/180 | 81.11% | 71.67%–89.44% |
| conditional_supported_contract_accuracy | 72/146 | 49.32% | 38.99%–59.20% |
| answer_coverage | 147/240 | 61.25% | 53.75%–67.50% |
| conditional_answer_contract_precision | 72/147 | 48.98% | 38.71%–58.60% |
| unsupported_strict_abstention | 59/60 | 98.33% | 95.00%–100.00% |
| unsupported_answer_rate | 1/60 | 1.67% | 0.00%–5.00% |
| unsupported_failure_to_strictly_abstain | 1/60 | 1.67% | 0.00%–5.00% |
| supported_strict_abstention | 34/180 | 18.89% | 10.56%–28.33% |
| retrieval_hit_at_k | 176/180 | 97.78% | 95.56%–99.44% |
| generation_complete | 240/240 | 100.00% | 100.00%–100.00% |
| json_contract_success | 240/240 | 100.00% | 100.00%–100.00% |
| request_error_rate | 0/240 | 0.00% | 0.00%–0.00% |

Answer-key matching alone passed **127/180 (70.56%)**; citation identity passed 136/180, verbatim quotation 125/180, complete annotated span 93/180, and the combined supplied-context contract 72/180. These components overlap and must not be added. Fifty-five answers matched keys but failed the joint contract. Thirty-four supported questions were declined.

The strict 40% joint score is **not** a 60% semantic error rate. The [selected failure review](selected_failure_review.json) records correct short answers rejected by unit aliases, adequate shorter quotes rejected by full-span requirements, punctuation mismatches, a substantive Coverage B error, and the unsupported dollar-deductible question answered with “2%.” No keys or labels were changed after test inference. The review is selected and AI-assisted, not a semantic-accuracy estimate.

Raw [predictions](../research_v2/qwen35_final_test/predictions.jsonl), [prepared inputs](../research_v2/final_prepared_test/inputs.jsonl), and [summary](../research_v2/qwen35_final_test/summary.json) are preserved. Cluster intervals are descriptive for authored documents; all-success strata can produce degenerate bootstrap intervals and do not imply certainty.

## Full expanded-corpus retrieval

All 65 documents searched together, no gold-document filter, 240 questions × three arms = 720 queries per corpus variant. The 180 supported questions define the metrics; no LLM runs here. Natural-language questions name their source, so this is not blind document discovery.

| Corpus | Graph mode | Page Hit@3 | MRR@3 | Full evidence in actual context |
| --- | --- | ---: | ---: | ---: |
| Frozen unlinked corpus | off | 79.44% | 0.6639 | 63.33% |
| Frozen unlinked corpus | explicit | 79.44% | 0.6639 | 63.33% |
| Frozen unlinked corpus | all | 78.89% | 0.6620 | 62.78% |
| Exploratory six-link corpus | off | 79.44% | 0.6639 | 63.33% |
| Exploratory six-link corpus | explicit | 79.44% | 0.6676 | 63.33% |
| Exploratory six-link corpus | all | 78.89% | 0.6657 | 62.78% |

The original expanded corpus contains 1,448 heuristic candidate edges and no explicit section-reference edges. The source-linked pilot adds six checked page references in three real guides. It was constructed after observing the primary results, without loading QA labels, and is **exploratory**, not a second untouched test. More graph edges do not automatically improve public-guide retrieval. Compare [frozen run](../expanded_retrieval_v1/test_frozen/summary.json) and [pilot run](../expanded_retrieval_v1/source_links_exploratory/summary.json).

## Controlled synthetic path study

96 test questions in 24 packets: 72 supported and 24 unsupported. Five arms yield 480 retrieval queries. Six scenario templates are shared with development; changed numbers/IDs do not create independent template diversity. All arms include the shared insurance reranker. No generation or abstention accuracy is measured here.

| Retrieval arm | Mean gold-page recall | Complete evidence in context |
| --- | ---: | ---: |
| bm25 | 61.11% | 33.33% |
| dense | 57.41% | 33.33% |
| hybrid | 57.41% | 33.33% |
| hybrid_explicit_graph | 100.00% | 100.00% |
| hybrid_candidate_graph | 100.00% | 100.00% |

Hybrid retrieval retained complete evidence in 24/72 supported cases; explicit path context retained it in 72/72. The 48 extra completed chains are a **constructed mechanism/regression result**. Perfect performance on these short explicit templates is not a general-policy score. Direct questions already succeed in the baseline; the constructed one-hop/two-hop categories account for the change. [Raw context/path records](../graph_paths_v1/test_frozen/predictions.jsonl) and [per-category summary](../graph_paths_v1/test_frozen/summary.json) are included.

## Iterations, provenance, and application limits

- Candidate expansion alone failed to retain complete synthetic chains. Path reservation was selected using development cases before final testing.
- Raising per-page context from 900 to 2,400 characters did not change the exact full-span development count (29/45), and the actual strict development answer count remained 23/45. This change is not claimed as a measured accuracy gain.
- The six real page mappings were added after the frozen primary runs. A final-code replay regenerated all **240 identical prompt and annotation hashes**, with zero new generation calls; see [replay audit](final_code_replay.json). Original run/code identities are not rewritten. The original implementation is in `frozen_source.zip`; final source is in the project archive.
- The new expanded-corpus demo entry point was exercised with actual Qwen generation. On the documented Basic Policy query the raw answer was correct but application serving abstained. This exposes conservative postprocessing; benchmark JSON-contract results are not application answer-delivery rates. Both raw and served responses remain in [smoke output](app_document_smoke.json). Generic glossary routing is separately labeled in [glossary smoke](app_glossary_smoke.json).
- Final local verification: **236 tests and 52 subtests** passed in the full research environment and the minimal environment. Remote CI was not run. Source audit verified all nine added PDF hashes and all 163 nonempty text pages.
- No new training, semantic accuracy guarantee, model-version causal comparison, production deployment, or GitHub push is claimed.
