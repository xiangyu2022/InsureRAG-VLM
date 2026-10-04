# Synthetic PDF packet stress diagnostic v1

This diagnostic exercises the actual **PDF ingestion → retrieval → local model → `query_structured` repair/citation/abstention** path. It is AI-authored synthetic regression data, not real issued policies, an insurance-expert assessment, or held-out production accuracy.

`packets.json` contains readable page text and document metadata; `cases.jsonl` contains 24 manually specified questions and expectations. Seed 42 generates twelve distinct invented amounts. Six packets become ten PDFs with thirteen physical pages, using deterministic PyMuPDF construction. `manifest.lock.json` freezes the text/expectations before model inference. The generated PDF checksums are recorded separately for each run. No actual customer identity or private insurance data appears.

| Packet | Cases | Stress condition |
| --- | ---: | --- |
| A | 4 supported | Separate dwelling, personal property, deductible, and liability numbers |
| B | 3 supported, 1 unsupported | Actual declarations versus an explicitly labeled worked example; missing annual premium |
| C | 2 supported, 2 unsupported | Conflicting policy identifiers on a claim report; absent settlement approval and policy dates |
| D | 3 supported, 1 unsupported | Explicit dated endorsement versus historical base terms; absent earthquake limit |
| E | 1 supported, 3 unsupported | Unsequenced conflicting versions; ask for abstention on controlling terms, but permit a fact about an explicitly named version |
| F | 3 supported, 1 unsupported | Near-duplicate camera/laptop/jewelry schedule pages; absent aggregate limit |

The expected abstentions are information-sufficiency judgments about invented text. They do not certify legal rules for interpreting real endorsements. Some packets explicitly state that a value is missing, so these are easy abstention probes. The mismatch case additionally lacks an approved payment. Each question searches all pages in its packet, including distractors; the runner never selects a gold page as input. Because packets are tiny (two or three pages), this is primarily an ingestion/generation/postprocessing diagnostic, not a challenging large-corpus retrieval benchmark.

## Execution

```powershell
# Actual PDF generation and loader/extraction validation; does not contact a model.
python scripts/eval_packet_stress.py --validate-only --output reports/packet_stress_v1/validation

# Parent-reviewed execution after configuration selection.
python scripts/eval_packet_stress.py --endpoint http://localhost:11435 --model qwen3.5:4b --num-predict 192 --timeout 600 --output reports/packet_stress_v1/qwen35
python scripts/eval_packet_stress.py --endpoint http://localhost:11435 --model qwen2.5:3b --num-predict 192 --timeout 600 --output reports/packet_stress_v1/qwen25
```

Use a new output directory for each run. A local embedding checkpoint can be selected with `--retrieval-model`; the default is explicitly `local-hashing`. The model tag is exact, requests use an explicit loopback endpoint, and model digest stability is checked. All generation options are fixed and logged. No custom benchmark JSON prompt is injected: the runner uses the application's actual default prompt and model client. Prompt capture is observational only. It does not make an additional model request.

The comparison of `raw` and `served` values is from **the same model call**: `raw_answer` precedes deterministic repair; `answer`, `abstain`, and `citations` are the served application result. A repair success must not be credited as a model-generation success. Model errors, truncation, prompts, repairs, support flags, conflict flags, backend information, data/code hashes, and generation parameters remain available in the output.

## Metrics

- Answer-key match checks boundary-aware values/strings and excludes detected abstention. It is not semantic correctness and can miss attribute reversals or contradictory wording.
- Citation match requires the expected physical page, normalizing only directory prefixes, filename case, and integer page formatting. A nearby page never substitutes for the required page.
- `supported_key_source_pass` combines answer keys and expected sources.
- `context_key_source_pass` additionally requires a completed, non-truncated generation, citations present as actual `SOURCE:` headers in the captured prompt, and the reference keys present within the cited supplied sections. It prevents credit for a memorized number that was absent from the prompt, but it is still not an entailment test.
- `retrieved_evidence_key_source_pass` separately checks reference keys in the cited retrieved page's snippet. This can distinguish a repair using retrieved evidence from raw generation supported by the smaller packed context.
- Raw abstention uses the application's explicit-abstention phrase heuristic; served abstention uses the actual returned boolean. Wording heuristics can misclassify qualified answers. Inspect responses rather than claiming semantic abstention accuracy.
- Eight unsupported cases assess abstention; sixteen supported cases assess keys and citations. Wilson intervals are descriptive only: authored cases share six packets and are not a random sample.

Validation checks the complete page text after actual PDF extraction, synthetic provenance metadata, source filenames, and physical page counts. A rendered contact sheet was reviewed during development: no clipped/overlapping text. The sparse page layout is intentional for reliable text ingestion; this does not test scanned PDFs, OCR, tables, or visual reasoning.

Keep all model outputs immutable. If a synthetic expectation is found incorrect, preserve the original run and create a new version. This diagnostic is intentionally available for regression-driven iteration and must never be advertised as a hidden test set.
