# Next experiments

These are proposed research experiments, not results already achieved. Freeze each protocol, annotations, split, and acceptance criterion before inspecting its final test outputs. Preserve the current reports as baselines.

## 1. Verify meaning and measure risk against answer coverage

**Observed motivation.** The [constructed guard diagnostic](../reports/guard_adversarial_v1/README.md) accepted 6 of 10 deliberately false candidates. Its failures include same-page field/value swaps, changed units, reversed negation, and swapped dates. High heuristic scores did not establish correctness. Application repairs and automatically selected citations can also differ from the original model answer.

**Experiment and controls.** Evaluate a verifier that represents each claim as entity/policy, field, value, unit or basis, polarity, effective version, and supporting source. Start with independently adjudicated correct/incorrect candidate pairs from new packets; include minimal changes to a single field, number, unit, date, or negation. Keep the current lexical guard as the baseline, replay identical candidates and context, and score raw generation, citation selection, and served or repaired answers separately. Keep the ten existing counterexamples for development only. Select thresholds on development packets and plot held-out semantic error among answered questions against the fraction answered, with uncertainty estimated at packet level.

**Prospective success criterion.** On the new held-out candidates, target at least a 50% relative reduction in false acceptance while retaining at least 90% of the baseline's correctly accepted candidates. Report uncertainty and every subgroup even if the target is missed. This is a research target, not a production assurance threshold. The current ten constructed cases estimate neither a model error rate nor population reliability; passing them alone would show only regression coverage.

## 2. Test representative documents and scans with domain review

**Observed motivation.** Current fixtures contain archived public guides and short fictional packets. Expert adjudication is absent. Upload regressions establish explicit unreadable-page reporting and transactional activation, but do not establish OCR transcription accuracy; clean image diagnostics do not represent difficult scans.

**Experiment and controls.** Assemble a permitted corpus covering born-digital, scanned, and mixed PDFs, tables, rotated or blank pages, endorsements, and conflicting versions. Have two independent insurance reviewers annotate requested facts, exclusions, source spans, and whether a packet supports an answer; adjudicate disagreements. Split by document family and complete packet, preserving an exposure ledger. Compare native text extraction, text plus OCR, and a separately identified direct-image path on matched questions. Measure readable-page coverage, field transcription errors, retrieval, semantic answer correctness, and abstention separately, including each document subgroup.

**Prospective success criterion.** Require every unreadable page to be disclosed and every rejected upload to preserve the previous corpus. Require independently adjudicated labels before using a case in final scoring. Pre-register acceptable semantic-risk and answer-coverage thresholds with reviewers before testing; publish subgroup uncertainty rather than aggregate-only success. Existing upload checks and small public or synthetic benchmarks do not establish robustness on real insurance packets.

## 3. Conduct an actual Qwen3.5-4B training study

**Observed motivation.** The [0.8B LoRA smoke](QWEN35_LORA_SMOKE.md) verifies two training updates and save/reload compatibility. It does not fine-tune the served 4B model. Historical SFT rows reuse question templates, and some public benchmark documents occur in that training lineage.

**Experiment and controls.** Train a pinned 4B checkpoint on permitted training packets, with development and final test packets disjoint from all cumulative training sources. Compare the same 4B base and adapter using identical retrieval, prompts, decoding, and evaluation format; isolate any quantization or serving-format change as a separate comparison. Use development data for hyperparameters, record multiple training seeds and resource costs, audit adapter reload parity, and run the frozen final evaluation once.

**Prospective success criterion.** Require a positive paired packet-level confidence interval for semantic improvement at matched answer coverage, with no observed increase in unsupported false acceptance, and passing provenance/reload checks. Report inconclusive results as such. Compatibility, historical training loss, and current unadapted serving results do not prove useful 4B fine-tuning or adapter deployment through Ollama.

Return to the [README](../README.md) or the [current limitations](KNOWN_LIMITATIONS.md).
