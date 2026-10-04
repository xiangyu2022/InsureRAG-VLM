# Known limitations and interpretation boundaries

The [domain-training update](DOMAIN_TRAINING_UPDATE_ZH.md) trains a MiniLM reranker
on 11,729 cleaned questions and raises historical test Hit@10 to 73.40%, compared
with the previous pipeline's 69.55% and BGE's 67.50%. Selection uses validation;
the historical test was inspected in prior work. Only one training seed was run.
Forty cases regress versus the previous pipeline, including numeric/qualification
confusion; 10 of 12 domains improve, one ties, and the 17-question critical-illness
subset loses one hit. The new 416-question government transfer test uses 37 public
source clusters and mechanical publisher Q/A labels, not expert adjudication.
Its raw results, including regressions, are retained separately. Larger HICRIC
snippet inventory is not additional independent labelled QA, and snippet-search
accuracy is not measured by the government answer-candidate benchmark.
Specifically, government transfer Hit@10 is 95.91% after training, below the original
public reranker's 97.12% and BGE's 96.15%. The explicit general query profile remains
available; it is not an automatically selected or newly validated routing policy.

The earlier [retrieval-fusion revision](RETRIEVAL_FUSION_UPDATE_ZH.md) improves historical
InsuranceQA Hit@10 from 67.50% to 69.55% with a validation-selected cross-encoder
pipeline. The test had previously been inspected. It is not a new blind or
expert-adjudicated policy QA test. BM25 adds only one net Hit@10 case beyond
BGE with the same reranker, and two domains have small regressions. This revision
does not rerun Qwen generation or establish a GraphRAG advantage.

The October 1 self-audit also found unresolved PDF-demo issues: plan-specific
questions can route to the glossary, version comparisons select the first two
PDFs rather than requested identities, citation amount checks do not establish
plan/amount binding, and graph context reservation can evict relevant pages.
The separate FAQ ranking improvement does not repair those issues.

The latest [data-scale revision](DATA_SCALE_RESEARCH.md) adds 1,228 source PDF pages, an external 2,000-question retrieval test, and 154 real-reference navigation test cases. Retrieval, navigation and the historical 240-question Qwen generation test have different populations and endpoints. The new FAQ data has research-only use terms, 8 exact split overlaps and 476 lexical similarity flags. Navigation shares its literal extraction process with graph construction and is an exploratory integration diagnostic, not expert-adjudicated QA or graph accuracy. No new generation-accuracy claim is established by this revision.

The earlier [graph/data revision](GRAPH_RESEARCH_PROTOTYPE.md) and [historical measurements](../reports/graph_research_v1/RESULTS.md) retain the 240-question public test, 96-question synthetic graph test and original six-link pilot. Public-guide graph benefits are not established by synthetic path completion. The full-quotation score has documented false negatives, including correct shorter answers and missing unit aliases.

The sections below describe historical evidence retained on 2026-09-20. A passing test checks its stated contract; it does not establish production insurance accuracy. Historical reports and frozen annotations remain unchanged when a later audit identifies a limitation.

## Data and evaluation scope

- The curated inventory has 3,850 SFT rows, but its 3,400 answerable rows use only 39 distinct question templates, each associated with multiple pages. These are evidence-conditioned training examples, not 3,400 independent retrieval questions. Global exact-source retrieval is underdetermined for repeated generic questions. The [supervision audit](../scripts/audit_sft_supervision.py) and [historical evaluation audit](../reports/evaluation_audit_2026-09-20.md) describe this distinction.
- The public-guide fixture has 12 development and 24 test questions. The document scopes are disjoint between those splits, but some test documents occur in the repository's old SFT pool. The comparison uses unadapted Ollama models; unknown public pretraining exposure remains possible. An old trained adapter must not inherit the fixture's holdout claim. See the [fixture review](../reports/research_v1/fixture_review_v1.md).
- These annotations were authored and reviewed with AI assistance, without independent insurance-expert adjudication. Public guide contents are assessed as archived statements, not as current law or an individual's issued policy. The North Carolina travel scope has only three pages, making retrieval hit@3 an easy case when all pages are returned.
- The [packet stress fixture](../data/benchmarks/packet_stress_v1/README.md) uses six fictional packets and 24 questions. Its short, clean documents and explicit missing facts are useful regressions, but do not represent the distribution of real policy packets, disputes, or difficult refusals.
- Per-question Wilson intervals are descriptive. Questions share documents, image templates, or packets, and the reported intervals do not model that dependence. They do not support population-level reliability or significance claims.

## Scores are specific contracts

The [matched frozen-test comparison](../reports/research_v1/frozen_comparison/comparison.md) shows a mixed result. In retrieved mode, Qwen3.5-4B passed the strict context-grounded contract on 9/16 supported cases versus 4/16 for Qwen2.5-3B, and completed strict abstention on 8/8 unsupported cases versus 2/8. However, answer-key matches decreased from 13/16 to 12/16 and citation matches from 12/16 to 11/16. A single claim of improved answer accuracy would conceal these different outcomes. Model family, size, and quantized artifact all change, so this is a deployment comparison, not a causal estimate of a version upgrade.

The deterministic checks have both false positives and false negatives:

- A correct number with the wrong semantic role, an incorrect qualification, or reversed attributes can satisfy a lexical key. The frozen HO-3 question is a documented example: it asks which coverage is open-peril versus named-peril, but the key does not fully verify that assignment.
- A correct short quote or paraphrase can fail an exact full-span requirement. The development hurricane answer containing “typically 2% of the property's value” can support the requested percentage while failing the longer frozen quote contract.
- Some annotated spans need adjacent text for their subject or negation. Exact quote presence is not, by itself, entailment. The v2 scorer additionally verifies that the reported source and quote occur in the actual supplied context, but cannot fully judge meaning.

Keep the [scoring version](BENCHMARK_SCORING.md), denominators, raw outputs, and context with every result. The benchmark's direct JSON generation contract differs from the application's answer checks and evidence repair. Scores from those paths should not be merged. The application retains raw-versus-served provenance because an evidence-repaired answer is not the original model response. Heuristic confidence and abstention thresholds are not calibrated correctness probabilities.

A separate [constructed adversarial guard diagnostic](../reports/guard_adversarial_v1/README.md)
demonstrates this gap directly: the current helper and served-answer path both accept
6 of 10 deliberately false candidates, including same-page field/value swaps, unit
substitution, reversed negation, and swapped dates. The candidates and evidence were
hand-authored; a fixed ranking isolates postprocessing and no model or retrieval ran.
This is a counterexample set, not an estimated model error rate. Accepted heuristic
scores remain high, which is why they must not be shown as correctness probabilities.

Application-selected citations (`citation_origin=evidence_selection`) must also
remain separate from the model's original citation correctness.

## Distinct model and training artifacts

| Artifact | Verified scope | Not established |
| --- | --- | --- |
| Unadapted Qwen3.5-4B Q4_K_M through Ollama | Local text serving, explicitly selected image input, frozen diagnostics | A fine-tuned 4B model, a production error rate, or deployment of the old adapter |
| Unadapted Qwen2.5-3B through Ollama | Matched local comparison arm | Performance of the historical Qwen2.5-7B adapter |
| Historical Qwen2.5-7B adapter | Archived A40 training logs and small development spot check | Reproducible weights in this clone or clean held-out improvement; the eight-example check overlaps the training lineage |
| Qwen3.5-0.8B rank-4 LoRA smoke | Two synthetic examples, two finite CUDA updates, save/reload parity, independent tensor/source audit | Useful fine-tuning, held-out quality, vision training, Qwen3.5-4B training, or Ollama adapter compatibility |

The [0.8B experiment](QWEN35_LORA_SMOKE.md) uses a pinned conditional-model loader and a separate dependency environment. Its vision tower is frozen and training inputs are text only. Reload parity is established within each run; identical training trajectories across runs are not asserted. PyTorch peak allocation excludes other processes and is not total GPU memory. Installing the current optional GPU requirements does not reconstruct the old A40 environment.

## Image input, OCR, and application scope

Normal document RAG generates from extracted or OCR text. Enabling layout/image retrieval signals or rendering PDF pages does not automatically send those pixels to the answer model. The explicit `generate_with_images` interface and its schema output are a separate path.

The public-page vision smoke uses four questions on one page. The [image-only counterfactual diagnostic](../data/benchmarks/visual_counterfactual_v1/README.md) changes nine target facts across three clean, fictional declarations images and adds three missing-value cases. Its question-only text and transmitted image hashes provide a check that image input is active. Neither experiment establishes OCR accuracy, dense-table interpretation, robustness to rotation/handwriting, or performance on real scans. The counterfactual test does not exercise retrieval or application repair.

The [image-removed control](../reports/visual_counterfactual_v1/no_image_control_review.md)
keeps the four unique question forms, system prompt, schema, model digest and
decoding settings but supplies no image. Three outputs decline semantically and
one invents a “War” exclusion; only one refusal satisfies the strict empty-field
contract. This small mechanism control is not a population rate, but demonstrates
that the model can guess an unsupported term when visual evidence is absent.

Scan ingestion depends on the separate local Tesseract executable; installing the Python wrapper alone does not supply that executable. The ingestion code and [upload regressions](../tests/test_ingestion_upload.py) distinguish readable text, blank pages, OCR failures, and partial ingestion. OCR is not a fallback to the VLM image API. Check the ingestion report before interpreting an answer from a partly readable packet: omitted pages can contain relevant exclusions or endorsements. OCR output itself has no measured transcription-quality guarantee in this release.

The HTTP application is a local, single-user research demo. Public reference guides do not establish the user's deductible or coverage; an upload may also be a general guide rather than a personal policy. Intent and document-type guards reduce obvious mistakes but remain heuristic. The demo is not a production service with a validated security, privacy, concurrency, or decision-governance design.

## Reproduction and packaging

- Ollama model tags can change. Exact result identities are the recorded digests, code and fixture hashes, prompts, and decoding settings. The application default context is 4,096 tokens; the saved research runs use their separately recorded settings. A model name or default command alone does not reproduce a historical run.
- `requirements-dev.txt` deliberately omits Torch/Transformers and GPU libraries. Local checkpoint embeddings and tokenizer audits need their separate dependencies; Qwen3.5 training uses the isolated pinned smoke environment. Anthropic is another optional SDK, not a dependency of the default CPU/Ollama path.
- Hosted provider branches were not exercised in these local experiments. The HF generation and embedding branches currently use the legacy `api-inference.huggingface.co/models` URL, whereas the [current official HF documentation](https://huggingface.co/docs/inference-providers/main/en/providers/hf-inference) specifies the router endpoint. Their presence in code is not evidence of a working current hosted deployment. Optional ColQwen2/ColPali retrieval is also outside the verified default path.
- Runtime models, source-cache PDFs, generated indexes, and adapter binaries are intentionally excluded from the source package. Compact reports retain their hashes and may include the original machine's absolute paths. The adapter tensor audit requires the corresponding generated adapter and pinned base, not just its JSON report. Public URLs can change; the archived text and checksums define the old experiment.
- Frozen fixture PDFs and PNGs under `data/benchmarks/` are test inputs and must remain in a source archive. Their original manifest always enforces exact hashes. Separate visual-fixture regeneration tests require byte repeatability within one environment and preserved text, facts, page/image dimensions, and PDF-to-PNG pixels; they do not require regenerated bytes on another OS to match the frozen originals. PyMuPDF/Pillow are pinned in `requirements-dev.txt`, but native renderer builds can still differ across platforms. The broader CI matrix is configured, but a local Windows pass is not proof that remote jobs have run.
- Evaluation and smoke commands intentionally refuse to overwrite existing results. Use a fresh output directory when reproducing a bundled run. Preserve original predictions and write rescoring or source-audit results to a separate directory.

The remaining work needed for broader claims is representative data, independent domain review, richer semantic scoring, measured scan/OCR coverage, and a genuinely fresh training/evaluation protocol. Those are research requirements, not accomplishments implied by the current diagnostics.
