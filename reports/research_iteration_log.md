# Research prototype iteration — 2026-09-20

User-requested active iteration began **2026-09-20 21:54:56 UTC**.
Minimum completion time: **2026-09-20 23:54:56 UTC**.
This log records work and observed outcomes; elapsed time is not itself validation.

## 21:54–22:05 UTC: establish an executable upgrade and evaluation plan

- Preserved the upstream clone at commit `2923b32e728387bfa788788eb731ef83c6e7e60d` and the earlier evaluation-integrity fixes on branch `codex/evaluation-integrity`.
- Hardware detected: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB VRAM, driver 596.08. No callable Ollama installation was present.
- Selected `qwen3.5:4b` via Ollama as the new local serving target. Official catalog identifies a Q4_K_M quantization (~3.4 GB); the official Qwen model card identifies Qwen3.5-4B as a vision-language model. Text-only experiments will not establish visual QA performance.
- Downloading the official standalone Windows Ollama v0.34.2 into the task workspace, with a separate local endpoint/model directory. Existing global settings are not needed.
- Parallel work: explicit model routing and provenance; truthful demo/backend behavior and embedding failures; a document-scoped, independently frozen QA development/test diagnostic.
- Historical Qwen2.5-7B training artifacts remain historical. They are not evidence of Qwen3.5 fine-tuning or an improvement in answer error rate.

Sources: https://ollama.com/library/qwen3.5:4b ; https://huggingface.co/Qwen/Qwen3.5-4B ; https://docs.ollama.com/windows ; https://docs.ollama.com/api/chat

## 22:05–22:28 UTC: real inference, development failures, and retrieval diagnostics

- Started a task-local Ollama endpoint at `127.0.0.1:11435`; downloaded and ran actual `qwen3.5:4b` and the unadapted `qwen2.5:3b` comparison artifact. Model digests and request options are retained in each run's metadata.
- Froze 12 development and 24 test questions before inference. Ran development retrieval and oracle-evidence generation. Kept the 24 test questions out of the tuning loop.
- Initial Qwen3.5 development retrieval found the gold page on 7/8 answerable cases; 6/8 answers matched keys and 5/8 met the strict quotation/source contract. All four unsupported responses met the strict empty-field refusal contract. These small development numbers are diagnostic.
- Inspected an omitted `$1,500` fact and revised general context packing to preserve relevance order, deduplicate contained snippets, retain source headers, and allocate page budgets. Key matching improved to 7/8; strict quotation success stayed 5/8.
- Downloaded the actual pinned BGE-small checkpoint. Independently checked CLS embedding parity (maximum absolute difference 0.0). BGE still retrieved 7/8 development gold pages and did not improve the small development answer result over the explicit hashing baseline.
- Sent actual PNG pixels from a public Delaware PDF page to Qwen3.5. The first four-question smoke had three numerically correct but fenced-JSON responses; native schema output fixed the contract on the second four-question smoke. This is an image-interface development check, not a visual accuracy benchmark.
- Audited all 3,850 current SFT records with the historical pinned tokenizer: no prompt-prefix mismatch and no answer truncation at 1,024 tokens (maximum formatted length 487). Fixed the general prompt-only truncation failure independently of that corpus result.

## 22:28–22:42 UTC: application and synthetic packet audit

- Made the active public-reference/uploaded-policy corpus visible, added source-file identities to index compatibility, and stopped automatic corpus selection from silently using an unrelated global snapshot. Old index schemas now require rebuilding.
- Ran 24 synthetic packet cases through the actual application answering path, preserving raw model and served answers. The first Qwen3.5 run gave the expected fact/source on 16/16 supported cases but the application served only 8/16 because of overbroad guards. The raw versus served discrepancy is a regression target, not evidence to hide.
- Independent manual review found all eight unsupported raw responses declined the requested conclusion. The automated raw-refusal score of 5/8 missed three wording variants; it must not be described as three hallucinations corrected by the app.
- Froze scoring version 2 with explicit checks that the cited source and quotation were actually present in the supplied prompt. Preserved old raw results and wrote separate rescored outputs.
- Began downloading an isolated CUDA PyTorch environment and pinned Qwen3.5-0.8B weights for a two-step language-only LoRA compatibility smoke. The 0.8B training check is separate from 4B serving and cannot establish training quality.

## 22:42–22:56 UTC: executed training compatibility and froze the test comparison

- Actually completed Qwen3.5-0.8B BF16 LoRA on the RTX 4070: two synthetic optimizer steps, 2,705,664 trainable parameters across186 language linear modules,372 adapter tensors changed, finite gradients. Peak allocated CUDA memory was2,075,731,968 bytes. Fresh-base reload produced bitwise-identical evaluation logits.
- A second run saved portable canonical base/revision metadata and verified saved-tokenizer parity before forward parity. An independent CPU-only audit matched all372 loaded adapter tensors exactly to the saved artifact and authenticated all10 local base/tokenizer files against official metadata for the pinned revision. A first audit's strict no-CUDA guard failed because PEFT inferred a device; that failed report remains, and the corrected explicit-CPU audit passed without initializing CUDA.
- Fixed a Windows CLI UTF-8 failure found by redirecting a real JSON query containing public PDF Unicode. Rebuilt and queried a curated index in the minimal environment; the command now emits valid UTF-8 JSON. The conservative extractive response abstained; this is an execution check, not a correctness pass.
- At22:49:17 UTC froze model order, data, prompts, decoding, scoring, and32 source files. Saved `test_protocol_v1.json` and `frozen_source_v1.zip` before first test inference. Ran both48-request arms (24questions ×retrieved/oracle) with matching actual prompts/code/options and no transport errors.
- Retrieved strict context-grounded contract: Qwen2.5-3B4/16, Qwen3.5-4B9/16. Frozen answer-key matching:13/16 versus12/16. Strict unsupported output contract:2/8 versus8/8. These are small diagnostic results; semantic review is separate and scores are not tuned after test exposure.
- Executed the frozen12-case image-only counterfactual diagnostic:9/9 supported keys and3/3 strict refusals, with changed numeric and exclusion values across three same-layout images. No OCR text or expected answer was supplied in prompts. Clean synthetic images do not represent the diversity of real scanned policies.
- Re-fetched all six official benchmark PDFs and checked24 annotated spans against physical pages:23 normalized exact matches; one development span differed only by a straight/curly apostrophe, confirmed in the original page text. Corpus, labels, and existing scores remain frozen.
- CPU upload audit discovered missing-OCR crashes, blank-page failures, zero-text success reports, and failed files poisoning later uploads. These application/ingestion issues are being repaired after the frozen test; the saved frozen source bundle preserves the precise evaluated code.

## 22:56–23:19 UTC: source review, application repair, and insurance design audit

- Independently reviewed all 96 frozen public responses and preserved original scores/hashes. Complete retrieved short-answer correctness is 13/16 for both models; Qwen3.5 has a correct negative answer missed by one frozen alias. The old model's six strict refusal failures are not six hallucinations.
- Repaired staged upload activation, missing-OCR handling, blank pages, partial-ingestion reporting, and zero-text rejection. Actual handler checks show scanned upload HTTP422 without retaining the failed file, followed by a successful readable upload and correct extractive answer. Mixed readable/scanned files retain explicit coverage warnings.
- Ran both real models through the synthetic packet suite, preserving v1/v3 and one pre-inference aborted run. Reviewed all 48 v3 raw/served pairs; traced correct-number/wrong-auto-citation, refusal-repair, and truncation failures.
- Added citation origin and evidence-supported fallback selection; preserved legitimate refusals before repair. Replayed the exact recorded public-guide response that was incorrectly rejected and fixed general wording normalization, preserving source/minimum-versus-maximum negative controls.
- Audited the insurance prospective SQL: found and fixed fractional-second truncation at the 14-day boundary. Clarified the outcome as any policy bound by the assigned customer, not attribution to an index quote. All 15 experiment/estimator tests pass. No observed retail number or insurance outcome was changed.

## 23:20–23:30 UTC: final application-default inference and portable delivery

- Ran a fresh 24-case arm for each model with the pre-existing application defaults (4096 context,384 output tokens). All paired prompts/code/fixtures match. Qwen2.5 served15/16 supported cases; Qwen3.5 served16/16; both served8/8 empty refusals, without truncation or runtime errors. All48 outputs were separately reviewed. Old D02's correct primary amount plus wrong filename/multi-page paragraph remains a withheld answer, not promoted to a pass.
- Restarted the actual UI with repaired code, uploaded the public18-page Delaware auto guide, asked the funeral-benefit question and verified `$5,000` with model-provided page5 citation, expanded page preview and retrieval trace. A second actual structured call preserves the successful trace alongside the earlier failed trace.
- Restored the insurance quote-recovery resume project; retained explicit prospective sample assumptions and the completed64,000-person retail RCT validation. Re-rendered the one-page PDF with Poppler and inspected the final layout.
- Prepared source/patch/insurance/optional-adapter archives. First reverse patch check exposed a host-global PDF text-conversion driver; added binary asset attributes and disabled text conversion during patch generation. The revised binary patch passes reverse application checking and archive CRC checks.
- Added10 deliberately false candidate-answer counterexamples to assess the postprocessor independently of model/retrieval. It falsely accepts6, including field/value, unit, negation and date swaps. Published all counterexamples and retained this limitation rather than describing heuristic scores as correctness probabilities.
- Began independent extraction, CPU test, frozen-manifest and forward-patch validation in isolated directories. No remote CI result is claimed.

## 23:30–23:44 UTC: independent reproduction and final provenance controls

- Independent ZIP and forward-applied patch trees passed204 tests plus52 subtests in the documented CPU environment without Torch/Transformers/PEFT. The first cleanroom attempt found a mocked visual test inheriting an Ollama-disabled environment; the test now scopes its mocked provider explicitly. Both the failure and successful retake were preserved outside the source tree.
- Independently extracted and fully reran the insurance analysis from the cached public CSV. All225 numeric metric values match exactly; all11 other generated files are byte-identical. SQL totals, power calculations and15 tests passed. This is not a fresh-OS installation claim.
- Exposed unvalidated original model output as plain text in a collapsed browser panel, distinguished citation origin, and corrected the answer-repair label. Verified the actual `$5,000` response and its separate raw source line in the browser.
- Found that filename-only PDF preview URLs could show a new upload's pixels for an old answer. Added opaque, session-local bindings to the question-time corpus and PDF SHA256, checked even on cache hits. Actual HTTP before/after reproduction and six regressions cover same-name replacements, a concurrent upload, changed bytes, invalid roots/tokens and missing identities. Full CPU suite:210 tests plus52 subtests. The public uploaded guide preview actually loads506×782 pixels through the new bound endpoint.
- Ran the four unique frozen image questions with images removed, matching original text/schema/model digest/decoding. Three outputs refuse semantically; one guesses a “War” exclusion without evidence. Only one satisfies strict empty-field refusal. Independent review preserves the distinction and raw results; the original12-image run is unchanged.
- Added a prospective research agenda for field/value semantic verification, representative expert-reviewed scan/packet evaluation, and a controlled4B fine-tuning study. These are proposed experiments, not achieved results.
