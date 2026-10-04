# Reproducible generation runtime

This project now separates the configured generator from installed models and available credentials. The default is the CPU `local-extractive` baseline. A real model runs only when explicitly selected. A missing model or failed request is an error; another local model, a hosted API, or the extractive baseline is never substituted.

## Runtime observed on 2026-09-20

The following identity was read from the task-local Ollama service at `http://127.0.0.1:11435` using `GET /api/tags` and `GET /api/version`:

| Field | Observed value |
|---|---|
| Ollama version | `0.34.2` |
| Requested model | `ollama:qwen3.5:4b` |
| Resolved model | `qwen3.5:4b` |
| Model digest | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` |
| Quantization | `Q4_K_M` |
| Downloaded model size | 3,389,983,735 bytes |
| Parameter size reported by Ollama | `4.7B` |
| Advertised capabilities | completion, vision, tools, thinking |
| Application default context | 4,096 tokens, including generated tokens |
| Public-guide / packet diagnostic context | 8,192 tokens, recorded separately in each run |

The registry tag is named **4b**; the local artifact reports **4.7B**. Report the tag and digest when identifying this quantized deployment. The artifact size is disk storage, not measured VRAM use. The advertised 262,144-token model context is not the configured context for this experiment. The [official Ollama tag](https://ollama.com/library/qwen3.5:4b) identifies the same tag and quantization family.

Runtime identity checks establish which model is available. They do not establish answer accuracy, training improvements, or production readiness. Real generation outcomes must be cited from a saved evaluation report, separately from CPU mock tests.

## Explicit selection and options

For the existing CLI or demo, set:

```powershell
$env:INSURERAG_VLM_MODEL = "ollama:qwen3.5:4b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11435"
```

Port 11435 is the task-local service; a standard Ollama installation usually uses 11434. `OLLAMA_MODEL` no longer overrides an explicit model. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `HF_API_TOKEN` do not activate a generator by themselves. Set `INSURERAG_VLM_MODEL=local-extractive` for the reproducible CPU baseline. Embeddings default independently to `local-hashing`, with explicit selection through `INSURERAG_RETRIEVAL_MODEL`.

For an experiment, use per-run configuration and pin the digest:

```python
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline

config = ModelConfig(
    vlm_model="ollama:qwen3.5:4b",
    ollama_base_url="http://127.0.0.1:11435",
    vlm_expected_digest="2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
    vlm_thinking=False,
    ollama_generation_options={
        "temperature": 0.0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 384,
        "top_k": 40,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
    },
)
pipeline = DocumentRetrievalPipeline(config)
print(pipeline.vlm_client.backend_metadata())
```

Defaults use greedy decoding for a stable local comparison, not the model author's general-purpose sampling recipe. Environment overrides for temperature, seed, context, and output length are read once when the client is created. Explicit `ollama_generation_options` take precedence. Numerical kernels, hardware, runtime updates, and model changes can still affect outputs; a fixed seed alone is not a cross-platform reproducibility guarantee. The [Ollama parameter reference](https://docs.ollama.com/modelfile) describes these options.

Hosted generation requires an explicit selection such as `openai:model-name`, `anthropic:model-name`, or `hf:organization/model` and its matching credential. Recognizable legacy `gpt-*`/`claude-*` model names still select their corresponding provider. A credential for another provider never changes the selected route. Unknown unqualified names are rejected.

## Actual request and response contract

The Ollama generator sends a non-streaming `POST /api/chat` request with the exact model tag and two messages. Non-thinking mode uses the **top-level** `think: false` API field. A `/nothink` string in the prompt is not used. This follows the [Ollama chat API](https://docs.ollama.com/api/chat); the [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-4B) also distinguishes an API parameter from the older prompt-based switch.

The client checks the installed digest before every generation and checks the model name in every response. `backend_metadata()` records the selected provider, model identity, quantization details, service version, endpoint, generation parameters, thinking setting, token counts, timings, completion reason, and whether output was truncated. It never records credentials or reasoning text. Per-generation metadata is isolated across request threads.

`generate_chat(..., response_format="json")` and `generate_with_images(..., response_format=schema)` can request Ollama's native structured output. A schema is sent as the top-level `format` field and saved in per-response metadata. Unsupported providers reject this option. The client returns the actual text; it does not remove Markdown fences or relabel an invalid result as valid JSON.

Structured pipeline results distinguish:

| Field | Meaning |
|---|---|
| `raw_answer` | Generator or extractive output before citation validation and repair |
| `generation_used` | Whether a real model was invoked for this query |
| `answer_repaired` | Whether evidence-based postprocessing replaced the generated answer text |
| `citation_origin` | `model_source` for a model's accepted source, `evidence_selection` for a page selected after validating the served claim, or null when neither applies |
| `explicit_abstention` | Whether a transparent wording heuristic detected an explicit decline for insufficient evidence |
| `generation_truncated` | Whether the server stopped at the output token budget |
| `answer_backend` | Actual answer path: generator, local extraction, retrieval abstention, or deterministic evidence repair |
| `backend_metadata` | Runtime snapshot for this query; no stale generation metadata for an extraction-only result |
| `abstain` / `citation_support` | Existing heuristic evidence-gate decisions, not a probability that the answer is factually correct |

Report raw model quality and the final guarded pipeline separately. A repaired answer must not be counted as an unassisted model success. A citation selected by the application is not a correct citation originally produced by the model, even when the served claim is supported. Model failures must not be silently dropped from evaluation denominators.

Explicit declines are preserved through postprocessing: for example, “the supplied guide does not state your own deductible” cannot be replaced by a generic $500 example from that guide. A response marked truncated is retained as `raw_answer` for research but the application abstains with `abstain_reason="generation_truncated"`. Structured numeric fields require a matching field type/coverage; missing fields remain null rather than taking the first unrelated amount on a page. These checks are deliberately conservative and are not a general factual entailment test.

## Dense-index identity

The local embedding model has an independent identity from the generator. `ModelConfig` exposes `retrieval_pooling`, `retrieval_query_instruction`, and `retrieval_max_length`, and the retriever records checkpoint file hashes plus its resolved pooling, normalization, length, and query-prefix contract.

New hybrid indices persist `hybrid_embedding_manifest.json` with that fingerprint, corpus path/hash, and vector shapes. The loader validates the configured encoder before using either disk artifacts or its in-memory cache. Legacy dense indices use an adjacent `.manifest.json` sidecar. Missing manifests, a changed checkpoint/pooling/query contract, or inconsistent vector shapes require a rebuild; an existing directory is not evidence of embedding compatibility. Explicit rebuilds clear cached source documents so source edits are incorporated.

## Text, images, and historical training

The default document-RAG path sends retrieved **text** to the generator. The model has a vision capability, but that alone does not make this path a measured vision-language system. Existing lightweight image retrieval features describe page layout; they are not the Qwen vision encoder.

A separate, explicit image-QA API is implemented:

```python
from pathlib import Path

answer = pipeline.vlm_client.generate_with_images(
    "Read the collision deductible from this declarations page. "
    "If not visible, state that the evidence is insufficient. SOURCE: sample-policy.pdf#page=1",
    [Path("sample-declarations.png")],
)
print(answer)
print(pipeline.vlm_client.backend_metadata()["last_generation"])
```

This method sends local image bytes as base64 in the user message's `images` array, following the [Ollama vision API](https://docs.ollama.com/capabilities/vision). It checks vision capability and supports one to four PNG/JPEG/WEBP images with explicit size limits. Missing/corrupt files, unsupported providers, and non-vision models raise errors instead of dropping images. Per-generation metadata records image count, SHA-256, dimensions, and format. Normal `generate()` calls remain text-only; no document image is uploaded automatically. Image-QA results require a separate saved evaluation and must not be mixed into a text-only benchmark. This direct method returns raw model text, without the document pipeline's citation validation or repair.

### Observed image-API smoke, not a held-out benchmark

Two preserved development runs used the same four author-selected questions on a single rendered page from an archived public auto-insurance guide. Both used the exact Qwen3.5 digest above, non-thinking mode, an 8,192-token context, and a 192-token output limit.

| Run | Native schema requested | Strict JSON plus expected-key checks | What happened |
|---|---|---|---|
| [v1 raw results](../reports/vision_smoke/qwen35_v1/results.json) | No | 1/4 | Three numerical responses contained the expected figures but were wrapped in Markdown fences; the strict parser rejected them. |
| [v2 raw results](../reports/vision_smoke/qwen35_v2/results.json) | Yes | 4/4 | All four returned parsable schema-shaped JSON with the expected numerical key or explicit abstention. |

This is an integration fix demonstrated on reused development cases. It does not estimate general visual accuracy, an error-rate improvement, OCR robustness, retrieval quality, or performance on unseen documents. The image hash, prompts, unmodified responses, token counts, completion reasons, and timings are retained with each run. The checks validate expected keys, not full insurance semantics.

The repository's historical QLoRA training used **Qwen2.5-7B-Instruct**, a different model and artifact. Selecting Qwen3.5 through Ollama does not migrate, retrain, or apply that adapter. Historical eight-record spot checks remain diagnostics and do not support a new Qwen3.5 error-rate claim.

### Modern training compatibility smoke

`scripts/smoke_qwen35_lora.py` is a separate, bounded text-only training experiment for `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`. It leaves the historical training path available. This script does not train the served Qwen3.5-4B model or load its output into Ollama.

The experiment pins Transformers 5.17.0 and uses its `AutoModelForImageTextToText` mapping to `Qwen3_5ForConditionalGeneration`, matching the [official checkpoint architecture](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json). It feeds only text tensors; the vision tower remains frozen. Transformers 5.17 also supports loading the Qwen3.5 text backbone with `AutoModelForCausalLM`, so incompatibility must be assessed against the installed version rather than inferred from the multimodal model name. See the [versioned Transformers documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5).

The script uses an existing local Hugging Face snapshot and records its file hashes. `--preflight` validates that snapshot, the exact chat-template token prefix, and complete assistant-only supervision without loading model weights or using CUDA. Two synthetic examples are defined in the script; benchmark development/test data is never loaded. Each complete conversation must fit within 256 tokens. The GPU path performs two AdamW updates on rank-4 language-only LoRA modules, checks finite losses/gradients and changed adapter tensors, saves the adapter/tokenizer/provenance, then loads a fresh base plus the saved adapter and compares all forward logits on one synthetic row. It records the declared comparison tolerance, exact equality, timings, library versions, and peak CUDA allocation/reservation.

Use a separate CUDA environment with BF16 support and no optional FLA, causal-conv1d, Triton, or quantization package. The reference DeltaNet path uses standard PyTorch operations and may consume more memory and time than optimized kernels. The 0.8B checkpoint has approximately 1.75 GB of BF16 base weights before training activations and runtime overhead; fitting the complete experiment remains an empirical check. The default CPU test environment needs none of these GPU dependencies.

Passing this smoke establishes training/save/reload compatibility only. Two updates do not demonstrate loss improvement, useful fine-tuning, generalization, visual training, or an error-rate reduction. Old Qwen2.5 adapters are never accepted by this script; its only adapter reload is the one just produced from the pinned Qwen3.5 base.

The [preserved v1 CUDA report](../reports/qwen35_lora_smoke/run_v1/report.json) passed on the RTX 4070 Laptop GPU: 2,705,664 trainable adapter parameters, two finite-gradient updates, 372 changed adapter tensors, 1.93 GiB peak PyTorch allocation, and bitwise-identical full forward logits after adapter reload. The 6.448-second elapsed time excludes tokenizer/hash preflight. [The detailed smoke note](QWEN35_LORA_SMOKE.md) records exact dependencies, scope, and the v2 portability/tokenizer checks added after reviewing v1.

The [v2 run](../reports/qwen35_lora_smoke/run_v2/report.json) passed those additional checks. An [independent CPU audit](../reports/qwen35_lora_smoke/run_v2_cpu_audit_explicit_device/audit_report.json) verified exact loading of all 372 serialized adapter tensors and authenticated all ten base model/tokenizer/config files against the fixed official Hub revision. This audit performed no forward pass or training and did not initialize CUDA.

A serving comparison against `qwen2.5:3b` would compare two quantized deployments with different architectures and parameter counts. It would not isolate model generation as a causal factor, nor demonstrate the effect of the historical 7B fine-tuning run. Hold prompts, source evidence, decoding settings, evaluation labels, and failure accounting fixed, and identify each artifact by digest.
