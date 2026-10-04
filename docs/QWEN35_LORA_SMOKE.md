# Qwen3.5 text-only LoRA compatibility experiment

This experiment checks that a current Qwen3.5 checkpoint can take supervised gradient updates and that its saved adapter can be reloaded correctly. It is separate from the historical Qwen2.5-7B QLoRA recipe and the Qwen3.5-4B Ollama serving benchmark. It uses two fictional training records defined in the script, with no benchmark development/test questions or customer documents.

The exact base is `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`. The script requires the corresponding Hugging Face snapshot cache directory and hashes its actual local files. Loading is local-only, with remote code and optional Hub kernels disabled. The tested environment used Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0, Accelerate 1.15.0, Safetensors 0.8.0, Tokenizers 0.23.2, and Hugging Face Hub 1.32.0. The CPU/default project requirements remain separate.

## Commands from the repository directory

The following Windows/PowerShell setup uses a new environment inside the cloned repository. The preserved reports used a sibling environment and cache; those original absolute paths describe the old machine and do not need to exist on yours. The source archive excludes base weights and generated adapters.

```powershell
py -3.12 -m venv .venv-qwen-smoke
& .venv-qwen-smoke/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
& .venv-qwen-smoke/Scripts/python.exe -m pip install -r requirements-qwen35-smoke.txt
@'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    cache_dir="models/hf-cache",
    allow_patterns=[
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "*.safetensors", "vocab.json", "merges.txt",
        "preprocessor_config.json", "video_preprocessor_config.json",
    ],
))
'@ | & .venv-qwen-smoke/Scripts/python.exe -
$qwenSnapshot = 'models/hf-cache/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
```

The Torch package comes from the [official CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/); the observed installed build was `2.14.0+cu130`. The base comes from the [exact official model revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). Downloads require network access and disk space for approximately 1.75 GB of model weights plus the separate Python/CUDA packages. CUDA training requires a compatible NVIDIA driver and BF16 GPU; the tested hardware is recorded below. This is not a CPU training recipe.

On Linux the interpreter path is `.venv-qwen-smoke/bin/python`, but this is a tested Windows training recipe, not a verified Linux GPU recipe. Linux Torch wheels may install Triton as a dependency; this bounded reference-kernel experiment explicitly rejects an environment containing Triton. Resolve that environment difference before claiming a Linux reproduction. The pinned model revision and cache layout remain the same on either platform. Do not use `local_dir=` or rename the revision directory: the smoke requires the pinned Hugging Face cache layout for provenance validation.

First run a tokenizer and snapshot check. This does not load model weights or call CUDA. Reading and hashing the local weight file is part of provenance validation.

```powershell
& .venv-qwen-smoke/Scripts/python.exe scripts/smoke_qwen35_lora.py `
  --model-path $qwenSnapshot `
  --preflight
```

The two complete conversations were checked with the pinned tokenizer: 101 and 105 tokens, including 21 and 17 assistant target tokens. Both prompt token sequences exactly prefix their full conversations. No truncation is permitted. The tokenizer preflight and the separate CUDA run below have passed; these are distinct checks.

After freeing GPU memory held by inference models, run the separate CUDA experiment. The output directory must be new. The script does not stop inference services or other processes itself.

```powershell
& .venv-qwen-smoke/Scripts/python.exe scripts/smoke_qwen35_lora.py `
  --model-path $qwenSnapshot `
  --output reports/qwen35_lora_smoke/my_new_run
```

Do not install the optional FLA, causal-conv1d, Triton, or bitsandbytes packages for this experiment. It deliberately uses BF16 base weights, native SDPA, the PyTorch DeltaNet reference implementation, and rank-4 LoRA. The 24-layer checkpoint config was instantiated on the Torch meta device, without weights or CUDA, confirming 186 language-only linear modules and 2,705,664 adapter parameters. The GPU run independently verifies the loaded class, actual module names, trainable tensors, and dtypes. It freezes the vision tower, embedding weights, and LM head.

## Evidence produced

`report.json` records stage/status, model and code hashes, library versions, exact input scope, supervised token counts, two optimizer-step losses/timings, finite-gradient checks, changed adapter tensor count, and peak CUDA allocated/reserved memory. Losses are observations from different synthetic examples, not an improvement metric. A failed run keeps its error and available diagnostics.

`synthetic_train.jsonl` contains the exact two inputs. `adapter/` contains the new LoRA weights, tokenizer, and `smoke_provenance.json`. The current v2 script saves a canonical base model ID and revision in the adapter config, reloads the saved tokenizer, and requires identical input IDs, labels, and attention masks on both records. It then releases the first model, loads a fresh base and only that saved adapter, and compares all evaluation forward logits for the first synthetic row using the reloaded tokenizer's input. It reports exact equality and maximum absolute difference; the declared numerical check uses `atol=0.02`, `rtol=0.002`. No text generation or held-out evaluation is part of this experiment.

## Observed CUDA run and follow-up review

[The immutable v1 report](../reports/qwen35_lora_smoke/run_v1/report.json) completed on 2026-09-20 using an NVIDIA GeForce RTX 4070 Laptop GPU. Both optimization steps had finite losses and adapter gradients; all 372 adapter tensors changed across the two updates.

| Observation | v1 result |
|---|---:|
| Trainable adapter parameters | 2,705,664 |
| Synthetic row 1 loss | 0.902749 |
| Synthetic row 2 loss | 2.750099 |
| PyTorch peak allocated GPU memory | 2,075,731,968 bytes (1.93 GiB) |
| PyTorch peak reserved GPU memory | 2,468,347,904 bytes (2.30 GiB) |
| Wall time after tokenizer/hash preflight | 6.448 seconds |
| Reload comparison shape | 1 × 101 × 248,320 logits |
| Maximum absolute reload difference | 0; bitwise equal |

The two losses correspond to different examples and do not measure improvement. PyTorch allocation/reservation is not total process or system GPU memory. The timing excludes snapshot hashing and tokenizer preflight and is a single compatibility run, not a throughput benchmark.

Review found that v1's PEFT adapter config stored an absolute local base path with no revision, although its separate provenance correctly recorded the canonical model ID, pinned revision, and file hashes. Its forward parity reused the original token IDs. A separate CPU check confirmed that the saved tokenizer reproduces both records' IDs, labels, and masks, but that check was not part of the original v1 GPU report. V1 remains unchanged. V2 addresses both issues in the script and reports them explicitly; its results must be read from its own saved report rather than inferred from v1.

[The separate v2 GPU report](../reports/qwen35_lora_smoke/run_v2/report.json) also passed. It records canonical adapter metadata, exact saved-tokenizer parity on both rows, 372 changed adapter tensors, and bitwise-equal logits after fresh-base reload. Its two losses were 0.902749 and 2.758850; elapsed time after preflight was 6.939 seconds, with the same 1.93 GiB allocated /2.30 GiB reserved PyTorch peaks. Seed 42 is recorded, but identical training trajectories across executions are not asserted; the second loss differs between v1 and v2. The equality claim applies to save/reload within each run.

For a moved adapter, explicitly load the pinned base with `AutoModelForImageTextToText` and attach the adapter with `PeftModel.from_pretrained(base, adapter_path)`, as this script does. Generic `AutoPeftModelForCausalLM` selects a different text-backbone loading route and is not the verified reload interface for these conditional-model adapter keys. Canonical metadata improves portability; it does not make the adapter compatible with a different base, model revision, loader, or Ollama deployment.

## Independent CPU tensor and source audit

The [separate successful CPU audit](../reports/qwen35_lora_smoke/run_v2_cpu_audit_explicit_device/audit_report.json) loads the conditional base and adapter explicitly on CPU and compares every adapter tensor against the serialized Safetensors file. All 372 tensors, comprising 2,705,664 parameters, match exactly in shape, dtype, and value. No forward pass or optimization step is performed, and `cuda_initialized` remains false. [Per-tensor hashes](../reports/qwen35_lora_smoke/run_v2_cpu_audit_explicit_device/tensor_comparison.json) are retained.

The audit also authenticated all ten base checkpoint/tokenizer/config files against the official Hugging Face metadata for the exact pinned commit. Large-file contents use LFS SHA-256; ordinary Git files use Git blob SHA-1, including the Git object header, and all byte sizes match. The [official metadata response](../reports/qwen35_lora_smoke/run_v2_cpu_audit_explicit_device/official_hf_metadata.json) is archived with its hash.

```powershell
& .venv-qwen-smoke/Scripts/python.exe scripts/audit_qwen35_adapter.py `
  --run reports/qwen35_lora_smoke/my_new_run `
  --output reports/qwen35_lora_smoke/new_cpu_audit
```

The audit command targets the new GPU run created above, including its generated `adapter/` directory. The bundled historical JSON reports alone are insufficient to rerun an adapter audit. It requires a new destination and verifies that source run/model/artifact hashes remain unchanged. Official Hub source authentication is read-only and requires network access; its status is reported separately from tensor equality. The first audit attempt is [preserved separately](../reports/qwen35_lora_smoke/run_v2_cpu_audit/audit_report.json): tensor and source checks passed, but its strict CPU guard detected CUDA initialization because PEFT inferred the adapter-loading device independently of the CPU base. The corrected audit supplies `torch_device='cpu'` explicitly and hides CUDA devices before imports. Both reports remain available; neither changed the original GPU run.

An old Qwen2.5 adapter cannot be supplied through this interface. No adapter is installed into Ollama. A passing report establishes training/save/reload compatibility for this pinned 0.8B text path; it does not establish useful fine-tuning, generalization, vision training, Qwen3.5-4B training, or reduced answer error rates.

The model architecture and supported loaders are documented in the [official checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json) and [Transformers 5.17 Qwen3.5 documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5). The unoptimized DeltaNet fallback is expected to be slower and use more memory than optional fused kernels; actual feasibility must come from the saved CUDA run.
