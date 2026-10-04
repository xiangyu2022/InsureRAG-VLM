# Reproducing the research prototype

The current serving target is **Ollama `qwen3.5:4b`**, independently of the historical Qwen2.5-7B LoRA experiments. The lightweight default remains an explicitly labeled `local-extractive` / `local-hashing` baseline, so installing Python dependencies never silently selects a hosted provider or a different local model.

## Tested local environment

- Windows, Python 3.12.10; NVIDIA RTX 4070 Laptop GPU, 8,188 MiB VRAM, driver 596.08.
- Ollama 0.34.2 standalone Windows CLI, listening on `127.0.0.1:11435` in this run.
- Qwen3.5-4B Q4_K_M, exact Ollama digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.
- The legacy comparison arm is the separately downloaded **unadapted** `qwen2.5:3b` Ollama model. It is not the old trained Qwen2.5-7B adapter. A comparison changes model family, size, and model artifact; it cannot isolate a pure version effect.
- Embedding experiment: BAAI/bge-small-en-v1.5 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, CLS pooling, L2 normalization, 512-token maximum, no query prefix. CPU PyTorch 2.14.0+cpu and Transformers 5.17.0 were used. The normal Ollama path does not require PyTorch.

Model names are mutable labels. Keep the digest, exact dependency versions, manifest hashes, code hashes, decoding options, and raw predictions with a result. A tag name alone is not an exact reproduction. The runtime checks that the installed digest does not change during a client session.

The source ZIP preserves the delivered working-file bytes. Applying the accompanying
Git patch can normalize CRLF/LF in text files according to Git settings; the delivery
audit lists those differences separately from content changes. Binary PDFs, PNGs and
ZIPs have explicit attributes and retain exact bytes. For the precise pre-test source
used by the frozen public comparison, use `reports/research_v1/frozen_source_v1.zip`;
the current application includes later development repairs and presentation changes.

## CPU unit checks

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
$env:INSURERAG_VLM_MODEL='local-extractive'
$env:INSURERAG_USE_OLLAMA='0'
.venv/Scripts/python -m pytest -q
```

No model download, CUDA, API credential, or live website is needed for these tests. CI now runs pytest on Windows/Linux and Python 3.11/3.12; the local report does not imply those remote CI jobs have executed.

`requirements-tested.txt` captures the complete tested lightweight Windows/Python 3.12 environment. It is a reproduction snapshot, not a claim that every listed version is required on other platforms. The CLI emits UTF-8, including when redirected from a legacy Windows console.

## Real Qwen3.5 serving

Install Ollama from its [official distribution](https://docs.ollama.com/windows), start its local server, and pull the exact tag:

```powershell
ollama pull qwen3.5:4b
$env:INSURERAG_USE_OLLAMA='1'
$env:INSURERAG_VLM_MODEL='ollama:qwen3.5:4b'
$env:OLLAMA_BASE_URL='http://127.0.0.1:11434'
.venv/Scripts/python main.py demo-web --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`. The sidebar identifies the configured model; each answer identifies the path actually used. Glossary lookup, extractive answers, deterministic evidence repair, and model generation are distinct. A missing or failing requested model returns an error, not a substitute answer backend.

This is a local single-user research demo. Public consumer guides are reference material, not the user's issued policy. Uploaded policy QA uses the uploaded documents. The image-QA API is explicitly opt-in; normal RAG generation receives text.

## Frozen public-guide diagnostic

Read [the benchmark card](../data/benchmarks/research_v1/README.md) before interpreting scores. It contains 12 development and 24 test questions on disjoint document scopes; public pretraining exposure is unknown, and the old SFT corpus included these documents. Only unadapted Ollama artifacts are compared.

```powershell
.venv/Scripts/python scripts/eval_research_benchmark.py --validate-only --split test --output reports/research_v1/validation
.venv/Scripts/python scripts/eval_research_benchmark.py --endpoint http://127.0.0.1:11434 --model qwen3.5:4b --split dev --mode both --output reports/research_v1/my_qwen35_dev
```

The retrieved mode runs the actual hybrid retrieval/context path with a benchmark JSON contract. Oracle mode injects gold pages solely to diagnose generation and is never a retrieval result. All decoding options are explicit; thinking is disabled through the API. Keep output folders immutable. Development observations may guide changes; the test split is not a tuning loop.

The deterministic key, quotation, and citation checks are not semantic correctness or an insurance decision error rate. In particular, missing a full annotated quote can fail the strict score while the short answer is correct. See the independent [fixture review](../reports/research_v1/fixture_review_v1.md) and [development error audit](../reports/research_v1/dev_v1_error_review.md).

## Optional real embedding checkpoint

Install CPU PyTorch if that is the intended backend, then Transformers and huggingface-hub. Download all necessary checkpoint metadata, not only weights:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    local_dir="models/bge-small-en-v1.5",
    allow_patterns=["*.json", "*.txt", "1_Pooling/config.json", "model.safetensors"],
)
```

Pass `--retrieval-model models/bge-small-en-v1.5` to the diagnostic runner. Checkpoints must declare pooling or receive an explicit configuration; unknown models no longer silently fall back to hashing. Existing indexes without matching model/pooling/tokenizer/weight fingerprints require rebuilding. Newly trained repository encoders declare their historical **mean pooling** contract rather than inheriting BGE's CLS contract.

## Vision API smoke

```powershell
.venv/Scripts/python scripts/eval_vision_smoke.py --endpoint http://127.0.0.1:11434 --pdf path/to/Auto-Insurance-Guide.pdf --structured-output --output reports/vision_smoke/my_run
```

The source URL and expected PDF checksum are recorded in the delivered results. This checks four authored questions on one public PDF page, using actual PNG pixels and image hashes. The first development run produced three correct numeric answers in unwanted Markdown fences; it passed only 1/4 strict JSON checks. The schema-constrained rerun passed 4/4. This demonstrates the image transport and output contract, not visual QA accuracy on a representative benchmark.

## Historical training and supervision audit

The current serving upgrade is not a new fine-tuning claim. Existing LoRA training code and historical reports still name their actual Qwen2.5-7B text backbone; old adapters cannot be attached to Qwen3.5 by renaming a configuration.

```powershell
.venv/Scripts/python -m pip install transformers==5.17.0 huggingface-hub==1.32.0
.venv/Scripts/python scripts/audit_sft_supervision.py --revision a09a35458c702b33eeacc393d103063234e8bc28 --cache-dir models/tokenizer-cache --output reports/sft_tokenization_audit/my_run
```

This downloads tokenizer files, not LLM weights. The current 3,850-row curated dataset has 39 answerable question templates, all referring to multiple gold pages. Its maximum formatted sequence length under that tokenizer is 487 tokens; no answers are truncated at 1,024. These are **not** the unavailable historical 3,231 retrieval-conditioned rows. The code now rejects prompt-only truncation instead of training on the last prompt token.

SFT provenance, source/document overlap audits, and the historical small spot-check limitations remain documented in [the evaluation audit](../reports/evaluation_audit_2026-09-20.md). None can establish absence from base-model pretraining.

## New-architecture training compatibility

The separate [Qwen3.5-0.8B LoRA smoke](QWEN35_LORA_SMOKE.md) completed two actual CUDA optimizer steps on two invented records. It updated 2,705,664 trainable adapter parameters, saved the adapter, and reloaded it into a fresh base with identical evaluation logits (maximum absolute difference 0.0). Peak PyTorch allocation was 2,075,731,968 bytes; this excludes desktop/inference services and is not total device usage. This is compatibility evidence only. It did not train the 4B serving model or establish any answer-quality improvement.

Use a separate environment and `requirements-qwen35-smoke.txt`; the tested Torch wheel was `2.14.0+cu130`. Base weights and generated adapters are excluded from source control. The compact report records their hashes; the generated adapter can be recreated by the documented command.
