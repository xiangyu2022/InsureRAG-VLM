# Retrieval-Conditioned MSI Run Summary

- Date: `2026-05-14`
- Cluster: `MSI`
- GPU: `1x NVIDIA A40`
- Dense retriever output: `models/retrieval/bge-base-insurerag`
- Retrieval-conditioned adapter output: `models/qwen7b-insurerag-lora-retrieval`

## Job Chain

| job_id | name | state | elapsed |
| --- | --- | --- | ---: |
| `9031895` | `insurerag-build-corpora` | `COMPLETED` | `00:15:31` |
| `9031925` | `insurerag-dense-train` | `COMPLETED` | `00:02:30` |
| `9031926` | `insurerag-build-corpora` | `COMPLETED` | `00:14:57` |
| `9031986` | `insurerag-sft-retrieval` | `COMPLETED` | `01:28:48` |
| `9080097` | `insurerag-sft-eval` | `COMPLETED` | `00:02:10` |

## Corpora Produced

The rebuilt doc-disjoint corpora in `reports/training_data_dense/` contain:

| split | retrieval | rag_sft | calibration |
| --- | ---: | ---: | ---: |
| train | `2873` | `3231` | `0` |
| dev | `29` | `35` | `35` |
| test | `298` | `334` | `334` |

## Training Summary

### Dense Retriever

- Base encoder: `BAAI/bge-base-en-v1.5`
- Training metadata: `models/retrieval/bge-base-insurerag/dense_retriever_metadata.json`
- Trainer-reported samples: `1050`
- Epochs: `2`
- Batch size: `8`
- Learning rate: `2e-5`

### Retrieval-Conditioned QLoRA

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Warm start adapter: `models/qwen7b-insurerag-lora`
- Training metadata: `models/qwen7b-insurerag-lora-retrieval/sft_metadata.json`
- Dataset: `reports/training_data_dense/rag_sft_train.jsonl`
- Train samples: `3231`
- Epochs: `2`
- Effective batch size: `8`
- Learning rate: `1e-4`
- Final train loss: `0.3378`
- Trainable params: `40,370,176`
- Peak CUDA memory: `10930.2 MB`

## Spot Check

The new adapter was evaluated with the same 8-sample spot check used for the original clean-evidence adapter:

- Base model comparison: `reports/sft_eval/retrieval_adapter_spot_check.{json,md}`
- Previous clean-evidence baseline: `reports/sft_eval/adapter_spot_check.{json,md}`

| variant | overall_f1 | answerable_f1 | unsupported_abstain_rate |
| --- | ---: | ---: | ---: |
| base model | `0.2168` | `0.3035` | `0.7500` |
| clean-evidence adapter | `1.0000` | `1.0000` | `1.0000` |
| retrieval-conditioned adapter | `0.7443` | `0.4886` | `1.0000` |

## Interpretation

- The retrieval-conditioned adapter is materially better than the base model on this spot check.
- It keeps the unsupported abstain behavior at `1.0`, which is desirable for noisy retrieval settings.
- It does **not** beat the earlier clean-evidence adapter on answerable quote fidelity in this 8-sample check.
- The retrieval-conditioned adapter should therefore be treated as a robust RAG-oriented candidate, not yet as an unconditional replacement for the current serving adapter.

## Recommended Next Step

Run a held-out pipeline evaluation with retrieved context, citations, and abstain metrics before changing the default answer adapter:

```bash
.venv/bin/python main.py evaluate data/04_curated reports/training_data_dense/calibration_test.jsonl \
  --index-dir reports/training_data_dense/index \
  --retrieval-model models/retrieval/bge-base-insurerag \
  --retrieval-mode hybrid_multimodal \
  --corpus-source curated \
  --disable-image-signal \
  --json
```
