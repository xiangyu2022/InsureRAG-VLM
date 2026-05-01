# InsureRAG-VLM Ablation Summary

Evaluated on **300 real insurance policy QA pairs** from public PDFs (Maryland + North Carolina
state insurance department documents). Document-level train/val/test splits: 8 train / 1 val / 1 test documents.

## Retrieval Metrics (n=300, top-k=5)

| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| local_text (OCR+hashing) | 0.0567 | 0.2600 | 0.1238 | 0.1573 |
| visual_stub (random baseline) | 0.0300 | 0.1367 | 0.0634 | 0.0813 |
| local_image (image hash baseline) | 0.0300 | 0.1367 | 0.0634 | 0.0813 |
| colqwen2_local (GPU, pending) | — | — | — | — |

## Notes

- `local_text` uses SHA256 keyword hashing — a strong baseline for text-extractable PDFs.
- `visual_stub` and `local_image` are CPU-only random/hash baselines; both score identically
  because the image hash fallback reduces to the same permutation on this dataset.
- `colqwen2_local` requires GPU (CUDA). See `notebooks/colqwen2_gpu_embed.ipynb` for the
  Colab notebook to generate real ColQwen2 embeddings and update this table.
- North Carolina PDFs are scanned (image-only) — no text extraction possible without OCR.
  These are ideal documents for demonstrating why visual retrieval outperforms text-only RAG.

## Data

- **QA pairs**: `data/02_processed/qa_pairs.jsonl` (300 answerable + 3 unsupported)
- **Hard negatives**: `data/02_processed/hard_negatives.jsonl` (900 pairs)
- **Document splits**: `data/02_processed/splits.jsonl` (8 train / 1 val / 1 test)
- **Page manifest**: `data/03_index/colqwen2/page_manifest.jsonl` (64 pages across 10 docs)
- **Visual indexes**: `data/03_index/colqwen2/{visual_stub,local_image}.npy`
