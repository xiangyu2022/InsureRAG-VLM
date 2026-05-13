# InsureRAG-VLM Ablation Summary

> Note: This summary predates the repository's `hybrid_multimodal` migration. Treat the
> numbers below as pre-migration baselines for `local_text`, `visual_stub`, and `local_image`,
> not as metrics for the current default retrieval stack.

## Retrieval

- local_text: Recall@1=0.3125, Recall@5=0.8125, MRR@10=0.5521, nDCG@10=0.6198
- visual_stub: Recall@1=0.3125, Recall@5=0.6250, MRR@10=0.4479, nDCG@10=0.4933
- local_image: Recall@1=0.3125, Recall@5=0.6250, MRR@10=0.4479, nDCG@10=0.4933

## Answering

- local_text: F1=0.5851, Citation Precision=0.4375, Evidence Recall=0.4375, Unsupported Abstention=1.0000
