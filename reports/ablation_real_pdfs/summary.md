# InsureRAG-VLM Ablation Summary

> Note: This summary predates the repository's `hybrid_multimodal` migration. Treat the
> numbers below as pre-migration baselines for `local_text`, `visual_stub`, and `local_image`,
> not as metrics for the current default retrieval stack.

## Retrieval

- local_text: Recall@1=0.2033, Recall@5=0.4467, MRR@10=0.2974, nDCG@10=0.3348
- visual_stub: Recall@1=0.1733, Recall@5=0.4400, MRR@10=0.2733, nDCG@10=0.3149
- local_image: Recall@1=0.1733, Recall@5=0.4400, MRR@10=0.2733, nDCG@10=0.3149

## Answering

- local_text: F1=0.3695, Citation Precision=0.2533, Evidence Recall=0.2533, Unsupported Abstention=0.6667
