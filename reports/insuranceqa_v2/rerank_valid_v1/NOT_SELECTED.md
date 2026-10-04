This scoring attempt was not used for selection. Its batched float32 GEMM
exchanged near-tied BGE answers in 85/2,000 validation rankings; no top-10
candidate set changed, and all BM25 rankings matched. The exact-ranking parity
gate rejected evaluation before metrics or hyperparameter selection. The next
run uses the original baseline's matrix-vector arithmetic. The original scoring
source is preserved as `scoring_source.py`.
