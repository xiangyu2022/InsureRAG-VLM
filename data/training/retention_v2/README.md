# Final source-isolated retention training groups

19,941 unique questions: 11,729 cleaned InsuranceQA, 213 government publisher questions and 7,999 original SQuAD crowdworker questions. There are 27,405 available positive pairs and 179,469 mined negatives. Frozen public-teacher predictions are cached for 206,874 final pairs; predictions are not human relevance labels.

Each group provides four BGE hard negatives, four BM25 hard negatives and one random negative. Training samples two of each hard-negative type and the random negative, along with one author positive. Unlabelled candidates may be plausible answers. Positive/negative text duplicates and new held-out source texts are excluded.

`source_isolation.json` records 2,522 remined groups after a pre-validation audit found 3,916 initial negatives from held-out source groups. The final fixture has zero positive/negative pairs from any of the new 76 held-out source groups (28 government URLs plus 48 Wikipedia titles). Source isolation applies to finetuning, not unknown exposure during upstream public-model pretraining.

Training repeats government questions four times per epoch; those repetitions are not additional data. The final two-epoch run contains 246,960 pair presentations, of which 24,484 positive and 139,926 negative question/answer pairs are distinct. See `reports/retention_v2/pair_exposure_audit.json` for the deterministic sampler replay. A selected earlier checkpoint may have fewer presentations.

Use the exact file hashes in `manifest.lock.json`. Attribution, data rights and retrieval-task limitations are detailed in `docs/RETENTION_REPRODUCTION.md` and `data/benchmarks/multidomain_v1/README.md`.
