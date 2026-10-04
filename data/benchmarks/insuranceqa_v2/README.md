# InsuranceQA V2: an external answer-selection retrieval track

Source: [authors' repository](https://github.com/shuzi/insuranceQA), pinned at
`5c380dd086067adacc2fec3bfa920a78e57bdcf6`.

Citation: Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou.
*Applying Deep Learning to Answer Selection: A Study and An Open Task*. ASRU 2015.

**Upstream provides this data as is, for research purpose only. It is not
relicensed under this project's code license.** Original README files, encoded
downloads, URLs, hashes and decoded records are included for this local research
prototype. Do not assume unrestricted commercial use or redistribution rights.

Counts: 12,889 train, 2,000 validation, 2,000 test questions; 27,413 answer IDs;
12 insurance domains. Original answer IDs, split membership and candidate pools
are preserved. Train uses the upstream 100-candidate file to recover split
membership; validation/test use the 1,000-candidate files. Positive labels are
never inserted into a pool. Consequently, the original test pool contains a
positive for only 1,744/2,000 questions.

Question text is decoded using the authors' raw vocabulary. This is historical
FAQ answer selection, not current policy interpretation or a graph benchmark.
Answers can be plausible without carrying a positive label. Model pretraining
exposure is unknown. No model was fitted on imported questions in this run.

The import audit flags 8 exact normalized test-question overlaps with train or
validation, 0 within-test normalized duplicates, and 18 redundant answer IDs
across 17 identical-text groups. A separately recorded, post-hoc char-ngram audit
flags 476 similar test questions; its 1,524-question remaining slice is a
sensitivity analysis, not a new independent benchmark.

Recreate into a new folder:

```powershell
python scripts/prepare_insuranceqa.py --output data/benchmarks/insuranceqa_v2_recreated
python scripts/prepare_insuranceqa.py --verify-only
python scripts/eval_insuranceqa_scale.py --model C:/models/bge-small-en-v1.5 --device cpu --output reports/insuranceqa_v2/new_run
```

The primary run uses unchanged settings across validation and test, CLS pooling,
512-token answer truncation, the declared BGE query instruction, exact dense
search, BM25 k1=1.5/b=0.75 and RRF k=60/depth=100. Scores are label-ID matches.
See raw rankings, protocol hashes, domain slices and intervals under
`reports/insuranceqa_v2/retrieval_frozen`.
