# Reproduce the domain reranker experiment

This track trains the retrieval reranker, using the fixed BGE encoder for candidate
retrieval. It does not fine-tune Qwen or change the PDF-demo generation path.
Use Python 3.12 and install `requirements-domain-training.txt`. The recorded GPU
runtime is PyTorch 2.14.0+cu130 / Transformers 5.17.0 on an RTX 4070 Laptop 8GB.
The trainer explicitly requires CUDA; query inference also supports CPU.

## Inputs and pinned models

The delivered source retains InsuranceQA's original author labels and research-only
use terms. Training v2 has 11,729 questions after normalized-question, lexical
near-duplicate, held-out positive-ID and held-out positive-text exclusions. Held-out
content is used only to remove contaminated training examples. It is not used to
fit model parameters or choose hyperparameters.

The base reranker is
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` (Apache 2.0). Acquire it with:

```powershell
python scripts/acquire_minilm_reranker.py --output ../models/ms-marco-MiniLM-L6-v2 --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

BGE-small-en-v1.5 uses revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
For this FAQ track it uses CLS pooling, 512 tokens, L2 normalization and the query
prefix `Represent this sentence for searching relevant passages: `. This differs
from the earlier PDF experiment's no-prefix encoding contract. See the original
scoring protocol's per-file model hashes before reusing its caches.

The data/index delivery restores the original InsuranceQA answer cache under
`reports/insuranceqa_v2/retrieval_frozen/` and the new HICRIC index under
`reports/hicric_public_v1/bge_index/`. Index files are excluded from Git. The source
delivery retains their protocols/hashes and the builders.

## Run training and select on validation

The following commands show the recorded run. Builders intentionally refuse to
overwrite artifacts. For a new experiment, use fresh report/checkpoint/output
directories consistently instead of deleting the historical reports.

```powershell
python scripts/prepare_reranker_training.py --model ../models/bge-small-en-v1.5 --device cuda --output data/training/insuranceqa_hardneg_v1
python scripts/refine_training_answer_dedup.py --source data/training/insuranceqa_hardneg_v1 --output data/training/insuranceqa_hardneg_v2
python scripts/train_domain_reranker.py --data data/training/insuranceqa_hardneg_v2 --model ../models/ms-marco-MiniLM-L6-v2 --output reports/domain_training_v2 --checkpoints ../models/insurerag-domain-reranker-v2
python scripts/eval_domain_training.py plan --run reports/domain_training_v2
foreach ($epochNumber in 1,2,3) {
    python scripts/score_domain_checkpoint.py --model "../models/insurerag-domain-reranker-v2/epoch-$epochNumber" --candidates reports/insuranceqa_v2/rerank_valid_v2 --split valid --output "reports/domain_training_v2/valid_epoch_$epochNumber"
    if ($LASTEXITCODE -ne 0) { throw 'Validation scoring failed' }
}
python scripts/eval_domain_training.py select --run reports/domain_training_v2
```

Register the selection plan before reading any validation metric. The recorded v2
plan was saved while training was still running. The candidate set stays fixed:
BGE top100 plus positive BM25 top100, then the registered hybrid100 policy. The grid
has three epochs × four cross-encoder weights; lexical weight is fixed at 0.2.
Select by validation Hit@10, then MRR@100, then Hit@1. The previous model is retained
if no trained checkpoint exceeds its validation Hit@10.

`hybrid100` describes the final candidate policy, not the number of forward passes
performed by the cache-building evaluator. The evaluator scores the complete union
(up to 200 answers; validation mean 174.43 per question) so the BGE-only candidate
ablation can reuse identical neural scores. The trained and untrained rerankers use
the same cached candidate union. Their extra computation is not included in a
standalone BGE-only search, so this is a quality comparison with additional reranking
cost, not an equal-cost replacement for BGE.

AdamW updates all 22,713,601 parameters. Training uses FP32 parameters with BF16
autocast, learning rate 2e-5, 10% warmup, weight decay 0.01, gradient clipping 1.0,
three epochs, seed 42, four question groups per microbatch and accumulation of four.
Each group contains one author positive, two BGE hard negatives, two BM25 hard
negatives and one random negative. Loss is listwise cross entropy with smoothing
0.05. These negatives are unlabelled candidates, not expert-confirmed wrong answers.

Three epochs produce 2,202 optimizer steps and 211,122 pair exposures. Only 17,994
of the available 19,193 distinct positive pairs are sampled across those epochs.
The selected checkpoint may be earlier than epoch 3. Keep unique question count,
available pairs, actual exposures and selected epoch distinct.

## Frozen evaluations and working query

Resolve the selected checkpoint from the validation lock; do not pick an epoch
using test results:

```powershell
$selection = Get-Content reports/domain_training_v2/selection.lock.json -Raw | ConvertFrom-Json
$selectedModel = "../models/insurerag-domain-reranker-v2/epoch-$($selection.epoch)"
python scripts/score_domain_checkpoint.py --model $selectedModel --candidates reports/insuranceqa_v2/rerank_test_v1 --split test --selection reports/domain_training_v2/selection.lock.json --output reports/domain_training_v2/test_selected
python scripts/eval_domain_training.py test --run reports/domain_training_v2
python scripts/eval_government_transfer.py --embedding-model ../models/bge-small-en-v1.5 --base-model ../models/ms-marco-MiniLM-L6-v2 --trained-model $selectedModel --selection reports/domain_training_v2/selection.lock.json --output reports/hicric_government_qa_v1/transfer_v1
python scripts/verify_domain_training.py --base ../models/ms-marco-MiniLM-L6-v2 --model $selectedModel
python scripts/query_domain_reranker.py --question "What is the difference between term and whole life insurance?" --embedding-model ../models/bge-small-en-v1.5 --reranker-model $selectedModel --device cuda
python scripts/query_domain_reranker.py --corpus hicric --profile general --question "What are the requirements for Medicaid to pay after a third-party insurer?" --embedding-model ../models/bge-small-en-v1.5 --reranker-model ../models/ms-marco-MiniLM-L6-v2 --device cuda
```

The government corpus and index build commands are in its
[data card](../data/research_corpus/hicric_public_v1/README.md). The retrieval CLI
returns source text and provenance, makes zero generation calls and does not read
question labels. The government FAQ evaluation uses 27,829 answer candidates;
free-form HICRIC snippet search uses the separate 78,830-snippet index. Results on
one do not measure the other.

The explicit `general` profile retains the original public reranker and original
validation lock. The government transfer test showed a regression for domain
training, so this option is provided for comparative research and the example above
uses it. It is a post-evaluation usage recommendation, not a newly validated
automatic routing policy. To investigate the trained model on the same HICRIC
corpus, supply `--profile trained --reranker-model $selectedModel` explicitly.

The v1 development run and interrupted partial test scores remain labelled
`NOT_SELECTED.md`. Ten positive-answer text collisions discovered during auditing
caused the v2 rebuild before any v1 test metric was computed. No government-test
metric was available during model selection. Frozen historical reports are retained.
