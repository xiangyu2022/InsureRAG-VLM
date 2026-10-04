# Mixed-domain retention experiment

This experiment continues the selected InsuranceQA v2 MiniLM cross-encoder. It does not train BGE or Qwen, and does not change the separately evaluated PDF/GraphRAG generator.

## Data and attribution

- InsuranceQA: 11,729 cleaned author-labelled training questions. The upstream research-only terms continue to apply.
- New government material: 213 training questions from 28 canonical URLs, 127 validation questions from 14 URLs, and 81 test questions from 14 URLs. Questions and answers are mechanically extracted from publisher Q/A in the pinned HICRIC archive; no LLM-created labels. All 37 URLs of the earlier 416-question government evaluation are excluded from this new fixture.
- SQuAD 1.1: 7,999 original crowdworker training questions, 400 validation and 600 test questions. At most one question per paragraph. Original train/dev titles are disjoint; dev titles are split into 24 validation and 24 test titles. Original answer spans are checked. This is a **derived paragraph-retrieval task**, not official SQuAD span EM/F1.
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [HICRIC](https://huggingface.co/datasets/Persius/hicric) attribution and CC BY-SA 4.0 terms are retained. Paragraph records include original Wikipedia/government URLs. Public pretrained-model exposure to these sources is unknown. The government archive is historical material, not present-day policy guidance.

The mixed index contains 47,969 answer candidates: the original 27,413 InsuranceQA answers plus 20,556 government/SQuAD paragraphs. It includes unlabelled distractor paragraphs. New cases are searched against all 47,969 candidates. The historical FAQ comparison retains its original 27,413-candidate corpus. These different test populations must not be pooled into a single accuracy claim.

## Integrity correction before validation

The initial retention-v1 negative miner searched the entire corpus, allowing 3,916 negatives from new held-out source groups into 2,522 training groups. No held-out questions or positive labels were used, but this violated the intended strict source separation. The attempt was interrupted and marked `reports/retention_v1/NOT_SELECTED.md` before any validation or new test scores were computed.

`isolate_retention_sources.py` remines affected groups while excluding **every paragraph from every new validation/test source**, including normalized text duplicates. It preserves the evaluation index. Retention v2 starts again from the original selected InsuranceQA v2 checkpoint. The final data audit checks both positive and negative pairs against forbidden source texts.

Two SQuAD training records share a normalized paragraph with a punctuation variant. Their original offsets are checked against the original upstream paragraph, and normalized equality with the canonical retrieval paragraph is verified. They are documented in `data_audit.json`; offsets must not be interpreted as positions in the canonical punctuation variant.

The pre-score quality screen also flags 8 possible context/footnote issues (3 train, 2 validation, 3 test), retained with their original labels. See `government_quality_flags.json`. Source alignment alone does not make each question self-contained.

## Training and finite selection

Training uses 19,941 unique questions, 27,405 available positive pairs and 179,469 mined negative pairs. Each sampled group contains one original positive, two dense hard negatives, two lexical hard negatives and one random negative. Negatives are unlabelled candidates, not expert-confirmed incorrect answers. Positives cycle across epochs where multiple labels exist.

Government groups are presented four times per epoch: 20,580 total group presentations per epoch, **not** 20,580 unique questions. The supervised objective is listwise cross entropy with 0.05 label smoothing, plus KL(public teacher || student) over the same six candidates, temperature 2. KL weights are 0.15 for InsuranceQA and 0.5 elsewhere. Teacher scores are predictions from the frozen public MS MARCO MiniLM checkpoint, not labels.

Full-parameter training: AdamW, learning rate 5e-6, weight decay 0.01, 10% warmup, gradient clipping 1, FP32 parameters with BF16 autocast, microbatch four groups and four-step accumulation. Seed 42; two epochs. No cross-seed stability is established.

Before validation, `reports/retention_v2/selection_protocol.json` registers two checkpoints and domain-model interpolation weights 0.5, 0.75 and 1.0. Retrieval/final fusion stays fixed: BGE/BM25 top-100 union, hybrid100 pool, lexical weight 0.2, cross weight 0.5. Each cross-encoder is min-max normalized before interpolation; final fusion normalizes within the retained pool. An interpolation weight below 1 costs two model passes.

Selection maximizes equal-domain macro validation Hit@10, with macro MRR then Hit@1 tie breaks. Eligibility requires InsuranceQA Hit@10 at least previous-v2 minus 0.5 percentage points, and each new domain at least public-reranker minus 1 point. A new checkpoint is promoted only if eligible and strictly better than unanchored previous-v2 macro Hit@10; otherwise retain previous v2. Previous-v2 with the same interpolation weights is a reported control, to distinguish new training from ensembling.

## Commands

Run from the repository with a CUDA PyTorch environment; the recorded hardware is an RTX 4070 Laptop GPU with 8 GB VRAM. The scripts refuse to overwrite existing run directories. To reproduce, use a clean extracted copy with input artifacts and remove/move only the intended generated run outputs after verifying their paths. Local-only model loading is used; no paid API.

```powershell
# Original frozen source fixture and initial cached teachers.
python scripts/prepare_multidomain_data.py --output data/benchmarks/multidomain_v1 --cache ../squad_download
python scripts/prepare_retention_training.py --output data/training/retention_v1 --index reports/multidomain_v1/index_cache --embedding-model ../models/bge-small-en-v1.5 --teacher ../models/ms-marco-MiniLM-L6-v2
# Required correction: do not train the initial v1 groups.
python scripts/isolate_retention_sources.py
New-Item -ItemType Directory reports/retention_v2
python scripts/eval_retention_v2.py plan
python scripts/audit_retention_data.py
python scripts/train_retention_reranker.py --data data/training/retention_v2 --model ../models/insurerag-domain-reranker-v2/epoch-1 --output reports/retention_v2/train_run --checkpoints ../models/insurerag-retention-v2 --epochs 2 --learning-rate 5e-6
python scripts/run_retention_validation.py
python scripts/run_retention_tests.py
```

Scoring and selection write source, model, data and prediction hashes. Fresh test scoring requires a validation selection lock and accepts only its selected model and declared public/previous controls. Test scores never alter the selected checkpoint, fusion settings or dataset.

The `model_path` in the recorded selection lock is the original machine's absolute provenance path. If replaying test jobs elsewhere, preserve that lock and use `score_multidomain.py` / `score_domain_checkpoint.py` directly with the relocated `--model` and the original `--selection`; hashes identify the checkpoint. The query CLI uses portable sibling model paths or explicit overrides.

## Query the selected model

```powershell
python scripts/query_retention_reranker.py --question "What does an insurance deductible mean?" --corpus insuranceqa --device cuda
python scripts/query_retention_reranker.py --question "What rules apply to Medicaid eligibility?" --corpus multidomain --device cuda
python scripts/query_retention_reranker.py --question "What are the requirements for mental health parity?" --corpus hicric --device cuda
```

The HICRIC option searches 78,830 archived snippets. Its query smoke is executable integration evidence, **not** a labelled full-snippet accuracy benchmark. Query results expose source metadata when available and make no generation call. The original CLI remains available for comparison.

## Interpretation

Report Hit@1, Hit@10 and MRR@100, raw wins/losses and paired uncertainty. New government/general comparisons resample canonical URLs/Wikipedia titles in 5,000 bootstrap draws. The 81-question government test has only 14 source groups. Historical InsuranceQA uses gold-label connected-component clusters; the earlier 416-question government set is now a historical regression set. Finetuning source separation does not prove absence from public model pretraining or semantic/topic independence.

Do not relabel these retrieval results as lower hallucination rates, end-to-end policy reasoning, Qwen accuracy, production readiness or an improved BGE encoder.
