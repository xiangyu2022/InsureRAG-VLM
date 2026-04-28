# InsureRAG-VLM
Built a citation-grounded multimodal RAG system for insurance policy and endorsement review using page-image retrieval and VLM generation, with selective abstention and clause-diff analysis.

## What it does
Page-image retrieval + grounded VLM answering for insurance-related questions.

## Why this project
Insurance policy packets are visually rich, and OCR-only RAG often misses layout, tables, and endorsement drift.

## Data
CUAD, ACORD, InsuranceQA, FUNSD/CORD proxies, plus synthetic policy QA pairs.

## Models
Retriever: ColQwen2/ColPali
Generator: Qwen2.5-VL-7B-Instruct (QLoRA)
Baselines: OCR-text RAG, Florence-2, PaliGemma 2

## Results
- Retrieval nDCG@10:
- Cited QA EM/F1:
- Citation precision:
- Abstention accuracy:

## Quickstart
1. Prepare data
2. Build page index
3. Run demo
4. Evaluate
