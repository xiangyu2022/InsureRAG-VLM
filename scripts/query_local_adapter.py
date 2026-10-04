"""Run offline hybrid retrieval with the official base or an explicit experimental adapter."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["INSURERAG_USE_OLLAMA"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer
from scripts.run_local_lora_experiment import load_model, prompt_ids, seed_all
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default="models/Qwen3.5-2B")
    parser.add_argument("--adapter", default=None, help="Explicit experimental adapter; omitted uses the official base")
    parser.add_argument("--corpus", type=Path, default=Path("data/04_curated"))
    args = parser.parse_args()
    adapter = args.adapter
    seed_all()
    config = ModelConfig(index_dir=Path("reports/local_20261004/retrieval_index"),
                         retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False,
                         curated_dataset_dir=args.corpus, corpus_source="curated", enable_image_signal=False)
    pipeline = DocumentRetrievalPipeline(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    pages = pipeline.rank_pages(args.question, args.corpus, top_k=5)
    def count_prompt(context):
        return len(prompt_ids(tokenizer, {"question": args.question, "evidence": context,
                                          "source": "", "input_source": "retrieved"}))
    packed = pipeline.pack_context_with_audit(pages, config.max_answer_pages,
                                              prompt_token_counter=count_prompt,
                                              max_prompt_tokens=2048, question=args.question)
    context = packed["context"]
    retrieval_seconds = time.perf_counter() - start
    model = load_model(args.model, adapter)
    model.eval()
    row = {"question": args.question, "evidence": context, "source": "",
           "input_source": "Use only the SOURCE identifiers present in the evidence above."}
    ids = torch.tensor([prompt_ids(tokenizer, row)], device="cuda")
    assert ids.shape[1] == packed["prompt_tokens"] <= 2048
    model_limit = model.config.text_config.max_position_embeddings
    assert ids.shape[1] + 128 <= model_limit
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=128, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    torch.cuda.synchronize()
    result = {"question": args.question, "answer": tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True),
              "model": args.model, "adapter": adapter, "retriever": "local-hashing hybrid",
              "retrieved_sources": [p["source"] for p in pages], "packed_context": context,
              "packing": packed,
              "retrieval_seconds": retrieval_seconds, "generation_seconds": time.perf_counter() - start,
              "scope": "local research prototype; citation correctness and answer support require review"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
