"""Corrected local-hashing retrieval baseline; this is NOT a trained-BGE rerun."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.retrieval_manifest import scope_and_merge
from src.insurerag_vlm.retrieval_metrics import mean_scores, score_ranking, validate_queries

ROOT = Path("reports/local_20261004")


def main():
    corpus = Path("data/04_curated")
    config = ModelConfig(retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False,
                         corpus_source="curated", curated_dataset_dir=corpus, enable_image_signal=False,
                         index_dir=ROOT / "retrieval_index")
    pipeline = DocumentRetrievalPipeline(config)
    pipeline.build_index(corpus)
    summary = {"encoder": "local-hashing", "trained_bge_rerun": False,
               "scope": "document-scoped synthetic queries; existing positives only",
               "ranking_budget": {"snippets": config.snippet_top_k, "pages": config.page_top_k,
                                  "merged_candidates": config.candidate_pool_size},
               "corpus_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in corpus.glob("rag_*.jsonl")}, "splits": {}}
    for split in ("dev", "test"):
        rows = [json.loads(x) for x in (ROOT / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        examples = [{"qa_id": r["id"], "question": r["question"], "answerable": True,
                     "source_doc_id": r["doc_id"], "evidence_sources": [r["source"]]}
                    for r in rows if r["answerable"]]
        examples = scope_and_merge(examples)
        validate_queries(examples, strict=True)
        (ROOT / f"retrieval_{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in examples))
        scores, latencies, outputs = [], [], []
        for row in examples:
            start = time.perf_counter()
            ten = pipeline.rank_pages(row["question"], corpus, top_k=10)
            latencies.append(time.perf_counter() - start)
            five = pipeline.rank_pages(row["question"], corpus, top_k=5)
            assert five == ten[:5], "Serving/evaluation prefix changed"
            values = score_ranking(row, ten)
            scores.append(values)
            outputs.append({"qa_id": row["qa_id"], "scores": values,
                            "top10_sources": [x["source"] for x in ten]})
        (ROOT / f"retrieval_{split}_rankings.jsonl").write_text("".join(json.dumps(x) + "\n" for x in outputs))
        summary["splits"][split] = {"queries": len(examples), **mean_scores(scores),
                                    "mean_seconds": sum(latencies) / len(latencies),
                                    "serving_prefix_check": "passed"}
        if split == "test":
            diagnostic = []
            for row in rows:
                if not row["answerable"]:
                    # Unsupported labels only apply to the supplied original
                    # evidence, not to the entire retrieval corpus.
                    continue
                query = f"In the document {row['doc_id']}, {row['question']}"
                ranked = pipeline.rank_pages(query, corpus, top_k=5)
                context = pipeline.pack_long_context(ranked, config.max_answer_pages)
                gold = {"question": query, "evidence_sources": [row["source"]]}
                matching = [p for p in ranked if score_ranking(gold, [p])["hit_at_5"]]
                diagnostic.append({**row, "question": query, "original_evidence": row["evidence"],
                                   "original_source": row["source"],
                                   "source": matching[0]["source"] if matching else row["source"],
                                   "evidence": context,
                                   "input_source": "Use only the SOURCE identifiers present in the evidence above.",
                                   "retrieved_sources": [p["source"] for p in ranked],
                                   "gold_in_top5": bool(matching),
                                   "label_scope": "original synthetic positive; retrieval/packing can miss it"})
            (ROOT / "retrieved_context_test.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in diagnostic), encoding="utf-8")
            summary["retrieved_context_diagnostic"] = {"rows": len(diagnostic),
                "gold_in_top5": sum(r["gold_in_top5"] for r in diagnostic),
                "unsupported_rows_excluded": True,
                "gold_source_not_provided_in_prompt": True}
    (ROOT / "retrieval_baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
