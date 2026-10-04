"""Reproducible corpus/evaluator diagnostic; does not train or load a GPU model."""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import _normalize_source, _read_jsonl, _retrieval_hit_positions
from src.insurerag_vlm.training_data import _source_doc_id, _split_doc_ids


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("data/04_curated"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/retrieval_eval/cpu_diagnostic"))
    args = parser.parse_args()
    os.environ["INSURERAG_USE_OLLAMA"] = "0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.data_folder / "sft_dataset.jsonl"
    sft = _read_jsonl(sft_path)
    answerable = [row for row in sft if row.get("answerable", True)]
    split = _split_doc_ids([_source_doc_id(row.get("source", "")) for row in sft], seed=42)
    test_rows = [
        {"qa_id": row["record_id"], "question": row["question"], "gold_sources": [row["source"]]}
        for row in answerable if split[_source_doc_id(row["source"])] == "test"
    ]
    write_jsonl(args.output_dir / "test_manifest.jsonl", test_rows)
    questions = defaultdict(set)
    for row in answerable:
        questions[row["question"]].add(_normalize_source(row["source"]))
    ambiguous = {question for question, sources in questions.items() if len(sources) > 1}
    suites = {
        "current_sft_document_test_split": test_rows,
        "legacy_59_development_examples": _read_jsonl(Path("reports/retrieval_eval/clean_exact_source.jsonl")),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU local-hashing retrieval and label-integrity diagnostic; no trained BGE or Qwen inference",
        "seed": 42,
        "data_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.data_folder.glob("*.jsonl"))},
        "corpus_audit": {
            "answerable_sft_records": len(answerable),
            "unique_answerable_questions": len(questions),
            "questions_with_multiple_gold_pages": len(ambiguous),
            "records_with_ambiguous_question": sum(row["question"] in ambiguous for row in answerable),
            "most_common_questions": Counter(row["question"] for row in answerable).most_common(5),
            "source_document_splits": dict(Counter(split.values())),
            "test_questions": len(test_rows),
            "test_unique_questions": len({row["question"] for row in test_rows}),
        },
        "results": {},
        "limitations": [
            "Question templates are reused across many source pages, making exact-source retrieval underdetermined without document context.",
            "The document split is regenerated from the current 3,850-record dataset; original historical train/test manifests and BGE weights are absent.",
            "No retriever training, LLM generation, adapter evaluation, image model, or GPU execution occurs in this diagnostic.",
            "The 59-example suite is selected from SFT data and is a development diagnostic, not a held-out generalization benchmark.",
            "dense_only and sparse_only modes retain the repository's metadata reranking, table signals, and graph expansion; these are pipeline ablations.",
            "Hit@k is the fraction of queries with any labeled page in the first k results, not answer correctness or hallucination rate.",
        ],
    }
    traces = []
    for mode in ("dense_only", "sparse_only", "hybrid_text"):
        pipeline = DocumentRetrievalPipeline(ModelConfig(
            retrieval_model="local-hashing", vlm_model="local-extractive",
            retrieval_mode=mode, corpus_source="curated", curated_dataset_dir=args.data_folder,
            enable_image_signal=False, index_dir=args.output_dir / "index" / mode,
        ))
        pipeline.build_index(args.data_folder)
        corpus = pipeline._load_hybrid_corpus(args.data_folder, include_images=False)
        sources = {_normalize_source(page["source"]) for page in corpus["pages"]}
        if mode == "dense_only":
            summary["corpus_audit"].update({
                "indexed_pages": len(corpus["pages"]), "indexed_snippets": len(corpus["snippets"]),
                "test_gold_sources_present": sum(_normalize_source(row["gold_sources"][0]) in sources for row in test_rows),
            })
        cache = {}
        for suite_name, rows in suites.items():
            hits = []
            for row in rows:
                question = row["question"]
                if question not in cache:
                    cache[question] = pipeline.rank_pages(question, args.data_folder, top_k=10)
                ranked = cache[question]
                positions = _retrieval_hit_positions(row, ranked)
                first_hit = min(positions) if positions else None
                hits.append(first_hit)
                traces.append({"suite": suite_name, "mode": mode, "qa_id": row["qa_id"],
                               "first_relevant_rank": first_hit, "ranked_sources": [page["source"] for page in ranked]})
            count = len(rows)
            summary["results"].setdefault(suite_name, {})[mode] = {
                "count": count, "hit_at_1": sum(hit == 1 for hit in hits) / count,
                "hit_at_5": sum(hit is not None and hit <= 5 for hit in hits) / count,
                "mrr_at_10": sum(1 / hit for hit in hits if hit is not None and hit <= 10) / count,
            }
    write_jsonl(args.output_dir / "predictions.jsonl", traces)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# CPU retrieval diagnostic", "", summary["purpose"] + ".", "", "## Corpus audit", ""]
    for key, value in summary["corpus_audit"].items():
        lines.append(f"- {key}: `{value}`")
    for suite, results in summary["results"].items():
        lines.extend(["", f"## {suite}", "", "| Mode | Examples | Hit@1 | Hit@5 | MRR@10 |", "| --- | ---: | ---: | ---: | ---: |"])
        for mode, metrics in results.items():
            lines.append(f"| {mode} | {metrics['count']} | {metrics['hit_at_1']:.4f} | {metrics['hit_at_5']:.4f} | {metrics['mrr_at_10']:.4f} |")
    lines.extend(["", "## Interpretation", ""] + [f"- {item}" for item in summary["limitations"]])
    lines.extend(["", "## Reproduce", "", "From the repository root, with requirements.txt installed:", "", "```bash", "python -m scripts.eval_cpu_diagnostic", "```", "", "The script rebuilds CPU indexes and writes input hashes, a test manifest, per-example ranked-source traces, and these summaries."])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"corpus_audit": summary["corpus_audit"], "results": summary["results"]}, indent=2))


if __name__ == "__main__":
    main()
