#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import compute_retrieval_metrics


def _metrics_to_dict(metrics) -> Dict[str, float]:
    return {
        "evaluated_count": metrics.evaluated_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_5": metrics.recall_at_5,
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
    }


def _run_eval(
    data_folder: Path,
    qa_path: Path,
    index_dir: Path,
    retrieval_model: str,
    retrieval_mode: str,
    corpus_source: str,
    enable_image_signal: bool,
    top_k: int,
) -> Dict[str, float]:
    pipeline = DocumentRetrievalPipeline(
        ModelConfig(
            retrieval_model=retrieval_model,
            retrieval_mode=retrieval_mode,
            corpus_source=corpus_source,
            enable_image_signal=enable_image_signal,
            index_dir=index_dir,
        )
    )
    metrics = compute_retrieval_metrics(pipeline, data_folder, qa_path, top_k=top_k)
    return _metrics_to_dict(metrics)


def _format_percent(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown(output_md: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Retrieval Before/After Evaluation",
        "",
        f"- Data folder: `{payload['data_folder']}`",
        f"- Retrieval mode: `{payload['retrieval_mode']}`",
        f"- Corpus source: `{payload['corpus_source']}`",
        f"- Image signal enabled: `{payload['enable_image_signal']}`",
        f"- Top-k: `{payload['top_k']}`",
        "",
    ]

    for split_name in ("dev", "test"):
        before = payload["results"][split_name]["before"]
        after = payload["results"][split_name]["after"]
        delta = payload["results"][split_name]["delta"]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- QA file: `{payload['results'][split_name]['qa_path']}`",
                "",
                "| metric | before | after | delta |",
                "| --- | ---: | ---: | ---: |",
                f"| evaluated_count | {before['evaluated_count']} | {after['evaluated_count']} | {delta['evaluated_count']} |",
                f"| recall_at_1 | {_format_percent(before['recall_at_1'])} | {_format_percent(after['recall_at_1'])} | {_format_percent(delta['recall_at_1'])} |",
                f"| recall_at_5 | {_format_percent(before['recall_at_5'])} | {_format_percent(after['recall_at_5'])} | {_format_percent(delta['recall_at_5'])} |",
                f"| mrr_at_10 | {_format_percent(before['mrr_at_10'])} | {_format_percent(after['mrr_at_10'])} | {_format_percent(delta['mrr_at_10'])} |",
                f"| ndcg_at_10 | {_format_percent(before['ndcg_at_10'])} | {_format_percent(after['ndcg_at_10'])} | {_format_percent(delta['ndcg_at_10'])} |",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval metrics before and after dense-retriever training.")
    parser.add_argument("--data-folder", type=Path, default=Path("data/04_curated"))
    parser.add_argument("--before-qa-dev", type=Path, default=Path("reports/training_data_seed/retrieval_dev.jsonl"))
    parser.add_argument("--before-qa-test", type=Path, default=Path("reports/training_data_seed/retrieval_test.jsonl"))
    parser.add_argument("--after-qa-dev", type=Path, default=Path("reports/training_data_dense/retrieval_dev.jsonl"))
    parser.add_argument("--after-qa-test", type=Path, default=Path("reports/training_data_dense/retrieval_test.jsonl"))
    parser.add_argument("--before-index-dir", type=Path, default=Path("reports/training_data_seed/index"))
    parser.add_argument("--after-index-dir", type=Path, default=Path("reports/training_data_dense/index"))
    parser.add_argument("--before-retrieval-model", type=str, default="local-hashing")
    parser.add_argument("--after-retrieval-model", type=str, default="models/retrieval/bge-base-insurerag")
    parser.add_argument("--retrieval-mode", type=str, default="hybrid_multimodal")
    parser.add_argument("--corpus-source", type=str, default="curated")
    parser.add_argument("--enable-image-signal", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("reports/retrieval_eval/before_after.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/retrieval_eval/before_after.md"))
    args = parser.parse_args()

    results = {}
    for split_name, before_qa, after_qa in (
        ("dev", args.before_qa_dev, args.after_qa_dev),
        ("test", args.before_qa_test, args.after_qa_test),
    ):
        before = _run_eval(
            args.data_folder,
            before_qa,
            args.before_index_dir,
            args.before_retrieval_model,
            args.retrieval_mode,
            args.corpus_source,
            args.enable_image_signal,
            args.top_k,
        )
        after = _run_eval(
            args.data_folder,
            after_qa,
            args.after_index_dir,
            args.after_retrieval_model,
            args.retrieval_mode,
            args.corpus_source,
            args.enable_image_signal,
            args.top_k,
        )
        delta = {
            key: (after[key] - before[key]) if isinstance(after[key], (int, float)) else None
            for key in after.keys()
        }
        results[split_name] = {
            "qa_path": str(after_qa),
            "before": before,
            "after": after,
            "delta": delta,
        }

    payload = {
        "data_folder": str(args.data_folder),
        "retrieval_mode": args.retrieval_mode,
        "corpus_source": args.corpus_source,
        "enable_image_signal": args.enable_image_signal,
        "top_k": args.top_k,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
