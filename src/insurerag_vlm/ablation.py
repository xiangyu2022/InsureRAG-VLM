import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import ModelConfig
from .evaluation import compute_citation_precision, f1_score
from .pipeline import DocumentRetrievalPipeline
from .qa import _read_jsonl, compute_retrieval_metrics
from .visual import build_visual_index, compute_visual_retrieval_metrics


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _answer_metrics(
    pipeline: DocumentRetrievalPipeline,
    data_folder: Path,
    qa_path: Path,
    backend: str,
    top_k: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    examples = _read_jsonl(qa_path)
    if not examples:
        return {
            "backend": backend,
            "evaluated_count": 0,
            "f1": 0.0,
            "citation_precision": 0.0,
            "evidence_recall": 0.0,
            "unsupported_abstention_accuracy": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }, []

    total_f1 = 0.0
    total_citation_precision = 0.0
    total_evidence_recall = 0.0
    unsupported_total = 0
    unsupported_correct = 0
    latencies = []
    errors: List[Dict[str, Any]] = []

    for example in examples:
        start = time.perf_counter()
        result = pipeline.query_structured(example["question"], data_folder, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)

        if not example.get("answerable", True):
            unsupported_total += 1
            unsupported_correct += int(bool(result["abstain"]))
            continue

        answer = str(result["answer"])
        gold = str(example.get("answer") or example.get("ground_truth") or "")
        total_f1 += f1_score(answer, gold)

        predicted_sources = [citation["source"] for citation in result.get("citations", [])]
        gold_sources = example.get("evidence_sources") or example.get("citations") or []
        citation_precision = compute_citation_precision(predicted_sources, gold_sources)
        total_citation_precision += citation_precision
        evidence_hit = int(any(source in gold_sources for source in predicted_sources))
        total_evidence_recall += evidence_hit
        if evidence_hit == 0:
            errors.append(
                {
                    "backend": backend,
                    "qa_id": example.get("qa_id"),
                    "question": example.get("question"),
                    "gold_sources": gold_sources,
                    "predicted_sources": predicted_sources,
                    "answer": answer,
                }
            )

    answerable_count = max(1, sum(1 for item in examples if item.get("answerable", True)))
    latencies.sort()
    p50 = latencies[int(0.5 * (len(latencies) - 1))]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return {
        "backend": backend,
        "evaluated_count": len(examples),
        "f1": total_f1 / answerable_count,
        "citation_precision": total_citation_precision / answerable_count,
        "evidence_recall": total_evidence_recall / answerable_count,
        "unsupported_abstention_accuracy": unsupported_correct / unsupported_total if unsupported_total else 0.0,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }, errors


def run_ablation(
    data_folder: Path,
    qa_path: Path,
    output_dir: Path,
    index_dir: Path = Path("data"),
    visual_index_dir: Path = Path("data/03_index/colqwen2"),
    top_k: int = 5,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    local_config = ModelConfig(index_dir=index_dir, retrieval_model="local-hashing", vlm_model="local-extractive")
    local_pipeline = DocumentRetrievalPipeline(local_config)
    if not local_config.index_path.exists() or not local_config.metadata_path.exists():
        local_pipeline.build_index(data_folder)
    local_metrics = compute_retrieval_metrics(local_pipeline, data_folder, qa_path, top_k=top_k)
    retrieval_rows.append({"backend": "local_text", **local_metrics.__dict__})
    local_answer_metrics, local_errors = _answer_metrics(local_pipeline, data_folder, qa_path, "local_text", top_k)
    answer_rows.append(local_answer_metrics)
    error_rows.extend(local_errors)

    if os.environ.get("OPENAI_API_KEY"):
        openai_config = ModelConfig(
            index_dir=output_dir / "openai_index",
            retrieval_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            vlm_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            use_hf_api=False,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        openai_pipeline = DocumentRetrievalPipeline(openai_config)
        openai_pipeline.build_index(data_folder)
        openai_metrics = compute_retrieval_metrics(openai_pipeline, data_folder, qa_path, top_k=top_k)
        retrieval_rows.append({"backend": "openai_text", **openai_metrics.__dict__})
        answer_metrics, answer_errors = _answer_metrics(openai_pipeline, data_folder, qa_path, "openai_text", top_k)
        answer_rows.append(answer_metrics)
        error_rows.extend(answer_errors)

    page_manifest = visual_index_dir / "page_manifest.jsonl"
    if page_manifest.exists():
        for visual_backend in ["visual_stub", "local_image"]:
            build_visual_index(page_manifest, visual_index_dir, backend=visual_backend)
            visual_metrics = compute_visual_retrieval_metrics(qa_path, visual_index_dir, backend=visual_backend, top_k=top_k)
            retrieval_rows.append({"backend": visual_backend, **visual_metrics})
        for visual_backend in ["colqwen2_hf", "colpali_hf"]:
            if (visual_index_dir / f"{visual_backend}.pt").exists():
                visual_metrics = compute_visual_retrieval_metrics(
                    qa_path,
                    visual_index_dir,
                    backend=visual_backend,
                    top_k=top_k,
                )
                retrieval_rows.append({"backend": visual_backend, **visual_metrics})

    retrieval_path = output_dir / "retrieval_metrics.csv"
    answer_path = output_dir / "answer_metrics.csv"
    error_path = output_dir / "error_cases.jsonl"
    summary_path = output_dir / "summary.md"
    _write_csv(retrieval_rows, retrieval_path)
    _write_csv(answer_rows, answer_path)
    _write_jsonl(error_rows, error_path)
    _write_summary(retrieval_rows, answer_rows, summary_path)
    return {
        "retrieval_metrics": retrieval_path,
        "answer_metrics": answer_path,
        "error_cases": error_path,
        "summary": summary_path,
    }


def _write_summary(retrieval_rows: List[Dict[str, Any]], answer_rows: List[Dict[str, Any]], path: Path) -> None:
    lines = ["# InsureRAG-VLM Ablation Summary", "", "## Retrieval", ""]
    for row in retrieval_rows:
        lines.append(
            f"- {row['backend']}: Recall@1={row.get('recall_at_1', 0):.4f}, "
            f"Recall@5={row.get('recall_at_5', 0):.4f}, MRR@10={row.get('mrr_at_10', 0):.4f}, "
            f"nDCG@10={row.get('ndcg_at_10', 0):.4f}"
        )
    lines.extend(["", "## Answering", ""])
    for row in answer_rows:
        lines.append(
            f"- {row['backend']}: F1={row.get('f1', 0):.4f}, "
            f"Citation Precision={row.get('citation_precision', 0):.4f}, "
            f"Evidence Recall={row.get('evidence_recall', 0):.4f}, "
            f"Unsupported Abstention={row.get('unsupported_abstention_accuracy', 0):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
