import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .calibration import run_calibration
from .config import ModelConfig
from .evaluation import compute_citation_precision, f1_score
from .pipeline import DocumentRetrievalPipeline
from .preprocess import PageImagePreprocessConfig, preprocess_page_images
from .qa import _read_jsonl, generate_policy_qa_pairs
from .visual import build_visual_index, compute_visual_retrieval_metrics


@dataclass
class RunGpuBenchmarkConfig:
    data_folder: Path
    output_dir: Path = Path("reports/research_proof")
    backend: str = "colqwen2_local"
    target_count: int = 300
    unsupported_count: int = 50
    top_k: int = 10
    render_dpi: int = 200
    run_ocr: bool = False
    allow_backend_failures: bool = False


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_text(command: List[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_commit() -> str:
    return _run_text(["git", "rev-parse", "HEAD"]) or "unknown"


def _dataset_fingerprint(data_folder: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(path for path in Path(data_folder).rglob("*") if path.is_file())
    total_bytes = 0
    pdf_count = 0
    for path in files:
        relative = path.relative_to(data_folder).as_posix()
        stat = path.stat()
        total_bytes += stat.st_size
        pdf_count += int(path.suffix.lower() == ".pdf")
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "data_folder": str(Path(data_folder)),
        "file_count": len(files),
        "pdf_count": pdf_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _torch_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        info["torch_error"] = str(exc)
    nvidia_smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )
    if nvidia_smi:
        info["nvidia_smi"] = nvidia_smi
    return info


def _max_cuda_memory_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except Exception:
        return None


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _latency_percentiles(latencies: List[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    latencies = sorted(latencies)
    p50 = latencies[int(0.5 * (len(latencies) - 1))]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return p50, p95


def _text_retrieval_metrics(
    pipeline: DocumentRetrievalPipeline,
    data_folder: Path,
    qa_path: Path,
    top_k: int,
) -> Dict[str, Any]:
    examples = [item for item in _read_jsonl(qa_path) if item.get("answerable", True)]
    if not examples:
        return {
            "backend": "local_text",
            "evaluated_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    recall_1 = 0
    recall_5 = 0
    mrr_10 = 0.0
    ndcg_10 = 0.0
    latencies: List[float] = []
    for item in examples:
        start = time.perf_counter()
        ranked = pipeline.rank_pages(item["question"], data_folder, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        gold_sources = set(item.get("evidence_sources") or item.get("citations") or [])
        ranked_sources = [candidate["source"] for candidate in ranked]
        hit_positions = [
            idx + 1
            for idx, source in enumerate(ranked_sources[:10])
            if source in gold_sources
        ]
        recall_1 += int(bool(ranked_sources[:1]) and ranked_sources[0] in gold_sources)
        recall_5 += int(any(source in gold_sources for source in ranked_sources[:5]))
        if hit_positions:
            first_hit = hit_positions[0]
            mrr_10 += 1.0 / first_hit
            import math

            ndcg_10 += 1.0 / math.log2(first_hit + 1)

    count = len(examples)
    p50, p95 = _latency_percentiles(latencies)
    return {
        "backend": "local_text",
        "evaluated_count": count,
        "recall_at_1": recall_1 / count,
        "recall_at_5": recall_5 / count,
        "mrr_at_10": mrr_10 / count,
        "ndcg_at_10": ndcg_10 / count,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def _citation_hit(predicted_sources: List[str], gold_sources: List[str]) -> bool:
    return any(pred == gold or pred in gold or gold in pred for pred in predicted_sources for gold in gold_sources)


def _answer_reliability_metrics(
    pipeline: DocumentRetrievalPipeline,
    data_folder: Path,
    qa_path: Path,
    top_k: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    examples = _read_jsonl(qa_path)
    answerable = [item for item in examples if item.get("answerable", True)]
    unsupported = [item for item in examples if not item.get("answerable", True)]
    f1_total = 0.0
    citation_precision_total = 0.0
    evidence_recall_total = 0.0
    unsupported_correct = 0
    answered = 0
    error_rows: List[Dict[str, Any]] = []

    for example in examples:
        result = pipeline.query_structured(example["question"], data_folder, top_k=top_k, force_extractive=True)
        predicted_sources = [str(citation.get("source", "")) for citation in result.get("citations", [])]
        gold_sources = [str(source) for source in (example.get("evidence_sources") or example.get("citations") or [])]
        if not example.get("answerable", True):
            unsupported_correct += int(bool(result.get("abstain")))
            if not result.get("abstain"):
                error_rows.append(
                    {
                        "error_type": "unsupported_false_positive",
                        "qa_id": example.get("qa_id"),
                        "question": example.get("question"),
                        "answer": result.get("answer"),
                        "predicted_sources": predicted_sources,
                    }
                )
            continue

        answer = str(result.get("answer", ""))
        gold = str(example.get("answer") or example.get("ground_truth") or "")
        answer_f1 = f1_score(answer, gold)
        f1_total += answer_f1
        citation_precision = compute_citation_precision(predicted_sources, gold_sources)
        citation_precision_total += citation_precision
        evidence_hit = _citation_hit(predicted_sources, gold_sources)
        evidence_recall_total += int(evidence_hit)
        answered += int(not result.get("abstain"))

        ranked_sources = [str(row.get("source", "")) for row in result.get("source_ranking", [])[:top_k]]
        retrieval_hit = _citation_hit(ranked_sources, gold_sources)
        if not retrieval_hit:
            error_type = "retrieval_miss"
        elif predicted_sources and not evidence_hit:
            error_type = "citation_mismatch"
        elif answer_f1 <= 0.05 or result.get("abstain"):
            error_type = "weak_answer_extraction"
        else:
            continue
        error_rows.append(
            {
                "error_type": error_type,
                "qa_id": example.get("qa_id"),
                "question": example.get("question"),
                "gold_sources": gold_sources,
                "ranked_sources": ranked_sources[:5],
                "predicted_sources": predicted_sources,
                "answer_f1": answer_f1,
                "abstain": bool(result.get("abstain")),
                "citation_support_reason": result.get("citation_support_reason"),
            }
        )

    answerable_count = max(1, len(answerable))
    return {
        "backend": "local_text",
        "evaluated_count": len(examples),
        "answerable_count": len(answerable),
        "unsupported_count": len(unsupported),
        "f1": f1_total / answerable_count,
        "citation_precision": citation_precision_total / answerable_count,
        "evidence_recall": evidence_recall_total / answerable_count,
        "unsupported_abstention_accuracy": unsupported_correct / len(unsupported) if unsupported else 0.0,
        "coverage": answered / len(examples) if examples else 0.0,
    }, error_rows


def _summarize_errors(error_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for row in error_rows:
        key = str(row.get("error_type") or "unknown")
        summary[key] = summary.get(key, 0) + 1
    return summary


def _write_summary(
    path: Path,
    config: RunGpuBenchmarkConfig,
    manifest: Dict[str, Any],
    retrieval_rows: List[Dict[str, Any]],
    answer_metrics: Dict[str, Any],
    error_summary: Dict[str, int],
    backend_errors: List[Dict[str, Any]],
) -> None:
    lines = [
        "# InsureRAG-VLM GPU Benchmark",
        "",
        "## Run",
        "",
        f"- Data folder: `{config.data_folder}`",
        f"- Dataset hash: `{manifest['dataset']['sha256'][:12]}`",
        f"- PDFs: {manifest['dataset']['pdf_count']}",
        f"- Pages: {manifest['preprocess']['page_count']}",
        f"- QA rows: {manifest['qa']['qa_count']} ({manifest['qa']['unsupported_count']} unsupported)",
        f"- Primary backend: `{config.backend}`",
        f"- CUDA device: {manifest['environment'].get('cuda_device_name') or 'not available'}",
        "",
        "## Retrieval",
        "",
        "| Backend | Recall@1 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Index sec | Peak CUDA MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    index_times = {row["backend"]: row for row in manifest.get("indexing", [])}
    for row in retrieval_rows:
        timing = index_times.get(row["backend"], {})
        lines.append(
            f"| {row['backend']} | {row.get('recall_at_1', 0):.4f} | {row.get('recall_at_5', 0):.4f} | "
            f"{row.get('mrr_at_10', 0):.4f} | {row.get('ndcg_at_10', 0):.4f} | "
            f"{row.get('p50_latency_ms', 0):.1f} | {row.get('p95_latency_ms', 0):.1f} | "
            f"{timing.get('index_seconds', 0):.1f} | {timing.get('peak_cuda_memory_mb') or 0} |"
        )
    lines.extend(
        [
            "",
            "## Answering And Abstention",
            "",
            f"- F1: {answer_metrics.get('f1', 0):.4f}",
            f"- Citation precision: {answer_metrics.get('citation_precision', 0):.4f}",
            f"- Evidence recall: {answer_metrics.get('evidence_recall', 0):.4f}",
            f"- Unsupported abstention accuracy: {answer_metrics.get('unsupported_abstention_accuracy', 0):.4f}",
            f"- Coverage: {answer_metrics.get('coverage', 0):.4f}",
            "",
            "## Error Groups",
            "",
        ]
    )
    if error_summary:
        for key, value in sorted(error_summary.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No reliability errors recorded.")
    if backend_errors:
        lines.extend(["", "## Backend Failures", ""])
        for row in backend_errors:
            lines.append(f"- {row['backend']}: {row['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gpu_benchmark(config: RunGpuBenchmarkConfig) -> Dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "artifacts"
    qa_dir = artifact_root / "qa"
    text_index_dir = artifact_root / "text_index"
    preprocess_root = artifact_root / "preprocess"

    started = time.perf_counter()
    manifest: Dict[str, Any] = {
        "command": " ".join(["python", "main.py", "run-gpu-benchmark"]),
        "git_commit": _git_commit(),
        "environment": _torch_environment(),
        "dataset": _dataset_fingerprint(config.data_folder),
        "config": config.__dict__,
        "indexing": [],
    }

    preprocess_start = time.perf_counter()
    preprocess_result = preprocess_page_images(
        PageImagePreprocessConfig(
            input_dir=config.data_folder,
            output_root=preprocess_root,
            render_dpi=config.render_dpi,
            run_ocr=config.run_ocr,
        )
    )
    manifest["preprocess"] = {
        "document_count": preprocess_result.document_count,
        "page_count": preprocess_result.page_count,
        "seconds": time.perf_counter() - preprocess_start,
        "page_manifest_path": str(preprocess_result.page_manifest_path),
    }

    qa_result = generate_policy_qa_pairs(
        data_folder=config.data_folder,
        output_dir=qa_dir,
        target_count=config.target_count,
        unsupported_count=config.unsupported_count,
    )
    qa_rows = _read_jsonl(qa_result.qa_path)
    manifest["qa"] = {
        "qa_path": str(qa_result.qa_path),
        "qa_count": qa_result.qa_count,
        "answerable_count": sum(1 for row in qa_rows if row.get("answerable", True)),
        "unsupported_count": sum(1 for row in qa_rows if not row.get("answerable", True)),
        "hard_negative_count": qa_result.hard_negative_count,
        "splits_path": str(qa_result.splits_path) if qa_result.splits_path else None,
    }

    retrieval_rows: List[Dict[str, Any]] = []
    backend_errors: List[Dict[str, Any]] = []

    text_pipeline = DocumentRetrievalPipeline(
        ModelConfig(index_dir=text_index_dir, retrieval_model="local-hashing", vlm_model="local-extractive")
    )
    text_start = time.perf_counter()
    text_pipeline.build_index(config.data_folder)
    manifest["indexing"].append(
        {
            "backend": "local_text",
            "index_seconds": time.perf_counter() - text_start,
            "model_name": "local-hashing",
            "dtype": "float32",
            "batch_size": None,
            "peak_cuda_memory_mb": None,
        }
    )
    retrieval_rows.append(_text_retrieval_metrics(text_pipeline, config.data_folder, qa_result.qa_path, config.top_k))

    visual_index_dir = preprocess_root / "03_index" / "colqwen2"
    for backend in ["local_image", config.backend]:
        if backend == "local_text" or any(row.get("backend") == backend for row in retrieval_rows):
            continue
        index_start = time.perf_counter()
        _reset_cuda_peak_memory()
        try:
            build_visual_index(preprocess_result.page_manifest_path, visual_index_dir, backend=backend)
            index_seconds = time.perf_counter() - index_start
            metrics = compute_visual_retrieval_metrics(qa_result.qa_path, visual_index_dir, backend=backend, top_k=config.top_k)
            retrieval_rows.append({"backend": backend, **metrics})
            manifest["indexing"].append(
                {
                    "backend": backend,
                    "index_seconds": index_seconds,
                    "model_name": os.environ.get("INSURERAG_COLQWEN2_MODEL" if "qwen" in backend else "INSURERAG_COLPALI_MODEL"),
                    "dtype": os.environ.get("INSURERAG_VISUAL_DTYPE", "bfloat16"),
                    "batch_size": int(os.environ.get("INSURERAG_VISUAL_BATCH_SIZE", "2")),
                    "peak_cuda_memory_mb": _max_cuda_memory_mb(),
                }
            )
        except Exception as exc:
            backend_errors.append({"backend": backend, "error": str(exc)})
            if backend == config.backend and not config.allow_backend_failures:
                manifest["backend_errors"] = backend_errors
                _write_json(manifest, output_dir / "experiment_manifest.json")
                raise

    answer_metrics, error_rows = _answer_reliability_metrics(text_pipeline, config.data_folder, qa_result.qa_path, top_k=min(config.top_k, 5))
    error_summary = _summarize_errors(error_rows)

    calibration_outputs = run_calibration(
        data_folder=config.data_folder,
        qa_path=qa_result.qa_path,
        output_dir=output_dir / "calibration",
        index_dir=text_index_dir,
        top_k=min(config.top_k, 5),
    )

    manifest["backend_errors"] = backend_errors
    manifest["calibration"] = {name: str(path) for name, path in calibration_outputs.items()}
    manifest["total_seconds"] = time.perf_counter() - started

    retrieval_path = output_dir / "retrieval_metrics.csv"
    answer_path = output_dir / "answer_metrics.json"
    errors_path = output_dir / "error_cases_by_type.jsonl"
    manifest_path = output_dir / "experiment_manifest.json"
    summary_path = output_dir / "summary.md"
    _write_csv(retrieval_rows, retrieval_path)
    _write_json(answer_metrics, answer_path)
    _write_jsonl(error_rows, errors_path)
    _write_json(manifest, manifest_path)
    _write_summary(summary_path, config, manifest, retrieval_rows, answer_metrics, error_summary, backend_errors)

    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "retrieval_metrics": retrieval_path,
        "answer_metrics": answer_path,
        "error_cases": errors_path,
        "calibration_summary": calibration_outputs["summary"],
    }
