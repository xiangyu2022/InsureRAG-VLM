import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import ModelConfig
from .evaluation import f1_score
from .pipeline import DocumentRetrievalPipeline
from .qa import _read_jsonl


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
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _citation_hit(predicted_sources: List[str], gold_sources: List[str]) -> bool:
    for pred in predicted_sources:
        for gold in gold_sources:
            if pred == gold or pred in gold or gold in pred:
                return True
    return False


def _score_examples(
    data_folder: Path,
    qa_path: Path,
    index_dir: Path,
    top_k: int,
) -> List[Dict[str, Any]]:
    examples = _read_jsonl(qa_path)
    config = ModelConfig(index_dir=index_dir, retrieval_model="local-hashing", vlm_model="local-extractive")
    pipeline = DocumentRetrievalPipeline(config)
    if not config.index_path.exists() or not config.metadata_path.exists():
        pipeline.build_index(data_folder)

    rows: List[Dict[str, Any]] = []
    for example in examples:
        start = time.perf_counter()
        result = pipeline.query_structured(example["question"], data_folder, top_k=top_k, force_extractive=True)
        latency_ms = (time.perf_counter() - start) * 1000
        answerable = bool(example.get("answerable", True))
        predicted_sources = [c.get("source", "") for c in result.get("citations", [])]
        gold_sources = example.get("evidence_sources") or example.get("citations") or []
        answer_f1 = f1_score(str(result.get("answer", "")), str(example.get("answer") or example.get("ground_truth") or ""))
        evidence_hit = _citation_hit(predicted_sources, gold_sources)
        correct_when_answered = (answer_f1 > 0.05 and evidence_hit) if answerable else False
        rows.append(
            {
                "qa_id": example.get("qa_id"),
                "question": example.get("question"),
                "answerable": answerable,
                "confidence": float(result.get("confidence", 0.0)),
                "pipeline_abstain": bool(result.get("abstain")),
                "answer_f1": answer_f1,
                "evidence_hit": evidence_hit,
                "correct_when_answered": correct_when_answered,
                "predicted_sources": predicted_sources,
                "gold_sources": gold_sources,
                "latency_ms": latency_ms,
            }
        )
    return rows


def _threshold_curve(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    thresholds = [round(i / 20, 2) for i in range(0, 21)]
    answerable_count = sum(1 for row in rows if row["answerable"])
    unsupported_count = sum(1 for row in rows if not row["answerable"])
    curve: List[Dict[str, Any]] = []

    for threshold in thresholds:
        answered = [row for row in rows if row["confidence"] >= threshold]
        answered_answerable = [row for row in answered if row["answerable"]]
        answered_unsupported = [row for row in answered if not row["answerable"]]
        correct_answered = [row for row in answered_answerable if row["correct_when_answered"]]
        errors = len(answered_answerable) - len(correct_answered) + len(answered_unsupported)
        unsupported_abstained = unsupported_count - len(answered_unsupported)
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(answered) / len(rows) if rows else 0.0,
                "answerable_coverage": len(answered_answerable) / answerable_count if answerable_count else 0.0,
                "selective_risk": errors / len(answered) if answered else 0.0,
                "answerable_accuracy_when_answered": len(correct_answered) / len(answered_answerable) if answered_answerable else 0.0,
                "unsupported_abstention_accuracy": unsupported_abstained / unsupported_count if unsupported_count else 0.0,
                "answered_count": len(answered),
                "error_count": errors,
            }
        )
    return curve


def run_calibration(
    data_folder: Path,
    qa_path: Path,
    output_dir: Path = Path("reports/calibration"),
    index_dir: Path = Path("data"),
    top_k: int = 3,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_rows = _score_examples(data_folder, qa_path, index_dir=index_dir, top_k=top_k)
    curve_rows = _threshold_curve(score_rows)

    scores_path = output_dir / "calibration_scores.jsonl"
    curve_path = output_dir / "calibration_curve.csv"
    summary_path = output_dir / "summary.md"
    _write_jsonl(score_rows, scores_path)
    _write_csv(curve_rows, curve_path)
    _write_summary(score_rows, curve_rows, summary_path)
    return {
        "calibration_scores": scores_path,
        "calibration_curve": curve_path,
        "summary": summary_path,
    }


def _write_summary(score_rows: List[Dict[str, Any]], curve_rows: List[Dict[str, Any]], path: Path) -> None:
    if not score_rows:
        path.write_text("# Calibration Summary\n\nNo examples found.\n", encoding="utf-8")
        return
    nonzero_curve = [row for row in curve_rows if row["answered_count"] > 0] or curve_rows
    acceptable = [row for row in nonzero_curve if row["selective_risk"] <= 0.35]
    if acceptable:
        target = max(acceptable, key=lambda row: (row["coverage"], row["unsupported_abstention_accuracy"]))
    else:
        target = min(nonzero_curve, key=lambda row: (row["selective_risk"], -row["coverage"]))
    unsupported = [row for row in score_rows if not row["answerable"]]
    answerable = [row for row in score_rows if row["answerable"]]
    lines = [
        "# InsureRAG-VLM Calibration Summary",
        "",
        f"- Examples: {len(score_rows)}",
        f"- Answerable examples: {len(answerable)}",
        f"- Unsupported examples: {len(unsupported)}",
        f"- Suggested threshold: {target['threshold']:.2f}",
        f"- Coverage at threshold: {target['coverage']:.4f}",
        f"- Selective risk at threshold: {target['selective_risk']:.4f}",
        f"- Unsupported abstention accuracy: {target['unsupported_abstention_accuracy']:.4f}",
        "",
        "This report is a local deterministic calibration smoke test. Use a larger validation set before treating the threshold as production-ready.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
