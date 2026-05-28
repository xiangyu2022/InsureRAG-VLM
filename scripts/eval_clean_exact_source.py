#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import _normalize_source, compute_retrieval_metrics

TARGET_TERMS = [
    "deductible",
    "premium",
    "declaration",
    "endorsement",
    "replacement cost",
    "actual cash value",
    "limit",
    "limits",
    "sublimit",
    "retention",
    "coinsurance",
]

BLACKLIST_QUESTIONS = {
    "what limitation or exclusion is supported by this evidence?",
    "what coverage information is explained by the evidence?",
    "what does the evidence say about exclusions?",
    "how should this exclusion-related point be explained?",
    "what should the reader understand about exclusions here?",
}

TARGET_DOC_TYPES = {"declarations", "endorsement", "schedule"}
TARGET_CLAUSE_TYPES = {"limit", "premium", "deductible", "endorsement", "definition"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_page_metadata(index_pages_path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(index_pages_path):
        source = _normalize_source(str(row.get("source") or ""))
        if source:
            lookup[source] = row
    return lookup


def _question_specificity_score(question: str) -> int:
    lowered = question.lower()
    score = 0
    for term in TARGET_TERMS:
        if term in lowered:
            score += 100
    return score


def _select_clean_examples(sft_path: Path, page_lookup: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: Dict[str, Tuple[int, Dict[str, Any], Dict[str, Any]]] = {}

    for record in _read_jsonl(sft_path):
        if not bool(record.get("answerable", True)):
            continue
        source = _normalize_source(str(record.get("source") or ""))
        if not source:
            continue
        page = page_lookup.get(source)
        if not page:
            continue

        question = str(record.get("question") or "")
        lowered_question = question.lower().strip()
        if lowered_question in BLACKLIST_QUESTIONS:
            continue
        if not any(term in lowered_question for term in TARGET_TERMS):
            continue

        document_type = str(page.get("document_type") or "")
        clause_type = str(page.get("primary_clause_type") or "")
        if document_type not in TARGET_DOC_TYPES and clause_type not in TARGET_CLAUSE_TYPES:
            continue

        score = _question_specificity_score(question) + len(str(record.get("evidence") or ""))
        previous = selected.get(source)
        if previous is None or score > previous[0]:
            selected[source] = (score, record, page)

    rows: List[Dict[str, Any]] = []
    doc_counter: Counter[str] = Counter()
    clause_counter: Counter[str] = Counter()
    for source, (_, record, page) in sorted(selected.items(), key=lambda item: item[0]):
        row = {
            "qa_id": record.get("record_id"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "evidence_sources": [source],
            "citations": [source],
            "gold_page_keys": [page.get("page_key")],
            "evidence_text": record.get("evidence") or "",
            "answerable": True,
            "question_type": "exact_source_clean",
            "document_type": page.get("document_type"),
            "primary_clause_type": page.get("primary_clause_type"),
            "coverage_tags": page.get("coverage_tags") or [],
        }
        rows.append(row)
        doc_counter[str(page.get("document_type") or "")] += 1
        clause_counter[str(page.get("primary_clause_type") or "")] += 1

    metadata = {
        "count": len(rows),
        "document_types": dict(doc_counter),
        "primary_clause_types": dict(clause_counter),
    }
    return rows, metadata


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
    return {
        "evaluated_count": metrics.evaluated_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_5": metrics.recall_at_5,
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
    }


def _write_markdown(output_md: Path, payload: Dict[str, Any]) -> None:
    before = payload["before"]
    after = payload["after"]
    delta = payload["delta"]
    lines = [
        "# Clean Exact-Source Retrieval Evaluation",
        "",
        f"- Manifest: `{payload['manifest_path']}`",
        f"- Example count: `{payload['manifest_metadata']['count']}`",
        f"- Retrieval mode: `{payload['retrieval_mode']}`",
        f"- Corpus source: `{payload['corpus_source']}`",
        f"- Image signal enabled: `{payload['enable_image_signal']}`",
        f"- Top-k: `{payload['top_k']}`",
        "",
        "## Composition",
        "",
        f"- Document types: `{payload['manifest_metadata']['document_types']}`",
        f"- Primary clause types: `{payload['manifest_metadata']['primary_clause_types']}`",
        "",
        "## Before vs After",
        "",
        "| metric | before | after | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| evaluated_count | {before['evaluated_count']} | {after['evaluated_count']} | {delta['evaluated_count']} |",
        f"| recall_at_1 | {before['recall_at_1']:.4f} | {after['recall_at_1']:.4f} | {delta['recall_at_1']:.4f} |",
        f"| recall_at_5 | {before['recall_at_5']:.4f} | {after['recall_at_5']:.4f} | {delta['recall_at_5']:.4f} |",
        f"| mrr_at_10 | {before['mrr_at_10']:.4f} | {after['mrr_at_10']:.4f} | {delta['mrr_at_10']:.4f} |",
        f"| ndcg_at_10 | {before['ndcg_at_10']:.4f} | {after['ndcg_at_10']:.4f} | {delta['ndcg_at_10']:.4f} |",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate a cleaned exact-source retrieval benchmark.")
    parser.add_argument("--data-folder", type=Path, default=Path("data/04_curated"))
    parser.add_argument("--sft-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    parser.add_argument("--page-index", type=Path, default=Path("reports/training_data_dense/index/hybrid_pages.jsonl"))
    parser.add_argument("--manifest-path", type=Path, default=Path("reports/retrieval_eval/clean_exact_source.jsonl"))
    parser.add_argument("--before-index-dir", type=Path, default=Path("reports/training_data_seed/index"))
    parser.add_argument("--after-index-dir", type=Path, default=Path("reports/training_data_dense/index"))
    parser.add_argument("--before-retrieval-model", type=str, default="local-hashing")
    parser.add_argument("--after-retrieval-model", type=str, default="models/retrieval/bge-base-insurerag")
    parser.add_argument("--retrieval-mode", type=str, default="hybrid_multimodal")
    parser.add_argument("--corpus-source", type=str, default="curated")
    parser.add_argument("--enable-image-signal", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("reports/retrieval_eval/clean_exact_source_before_after.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/retrieval_eval/clean_exact_source_before_after.md"))
    args = parser.parse_args()

    page_lookup = _load_page_metadata(args.page_index)
    manifest_rows, manifest_metadata = _select_clean_examples(args.sft_path, page_lookup)
    _write_jsonl(manifest_rows, args.manifest_path)

    before = _run_eval(
        args.data_folder,
        args.manifest_path,
        args.before_index_dir,
        args.before_retrieval_model,
        args.retrieval_mode,
        args.corpus_source,
        args.enable_image_signal,
        args.top_k,
    )
    after = _run_eval(
        args.data_folder,
        args.manifest_path,
        args.after_index_dir,
        args.after_retrieval_model,
        args.retrieval_mode,
        args.corpus_source,
        args.enable_image_signal,
        args.top_k,
    )
    delta = {key: after[key] - before[key] for key in before.keys()}
    payload = {
        "manifest_path": str(args.manifest_path),
        "manifest_metadata": manifest_metadata,
        "retrieval_mode": args.retrieval_mode,
        "corpus_source": args.corpus_source,
        "enable_image_signal": args.enable_image_signal,
        "top_k": args.top_k,
        "before": before,
        "after": after,
        "delta": delta,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
