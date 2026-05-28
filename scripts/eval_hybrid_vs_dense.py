#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import _normalize_source, compute_retrieval_metrics
from src.insurerag_vlm.query_understanding import understand_query

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
TARGET_CLAUSE_TYPES = {"limit", "premium", "deductible", "endorsement", "definition", "coverage"}


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


def _source_doc_id(source: str) -> str:
    return _normalize_source(source).split("#page=", 1)[0]


def _load_page_metadata(index_pages_path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(index_pages_path):
        source = _normalize_source(str(row.get("source") or ""))
        if source:
            lookup[source] = row
    return lookup


def _question_score(question: str, coverage_tags: List[str]) -> int:
    lowered = question.lower()
    score = 0
    for term in TARGET_TERMS:
        if term in lowered:
            score += 100
    score += 80 * len(coverage_tags or [])
    return score


def _select_targeted_examples(sft_path: Path, page_lookup: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: Dict[str, Tuple[int, Dict[str, Any], Dict[str, Any], List[str]]] = {}

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

        understanding = understand_query(question)
        target_coverages = list(understanding.target_coverages or [])
        targeted_question = any(term in lowered_question for term in TARGET_TERMS) or bool(target_coverages)

        document_type = str(page.get("document_type") or "")
        clause_type = str(page.get("primary_clause_type") or "")
        structure_hit = document_type in TARGET_DOC_TYPES or clause_type in TARGET_CLAUSE_TYPES
        if not (targeted_question and structure_hit):
            continue

        score = _question_score(question, target_coverages) + len(str(record.get("evidence") or ""))
        previous = selected.get(source)
        if previous is None or score > previous[0]:
            selected[source] = (score, record, page, target_coverages)

    rows: List[Dict[str, Any]] = []
    doc_counter: Counter[str] = Counter()
    clause_counter: Counter[str] = Counter()
    for source, (_, record, page, target_coverages) in selected.items():
        row = {
            "qa_id": record.get("record_id"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "evidence_sources": [source],
            "citations": [source],
            "gold_page_keys": [page.get("page_key")],
            "evidence_text": record.get("evidence") or "",
            "answerable": True,
            "question_type": "targeted_exact_source",
            "document_type": page.get("document_type"),
            "primary_clause_type": page.get("primary_clause_type"),
            "coverage_tags": target_coverages or page.get("coverage_tags") or [],
            "source_doc_id": _source_doc_id(source),
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


def _split_doc_ids(doc_ids: List[str], seed: int = 42, valid_ratio: float = 0.3) -> Dict[str, str]:
    unique_doc_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    rng = random.Random(seed)
    rng.shuffle(unique_doc_ids)
    total = len(unique_doc_ids)
    valid_n = max(1, int(total * valid_ratio))
    if valid_n >= total:
        valid_n = max(1, total - 1)
    split_map: Dict[str, str] = {}
    for idx, doc_id in enumerate(unique_doc_ids):
        split_map[doc_id] = "valid" if idx < valid_n else "test"
    return split_map


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


def _write_markdown(output_md: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Hybrid vs Dense Retrieval Evaluation",
        "",
        f"- Manifest root: `{payload['manifest_root']}`",
        f"- Retrieval model: `{payload['retrieval_model']}`",
        f"- Hybrid mode: `{payload['hybrid_mode']}`",
        f"- Dense mode: `{payload['dense_mode']}`",
        f"- Corpus source: `{payload['corpus_source']}`",
        f"- Image signal enabled: `{payload['enable_image_signal']}`",
        f"- Top-k: `{payload['top_k']}`",
        "",
        "## Composition",
        "",
        f"- Total examples: `{payload['manifest_metadata']['count']}`",
        f"- Document types: `{payload['manifest_metadata']['document_types']}`",
        f"- Primary clause types: `{payload['manifest_metadata']['primary_clause_types']}`",
        "",
    ]

    for split_name in ("valid", "test"):
        hybrid = payload["results"][split_name]["hybrid"]
        dense = payload["results"][split_name]["dense"]
        delta = payload["results"][split_name]["delta"]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- QA file: `{payload['results'][split_name]['qa_path']}`",
                "",
                "| metric | dense_only | hybrid | delta |",
                "| --- | ---: | ---: | ---: |",
                f"| evaluated_count | {dense['evaluated_count']} | {hybrid['evaluated_count']} | {delta['evaluated_count']} |",
                f"| recall_at_1 | {dense['recall_at_1']:.4f} | {hybrid['recall_at_1']:.4f} | {delta['recall_at_1']:.4f} |",
                f"| recall_at_5 | {dense['recall_at_5']:.4f} | {hybrid['recall_at_5']:.4f} | {delta['recall_at_5']:.4f} |",
                f"| mrr_at_10 | {dense['mrr_at_10']:.4f} | {hybrid['mrr_at_10']:.4f} | {delta['mrr_at_10']:.4f} |",
                f"| ndcg_at_10 | {dense['ndcg_at_10']:.4f} | {hybrid['ndcg_at_10']:.4f} | {delta['ndcg_at_10']:.4f} |",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hybrid RAG and dense-only RAG on an expanded targeted val/test benchmark.")
    parser.add_argument("--data-folder", type=Path, default=Path("data/04_curated"))
    parser.add_argument("--sft-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    parser.add_argument("--page-index", type=Path, default=Path("reports/training_data_dense/index/hybrid_pages.jsonl"))
    parser.add_argument("--manifest-root", type=Path, default=Path("reports/retrieval_eval/expanded_targeted"))
    parser.add_argument("--index-dir", type=Path, default=Path("reports/training_data_dense/index"))
    parser.add_argument("--retrieval-model", type=str, default="models/retrieval/bge-base-insurerag")
    parser.add_argument("--hybrid-mode", type=str, default="hybrid_multimodal")
    parser.add_argument("--dense-mode", type=str, default="dense_only")
    parser.add_argument("--corpus-source", type=str, default="curated")
    parser.add_argument("--enable-image-signal", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.3)
    parser.add_argument("--output-json", type=Path, default=Path("reports/retrieval_eval/hybrid_vs_dense.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/retrieval_eval/hybrid_vs_dense.md"))
    args = parser.parse_args()

    page_lookup = _load_page_metadata(args.page_index)
    rows, metadata = _select_targeted_examples(args.sft_path, page_lookup)
    split_map = _split_doc_ids([row["source_doc_id"] for row in rows], seed=args.seed, valid_ratio=args.valid_ratio)

    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for row in rows:
        split = split_map.get(row["source_doc_id"], "test")
        clean_row = {key: value for key, value in row.items() if key != "source_doc_id"}
        if split == "valid":
            valid_rows.append(clean_row)
        else:
            test_rows.append(clean_row)

    valid_path = args.manifest_root / "valid.jsonl"
    test_path = args.manifest_root / "test.jsonl"
    _write_jsonl(valid_rows, valid_path)
    _write_jsonl(test_rows, test_path)

    results: Dict[str, Any] = {}
    for split_name, qa_path in (("valid", valid_path), ("test", test_path)):
        dense = _run_eval(
            args.data_folder,
            qa_path,
            args.index_dir,
            args.retrieval_model,
            args.dense_mode,
            args.corpus_source,
            False,
            args.top_k,
        )
        hybrid = _run_eval(
            args.data_folder,
            qa_path,
            args.index_dir,
            args.retrieval_model,
            args.hybrid_mode,
            args.corpus_source,
            args.enable_image_signal,
            args.top_k,
        )
        delta = {key: hybrid[key] - dense[key] for key in dense.keys()}
        results[split_name] = {
            "qa_path": str(qa_path),
            "dense": dense,
            "hybrid": hybrid,
            "delta": delta,
        }

    payload = {
        "manifest_root": str(args.manifest_root),
        "manifest_metadata": {
            **metadata,
            "valid_count": len(valid_rows),
            "test_count": len(test_rows),
        },
        "retrieval_model": args.retrieval_model,
        "hybrid_mode": args.hybrid_mode,
        "dense_mode": args.dense_mode,
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
