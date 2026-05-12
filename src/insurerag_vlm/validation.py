import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CURATED_FILES = {
    "rag_pages": "rag_pages.jsonl",
    "rag_snippets": "rag_snippets.jsonl",
    "rag_corpus": "rag_corpus.jsonl",
    "sft_dataset": "sft_dataset.jsonl",
}

RAG_REQUIRED_FIELDS = {
    "record_id",
    "record_type",
    "doc_id",
    "source_file",
    "text",
    "citation",
}
SFT_REQUIRED_FIELDS = {
    "record_id",
    "instruction",
    "question",
    "evidence",
    "answer",
    "source",
    "answerable",
}


@dataclass
class CuratedValidationConfig:
    dataset_dir: Path = Path("data/04_curated")
    output_dir: Path = Path("reports/research_proof")
    min_unsupported: int = 50
    min_sft_records: int = 1
    min_rag_records: int = 1
    update_summary: bool = True
    low_value_source_bytes: int = 250

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    return records


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_missing_field_errors(
    errors: List[Dict[str, Any]],
    dataset_name: str,
    records: Iterable[Dict[str, Any]],
    required_fields: Set[str],
) -> None:
    for index, record in enumerate(records):
        missing = sorted(field for field in required_fields if field not in record)
        empty = sorted(
            field
            for field in required_fields
            if field in record and isinstance(record.get(field), str) and not record.get(field, "").strip()
        )
        if missing or empty:
            errors.append(
                {
                    "dataset": dataset_name,
                    "record_index": index,
                    "record_id": record.get("record_id"),
                    "type": "missing_or_empty_required_fields",
                    "missing": missing,
                    "empty": empty,
                }
            )


def _source_document(source: str) -> str:
    source = str(source or "")
    if "#page=" in source:
        source = source.split("#page=", 1)[0]
    return source.strip()


def _count_duplicate_keys(records: Iterable[Dict[str, Any]], fields: Tuple[str, ...]) -> Dict[str, Any]:
    counts: Counter[Tuple[str, ...]] = Counter()
    for record in records:
        key = tuple(str(record.get(field, "")).strip() for field in fields)
        if any(key):
            counts[key] += 1
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "duplicate_groups": len(duplicate_groups),
        "duplicate_records": sum(duplicate_groups),
    }


def _split_leakage(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    splits_by_source: Dict[str, Set[str]] = defaultdict(set)
    split_records = 0
    for record in records:
        split = record.get("split")
        source = _source_document(record.get("source") or record.get("citation") or record.get("source_file"))
        if split and source:
            split_records += 1
            splits_by_source[source].add(str(split))
    leaked = {
        source: sorted(splits)
        for source, splits in splits_by_source.items()
        if len(splits) > 1
    }
    return {
        "checked_records": split_records,
        "leaked_source_count": len(leaked),
        "leaked_sources": leaked,
        "status": "not_applicable" if split_records == 0 else ("failed" if leaked else "passed"),
    }


def _low_value_sources(source_cache_dir: Path, max_bytes: int) -> List[Dict[str, Any]]:
    if not source_cache_dir.exists():
        return []
    rows = []
    for path in sorted(source_cache_dir.glob("*.txt")):
        size = path.stat().st_size
        if size <= max_bytes:
            rows.append({"path": str(path), "bytes": size})
    return rows


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASSED" if payload["passed"] else "FAILED"
    counts = payload["counts"]
    quality = payload["quality"]
    lines = [
        "# Curated Dataset Validation",
        "",
        f"- Status: {status}",
        f"- Dataset directory: `{payload['dataset_dir']}`",
        f"- RAG pages: {counts.get('rag_pages', 0)}",
        f"- RAG snippets: {counts.get('rag_snippets', 0)}",
        f"- RAG corpus records: {counts.get('rag_corpus', 0)}",
        f"- SFT records: {counts.get('sft_dataset', 0)}",
        f"- SFT answerable records: {quality['sft_answerable_count']}",
        f"- SFT unsupported records: {quality['sft_unsupported_count']}",
        f"- Errors: {len(payload['errors'])}",
        f"- Warnings: {len(payload['warnings'])}",
        "",
        "## Quality Checks",
        "",
        f"- Empty answerable SFT evidence: {quality['empty_answerable_evidence_count']}",
        f"- Duplicate SFT prompt groups: {quality['sft_duplicate_prompts']['duplicate_groups']}",
        f"- Duplicate RAG citation groups: {quality['rag_duplicate_citations']['duplicate_groups']}",
        f"- Split leakage status: {quality['split_leakage']['status']}",
    ]
    if payload["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in payload["errors"][:20]:
            lines.append(f"- `{error.get('type')}` in `{error.get('dataset')}` record `{error.get('record_id')}`")
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in payload["warnings"][:20]:
            lines.append(f"- {warning['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_payload(
    dataset_dir: Path,
    records_by_name: Dict[str, List[Dict[str, Any]]],
    warnings: List[Dict[str, Any]],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    rag_pages = records_by_name["rag_pages"]
    rag_snippets = records_by_name["rag_snippets"]
    rag_corpus = records_by_name["rag_corpus"]
    sft_records = records_by_name["sft_dataset"]
    rag_docs = sorted({str(record.get("source_file")) for record in rag_corpus if record.get("source_file")})
    sft_docs = sorted({_source_document(record.get("source")) for record in sft_records if record.get("source")})
    low_value_sources = [
        warning["source"]
        for warning in warnings
        if warning.get("type") == "low_value_source_cache"
    ]
    return {
        "rag_pages": {
            "path": str(dataset_dir / CURATED_FILES["rag_pages"]),
            "records": len(rag_pages),
            "source_documents": len(sorted({record.get("source_file") for record in rag_pages if record.get("source_file")})),
        },
        "rag_snippets": {
            "path": str(dataset_dir / CURATED_FILES["rag_snippets"]),
            "records": len(rag_snippets),
            "source_documents": len(sorted({record.get("source_file") for record in rag_snippets if record.get("source_file")})),
        },
        "rag_corpus": {
            "path": str(dataset_dir / CURATED_FILES["rag_corpus"]),
            "records": len(rag_corpus),
            "page_records": len(rag_pages),
            "snippet_records": len(rag_snippets),
            "source_documents": len(rag_docs),
            "documents": rag_docs,
        },
        "sft_dataset": {
            "path": str(dataset_dir / CURATED_FILES["sft_dataset"]),
            "records": len(sft_records),
            "answerable_records": quality["sft_answerable_count"],
            "unsupported_records": quality["sft_unsupported_count"],
            "source_documents": len(sft_docs),
            "documents": sft_docs,
        },
        "quality_policy": {
            "rag": "Hybrid RAG includes page-level parent records plus sentence-boundary snippet records for second-stage evidence selection.",
            "sft": "Uses evidence-grounded insurance questions, explicit citations, and unsupported counterexamples.",
            "excluded": "Pages with empty text extraction, weak insurance signal, contact-only content, or low-value cached source text are excluded or flagged.",
        },
        "validation": {
            "low_value_source_cache": low_value_sources,
            "split_leakage": quality["split_leakage"],
        },
    }


def validate_curated_record_sets(
    records_by_name: Dict[str, List[Dict[str, Any]]],
    min_unsupported: int = 50,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    rag_pages = records_by_name.get("rag_pages", [])
    rag_snippets = records_by_name.get("rag_snippets", [])
    rag_corpus = records_by_name.get("rag_corpus", [])
    sft_records = records_by_name.get("sft_dataset", [])

    _add_missing_field_errors(errors, "rag_pages", rag_pages, RAG_REQUIRED_FIELDS | {"page"})
    _add_missing_field_errors(errors, "rag_snippets", rag_snippets, RAG_REQUIRED_FIELDS | {"page", "parent_page_id"})
    _add_missing_field_errors(errors, "rag_corpus", rag_corpus, RAG_REQUIRED_FIELDS)
    _add_missing_field_errors(errors, "sft_dataset", sft_records, SFT_REQUIRED_FIELDS)

    page_ids = {record.get("record_id") for record in rag_pages}
    missing_parent_snippets = [
        record.get("record_id")
        for record in rag_snippets
        if record.get("parent_page_id") not in page_ids
    ]
    if missing_parent_snippets:
        errors.append(
            {
                "dataset": "rag_snippets",
                "type": "missing_parent_page",
                "count": len(missing_parent_snippets),
                "examples": missing_parent_snippets[:10],
            }
        )

    corpus_ids = {record.get("record_id") for record in rag_corpus}
    missing_from_corpus = sorted((page_ids | {record.get("record_id") for record in rag_snippets}) - corpus_ids)
    if missing_from_corpus:
        errors.append(
            {
                "dataset": "rag_corpus",
                "type": "missing_page_or_snippet_records",
                "count": len(missing_from_corpus),
                "examples": missing_from_corpus[:10],
            }
        )

    answerable = [record for record in sft_records if bool(record.get("answerable"))]
    unsupported = [record for record in sft_records if not bool(record.get("answerable"))]
    if len(unsupported) < min_unsupported:
        errors.append(
            {
                "dataset": "sft_dataset",
                "type": "too_few_unsupported_records",
                "count": len(unsupported),
                "minimum": min_unsupported,
            }
        )

    empty_answerable_evidence = [
        record.get("record_id")
        for record in answerable
        if not str(record.get("evidence", "")).strip()
    ]
    if empty_answerable_evidence:
        errors.append(
            {
                "dataset": "sft_dataset",
                "type": "empty_answerable_evidence",
                "count": len(empty_answerable_evidence),
                "examples": empty_answerable_evidence[:10],
            }
        )

    answerable_without_source = [
        record.get("record_id")
        for record in answerable
        if not str(record.get("source", "")).strip()
    ]
    if answerable_without_source:
        errors.append(
            {
                "dataset": "sft_dataset",
                "type": "answerable_missing_source",
                "count": len(answerable_without_source),
                "examples": answerable_without_source[:10],
            }
        )

    split_leakage = _split_leakage(sft_records)
    if split_leakage["status"] == "not_applicable":
        warnings.append(
            {
                "type": "split_metadata_absent",
                "message": "SFT records do not contain split metadata; source-level leakage cannot be checked for SFT training splits.",
            }
        )
    elif split_leakage["status"] == "failed":
        errors.append({"dataset": "sft_dataset", "type": "split_leakage", **split_leakage})

    quality = {
        "sft_answerable_count": len(answerable),
        "sft_unsupported_count": len(unsupported),
        "empty_answerable_evidence_count": len(empty_answerable_evidence),
        "sft_duplicate_prompts": _count_duplicate_keys(
            sft_records,
            ("instruction", "question", "evidence", "source"),
        ),
        "rag_duplicate_citations": _count_duplicate_keys(rag_corpus, ("record_type", "citation", "text")),
        "split_leakage": split_leakage,
    }

    return {
        "passed": not errors,
        "counts": {name: len(records) for name, records in records_by_name.items()},
        "quality": quality,
        "errors": errors,
        "warnings": warnings,
    }


def validate_curated_data(config: Optional[CuratedValidationConfig] = None) -> Dict[str, Any]:
    config = config or CuratedValidationConfig()
    dataset_dir = config.dataset_dir
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    records_by_name: Dict[str, List[Dict[str, Any]]] = {}
    file_fingerprints: Dict[str, Dict[str, Any]] = {}

    for name, filename in CURATED_FILES.items():
        path = dataset_dir / filename
        if not path.exists():
            errors.append({"dataset": name, "type": "missing_file", "path": str(path)})
            records_by_name[name] = []
            continue
        records = _read_jsonl(path)
        records_by_name[name] = records
        file_fingerprints[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    validation = validate_curated_record_sets(records_by_name, min_unsupported=config.min_unsupported)
    errors.extend(validation["errors"])
    warnings.extend(validation["warnings"])

    rag_corpus = records_by_name.get("rag_corpus", [])
    sft_records = records_by_name.get("sft_dataset", [])
    if len(rag_corpus) < config.min_rag_records:
        errors.append({"dataset": "rag_corpus", "type": "too_few_records", "count": len(rag_corpus)})
    if len(sft_records) < config.min_sft_records:
        errors.append({"dataset": "sft_dataset", "type": "too_few_records", "count": len(sft_records)})

    low_value_sources = _low_value_sources(dataset_dir / "source_cache", config.low_value_source_bytes)
    for row in low_value_sources:
        warnings.append(
            {
                "type": "low_value_source_cache",
                "source": row["path"],
                "bytes": row["bytes"],
                "message": f"Low-value cached source text should remain excluded or manifest-only: {row['path']} ({row['bytes']} bytes)",
            }
        )

    quality = validation["quality"]

    payload = {
        "passed": not errors,
        "dataset_dir": str(dataset_dir),
        "counts": {name: len(records) for name, records in records_by_name.items()},
        "files": file_fingerprints,
        "quality": quality,
        "errors": errors,
        "warnings": warnings,
    }

    validation_path = config.output_dir / "dataset_validation.json"
    summary_path = config.output_dir / "dataset_validation.md"
    _write_json(payload, validation_path)
    _write_markdown(payload, summary_path)
    payload["validation_path"] = str(validation_path)
    payload["validation_summary_path"] = str(summary_path)

    if config.update_summary and not errors:
        summary = _summary_payload(dataset_dir, records_by_name, warnings, quality)
        _write_json(summary, dataset_dir / "dataset_summary.json")
        payload["dataset_summary_path"] = str(dataset_dir / "dataset_summary.json")

    return payload
