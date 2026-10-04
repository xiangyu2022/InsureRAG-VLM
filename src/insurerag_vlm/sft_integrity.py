"""Small, dependency-free provenance checks for SFT evaluation.

These are conservative exact-identity checks, not a semantic deduplication system.
Fresh runs must start from a declared base model; prior adapter/checkpoint exposure
is accumulated, and missing historical provenance is never treated as clean.
"""
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROVENANCE_FILENAME = "sft_provenance.json"
IDENTITY_FIELDS = (
    "record_ids", "prompt_fingerprints", "question_source_fingerprints",
    "question_fingerprints", "source_fingerprints", "document_fingerprints",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def record_identities(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    identities = {key: set() for key in IDENTITY_FIELDS}
    count = 0
    records_without_source = 0
    for record in records:
        count += 1
        record_id = str(record.get("record_id") or "").strip()
        if record_id:
            identities["record_ids"].add(record_id)
        question = _normalized(record.get("question"))
        evidence = _normalized(record.get("evidence"))
        sources = [record.get("source")] if record.get("source") else []
        for key in ("evidence_sources", "retrieval_context_sources"):
            values = record.get(key) or []
            sources.extend(values if isinstance(values, list) else [values])
        sources = sorted({_normalized(source).replace("\\", "/") for source in sources if source})
        if not sources:
            records_without_source += 1
        for source in sources:
            identities["source_fingerprints"].add(_fingerprint(source))
            # Keep URL path/query intact; page fragments describe the same document.
            document = re.sub(r"#page=\d+.*$", "", source)
            identities["document_fingerprints"].add(_fingerprint(document))
            identities["question_source_fingerprints"].add(_fingerprint([question, source]))
        identities["question_fingerprints"].add(_fingerprint(question))
        # Do not include the answer: a relabeled copy of a seen prompt is still seen.
        identities["prompt_fingerprints"].add(_fingerprint([question, evidence, sources]))
    return {
        "record_count": count,
        "records_without_source": records_without_source,
        **{key: sorted(values) for key, values in identities.items()},
    }


def load_provenance(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(adapter_dir) / PROVENANCE_FILENAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or any(
        not isinstance(manifest.get(key), list) for key in IDENTITY_FIELDS
    ):
        raise ValueError(f"Invalid SFT provenance manifest: {path}")
    return manifest


def build_training_provenance(
    records: Iterable[Dict[str, Any]],
    dataset_path: Path,
    model_name: str,
    parent_dirs: Iterable[Path] = (),
) -> Dict[str, Any]:
    identities = record_identities(records)
    merged = {key: set(identities[key]) for key in IDENTITY_FIELDS}
    complete = True
    missing_sources = identities["records_without_source"]
    parents = []
    for parent_dir in dict.fromkeys(Path(path) for path in parent_dirs):
        parent = load_provenance(parent_dir)
        parents.append({
            "path": str(parent_dir),
            "manifest_sha256": file_sha256(parent_dir / PROVENANCE_FILENAME) if parent else None,
            "provenance_complete": bool(parent and parent.get("provenance_complete")),
        })
        if parent is None:
            complete = False
            continue
        complete = complete and bool(parent.get("provenance_complete")) and parent.get("model_name") == model_name
        missing_sources += int(parent.get("records_without_source", 0))
        for key in IDENTITY_FIELDS:
            merged[key].update(parent[key])
    return {
        "schema_version": 1,
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "current_run_record_count": identities["record_count"],
        "records_without_source": missing_sources,
        "provenance_complete": complete,
        "parents": parents,
        "scope": "Cumulative declared SFT exposure; base-model pretraining and semantic duplicates are not audited.",
        **{key: sorted(values) for key, values in merged.items()},
    }


def write_provenance(output_dir: Path, manifest: Dict[str, Any]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / PROVENANCE_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def audit_evaluation_records(
    records: Iterable[Dict[str, Any]],
    provenance: Optional[Dict[str, Any]],
    require_heldout: bool = False,
) -> Dict[str, Any]:
    identities = record_identities(records)
    audit = {
        "provenance_available": provenance is not None,
        "provenance_complete": bool(provenance and provenance.get("provenance_complete")),
        "evaluation_record_count": identities["record_count"],
        "evaluation_records_without_source": identities["records_without_source"],
        "training_records_without_source": provenance.get("records_without_source", 0) if provenance else None,
        "overlap_counts": {
            key: len(set(identities[key]) & set(provenance[key])) if provenance else None
            for key in IDENTITY_FIELDS
        },
        "limitations": "Exact normalized identities only; does not establish semantic independence or absence from base-model pretraining.",
    }
    blockers = []
    if not audit["provenance_complete"]:
        blockers.append("missing or incomplete adapter/checkpoint training lineage")
    if identities["records_without_source"] or audit["training_records_without_source"]:
        blockers.append("source identifiers missing; document disjointness cannot be checked")
    # Generic questions may legitimately recur in different unseen policy documents.
    for key in IDENTITY_FIELDS:
        if key != "question_fingerprints" and audit["overlap_counts"][key]:
            blockers.append(f"{key} overlap: {audit['overlap_counts'][key]}")
    audit["heldout_blockers"] = blockers
    audit["document_disjoint_sft_check_passed"] = not blockers
    if require_heldout and blockers:
        raise ValueError("Held-out evaluation rejected: " + "; ".join(blockers))
    return audit
