import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


POLICY_SECTION_PATTERNS = [
    "declarations",
    "liability",
    "coverage",
    "deductible",
    "exclusions",
    "endorsement",
    "definitions",
    "conditions",
    "duties",
    "limits",
    "loss",
    "claim",
]


@dataclass
class PageImagePreprocessConfig:
    input_dir: Path
    output_root: Path = Path("data")
    render_dpi: int = 200
    run_ocr: bool = False
    source_type: str = "public_policy_pdf"
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    seed: int = 42

    @property
    def page_image_dir(self) -> Path:
        return self.output_root / "01_interim" / "page_images"

    @property
    def processed_dir(self) -> Path:
        return self.output_root / "02_processed"

    @property
    def index_dir(self) -> Path:
        return self.output_root / "03_index" / "colqwen2"

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "manifests"


@dataclass
class PageImagePreprocessResult:
    documents_path: Path
    pages_path: Path
    ocr_aux_path: Path
    page_manifest_path: Path
    retrieval_pairs_path: Path
    splits_path: Path
    document_count: int
    page_count: int


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _write_table(records: List[Dict[str, Any]], parquet_path: Path) -> Path:
    jsonl_path = parquet_path.with_suffix(".jsonl")
    note_path = parquet_path.with_suffix(".parquet.unwritten.txt")
    _write_jsonl(records, jsonl_path)
    if not records:
        return jsonl_path

    try:
        import pandas as pd
    except ImportError:
        note_path.write_text(
            "Parquet was not written because pandas/pyarrow is not installed. "
            f"Install project requirements and use {jsonl_path.name} meanwhile.\n",
            encoding="utf-8",
        )
        return jsonl_path

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
    if note_path.exists():
        note_path.unlink()
    return parquet_path


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "document"


def _infer_family_and_version(stem: str) -> tuple[str, str]:
    slug = _slugify(stem)
    version_match = re.search(r"(v\d+|\d{4}[a-z]?|[a-z]{1,4}\d{2,})", slug)
    if not version_match:
        return slug, slug
    version_id = version_match.group(1)
    family = slug[: version_match.start()].strip("_") or slug
    return family, version_id


def _extract_section_hint(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:30]:
        lowered = line.lower()
        if any(pattern in lowered for pattern in POLICY_SECTION_PATTERNS):
            return line[:120]
    for line in lines[:10]:
        if 4 <= len(line) <= 120:
            return line
    return None


def _ocr_page_text(image_path: Path) -> Dict[str, Any]:
    try:
        from .ocr import extract_text_from_image

        return {
            "ocr_engine": "tesseract",
            "ocr_text": extract_text_from_image(image_path),
            "ocr_error": None,
        }
    except Exception as exc:
        return {
            "ocr_engine": "tesseract",
            "ocr_text": "",
            "ocr_error": str(exc),
        }


def _render_pdf_to_page_records(
    pdf_path: Path,
    config: PageImagePreprocessConfig,
    doc_id: str,
    document_record: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for page-image preprocessing. Install it with `pip install PyMuPDF`."
        ) from exc

    page_records: List[Dict[str, Any]] = []
    ocr_records: List[Dict[str, Any]] = []
    config.page_image_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    for page_index, page in enumerate(document):
        page_number = page_index + 1
        page_id = f"{doc_id}_p{page_number:04d}"
        image_path = config.page_image_dir / f"{page_id}.png"

        pix = page.get_pixmap(dpi=config.render_dpi, annots=False)
        pix.save(str(image_path))

        text_layer = page.get_text("text").strip()
        image_sha256 = compute_sha256(image_path)
        section_hint = _extract_section_hint(text_layer)

        page_record = {
            "page_id": page_id,
            "doc_id": doc_id,
            "page_index": page_index,
            "page_number": page_number,
            "image_path": str(image_path),
            "width_px": pix.width,
            "height_px": pix.height,
            "render_dpi": config.render_dpi,
            "policy_family_id": document_record["policy_family_id"],
            "version_id": document_record["version_id"],
            "source": str(pdf_path),
            "source_type": config.source_type,
            "sha256": image_sha256,
            "text_layer_chars": len(text_layer),
            "has_text_layer": bool(text_layer),
            "section_hint": section_hint,
        }
        page_records.append(page_record)

        ocr_record = {
            "page_id": page_id,
            "doc_id": doc_id,
            "page_index": page_index,
            "text_layer": text_layer,
            "text_layer_chars": len(text_layer),
            "section_hint": section_hint,
            "ocr_engine": None,
            "ocr_text": "",
            "ocr_error": None,
        }
        if config.run_ocr:
            ocr_record.update(_ocr_page_text(image_path))
        ocr_records.append(ocr_record)

    return page_records, ocr_records


def _build_document_record(pdf_path: Path, raw_sha256: str, page_count: int) -> Dict[str, Any]:
    policy_family_id, version_id = _infer_family_and_version(pdf_path.stem)
    doc_id = f"{_slugify(pdf_path.stem)}_{raw_sha256[:8]}"
    return {
        "doc_id": doc_id,
        "source_name": pdf_path.name,
        "source_path": str(pdf_path),
        "source_license": "unknown",
        "raw_sha256": raw_sha256,
        "policy_family_id": policy_family_id,
        "version_id": version_id,
        "page_count": page_count,
        "is_internal": False,
    }


def _create_document_splits(
    documents: List[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    docs = sorted(documents, key=lambda item: item["doc_id"])
    rng.shuffle(docs)

    total = len(docs)
    if total == 1:
        return [
            {
                "doc_id": docs[0]["doc_id"],
                "policy_family_id": docs[0]["policy_family_id"],
                "version_id": docs[0]["version_id"],
                "split": "train",
            }
        ]

    train_cut = int(total * train_ratio)
    valid_cut = train_cut + int(total * valid_ratio)
    splits: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if idx < train_cut:
            split = "train"
        elif idx < valid_cut:
            split = "valid"
        else:
            split = "test"
        splits.append(
            {
                "doc_id": doc["doc_id"],
                "policy_family_id": doc["policy_family_id"],
                "version_id": doc["version_id"],
                "split": split,
            }
        )
    return splits


def _build_page_manifest(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "page_id": page["page_id"],
            "doc_id": page["doc_id"],
            "image_path": page["image_path"],
            "page_index": page["page_index"],
            "page_number": page["page_number"],
            "source": page.get("source"),
            "section_hint": page.get("section_hint"),
            "version_id": page["version_id"],
            "source_type": page["source_type"],
            "sha256": page["sha256"],
        }
        for page in pages
    ]


def _build_weak_retrieval_pairs(
    pages: List[Dict[str, Any]],
    ocr_aux: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pages_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        pages_by_doc.setdefault(page["doc_id"], []).append(page)

    ocr_by_page = {item["page_id"]: item for item in ocr_aux}
    pairs: List[Dict[str, Any]] = []
    for page in pages:
        section_hint = page.get("section_hint")
        if not section_hint:
            continue

        same_doc_pages = [
            item["page_id"]
            for item in pages_by_doc.get(page["doc_id"], [])
            if item["page_id"] != page["page_id"]
        ]
        cross_doc_pages = [
            item["page_id"]
            for item in pages
            if item["doc_id"] != page["doc_id"]
        ]
        hard_negatives = (same_doc_pages[:2] + cross_doc_pages[:2])[:4]
        aux = ocr_by_page.get(page["page_id"], {})
        evidence_text = aux.get("text_layer", "")[:500] or section_hint
        pairs.append(
            {
                "query": f"Find the policy page that discusses {section_hint}",
                "positive_page_id": page["page_id"],
                "positive_image": page["image_path"],
                "hard_negative_page_ids": hard_negatives,
                "answer": section_hint,
                "citation": {
                    "doc_id": page["doc_id"],
                    "page_index": page["page_index"],
                    "page_number": page["page_number"],
                    "evidence_text": evidence_text,
                },
                "pair_source": "weak_section_hint",
            }
        )
    return pairs


def preprocess_page_images(config: PageImagePreprocessConfig) -> PageImagePreprocessResult:
    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found under {input_dir}")

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.index_dir.mkdir(parents=True, exist_ok=True)
    (config.index_dir / "page_embeddings").mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)

    documents: List[Dict[str, Any]] = []
    pages: List[Dict[str, Any]] = []
    ocr_aux: List[Dict[str, Any]] = []
    run_manifest: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        raw_sha256 = compute_sha256(pdf_path)
        try:
            import fitz

            page_count = len(fitz.open(pdf_path))
        except ImportError as exc:
            raise ImportError("PyMuPDF is required for preprocessing.") from exc

        document_record = _build_document_record(pdf_path, raw_sha256, page_count)
        doc_pages, doc_ocr = _render_pdf_to_page_records(
            pdf_path=pdf_path,
            config=config,
            doc_id=document_record["doc_id"],
            document_record=document_record,
        )
        documents.append(document_record)
        pages.extend(doc_pages)
        ocr_aux.extend(doc_ocr)
        run_manifest.append(
            {
                "doc_id": document_record["doc_id"],
                "source_path": str(pdf_path),
                "raw_sha256": raw_sha256,
                "page_count": page_count,
                "render_dpi": config.render_dpi,
                "run_ocr": config.run_ocr,
            }
        )

    splits = _create_document_splits(
        documents,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )
    page_manifest = _build_page_manifest(pages)
    retrieval_pairs = _build_weak_retrieval_pairs(pages, ocr_aux)

    documents_path = _write_table(documents, config.processed_dir / "documents.parquet")
    pages_path = _write_table(pages, config.processed_dir / "pages.parquet")
    splits_path = _write_table(splits, config.processed_dir / "splits.parquet")
    ocr_aux_path = config.processed_dir / "ocr_aux.jsonl"
    retrieval_pairs_path = config.processed_dir / "colqwen2_retrieval_pairs.jsonl"
    page_manifest_path = config.index_dir / "page_manifest.jsonl"

    _write_jsonl(ocr_aux, ocr_aux_path)
    _write_jsonl(retrieval_pairs, retrieval_pairs_path)
    _write_jsonl(page_manifest, page_manifest_path)
    _write_jsonl(run_manifest, config.manifest_dir / "preprocess_run_manifest.jsonl")

    return PageImagePreprocessResult(
        documents_path=documents_path,
        pages_path=pages_path,
        ocr_aux_path=ocr_aux_path,
        page_manifest_path=page_manifest_path,
        retrieval_pairs_path=retrieval_pairs_path,
        splits_path=splits_path,
        document_count=len(documents),
        page_count=len(pages),
    )


def clean_preprocess_outputs(output_root: Path) -> None:
    for relative_path in [
        "01_interim/page_images",
        "02_processed",
        "03_index/colqwen2",
        "manifests/preprocess_run_manifest.jsonl",
    ]:
        path = Path(output_root) / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
