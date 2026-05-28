import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .pdf import extract_text_by_page

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv", ".jsonl", ".md"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
MAX_TEXT_DOCUMENT_CHARS = 12000
PACKET_MANIFEST_FILENAMES = (
    "packet_manifest.json",
    "manifests/packet_manifest.json",
)

@dataclass
class PageDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    image_path: Optional[Path] = None
    page_number: Optional[int] = None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _relative_key(path: Path, data_folder: Path) -> str:
    return path.relative_to(data_folder).as_posix()


def _has_metadata_value(value: Any) -> bool:
    return value is not None and value != ""


def _manifest_defaults(payload: Dict[str, Any], excluded_keys: set[str]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in excluded_keys and _has_metadata_value(value)
    }


def _load_packet_manifest(data_folder: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = next((data_folder / filename for filename in PACKET_MANIFEST_FILENAMES if (data_folder / filename).exists()), None)
    if manifest_path is None:
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif isinstance(payload, dict):
        root_defaults = _manifest_defaults(payload, {"documents", "packets"})
        for entry in payload.get("documents", []) or []:
            if isinstance(entry, dict):
                entries.append({**root_defaults, **entry})
        for packet in payload.get("packets", []) or []:
            if not isinstance(packet, dict):
                continue
            packet_defaults = {**root_defaults, **_manifest_defaults(packet, {"documents"})}
            for entry in packet.get("documents", []) or []:
                if isinstance(entry, dict):
                    entries.append({**packet_defaults, **entry})
    manifest: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        raw_path = str(entry.get("path") or entry.get("file") or entry.get("relative_path") or "").strip()
        if not raw_path:
            continue
        normalized_path = Path(raw_path).as_posix()
        metadata = {
            key: value
            for key, value in entry.items()
            if key not in {"path", "file", "relative_path"} and _has_metadata_value(value)
        }
        metadata.setdefault("source_origin", "local_real_policy_packet")
        metadata.setdefault("source_name", "Local real policy packet")
        manifest[normalized_path] = metadata
    return manifest


def _split_text_documents(
    path: Path,
    data_folder: Path,
    text: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[PageDocument]:
    relative_path = _relative_key(path, data_folder)
    extra_metadata = dict(extra_metadata or {})
    if len(text) <= MAX_TEXT_DOCUMENT_CHARS:
        return [
            PageDocument(
                doc_id=relative_path,
                text=text,
                metadata={"source": relative_path, "path": relative_path, **extra_metadata},
            )
        ]

    documents: List[PageDocument] = []
    for chunk_number, start in enumerate(range(0, len(text), MAX_TEXT_DOCUMENT_CHARS), start=1):
        chunk = text[start : start + MAX_TEXT_DOCUMENT_CHARS].strip()
        if not chunk:
            continue
        source = f"{relative_path}#chunk={chunk_number}"
        documents.append(
            PageDocument(
                doc_id=source,
                text=chunk,
                metadata={
                    "source": source,
                    "path": relative_path,
                    "chunk": str(chunk_number),
                    **extra_metadata,
                },
            )
        )
    return documents


def _render_pdf_pages(path: Path, output_dir: Path, dpi: int = 150) -> List[Path]:
    from .pdf import render_pdf_pages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = render_pdf_pages(path, dpi=dpi)
    rendered_paths: List[Path] = []
    for page_number, image_bytes in enumerate(images, start=1):
        output_path = output_dir / f"{path.stem}_page_{page_number:03}.png"
        output_path.write_bytes(image_bytes)
        rendered_paths.append(output_path)
    return rendered_paths


def load_pdf_documents(
    path: Path,
    render_images: bool = False,
    render_dir: Optional[Path] = None,
    relative_path: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[PageDocument]:
    texts = extract_text_by_page(path)
    rendered_paths = []
    if render_images and render_dir is not None:
        rendered_paths = _render_pdf_pages(path, render_dir)
    relative_path = relative_path or path.name
    extra_metadata = dict(extra_metadata or {})

    pages: List[PageDocument] = []
    for page_number, page_text in enumerate(texts, start=1):
        source = f"{relative_path}#page={page_number}"
        image_path = rendered_paths[page_number - 1] if rendered_paths else None
        pages.append(
            PageDocument(
                doc_id=source,
                text=page_text,
                image_path=image_path,
                page_number=page_number,
                metadata={
                    "source": source,
                    "page": str(page_number),
                    "path": relative_path,
                    **extra_metadata,
                },
            )
        )
    return pages


def load_documents(data_folder: Path, render_pdf_pages: bool = False, pdf_render_dir: Optional[Path] = None) -> List[PageDocument]:
    data_folder = Path(data_folder)
    documents: List[PageDocument] = []
    packet_manifest = _load_packet_manifest(data_folder)

    for path in sorted(data_folder.rglob("*")):
        suffix = path.suffix.lower()
        relative_key = _relative_key(path, data_folder)
        extra_metadata = packet_manifest.get(relative_key, {})
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = _read_text_file(path)
            if not text:
                continue
            documents.extend(_split_text_documents(path, data_folder, text, extra_metadata=extra_metadata))
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            documents.append(
                PageDocument(
                    doc_id=relative_key,
                    text="",
                    image_path=path,
                    metadata={"source": relative_key, "path": relative_key, **extra_metadata},
                )
            )
        elif suffix in SUPPORTED_PDF_EXTENSIONS:
            documents.extend(
                load_pdf_documents(
                    path,
                    render_images=render_pdf_pages,
                    render_dir=pdf_render_dir or data_folder,
                    relative_path=relative_key,
                    extra_metadata=extra_metadata,
                )
            )

    return documents


def save_metadata(metadata: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def load_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))
