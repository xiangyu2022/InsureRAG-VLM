import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .pdf import extract_text_by_page

SUPPORTED_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

@dataclass
class PageDocument:
    doc_id: str
    text: str
    metadata: Dict[str, str]
    image_path: Optional[Path] = None
    page_number: Optional[int] = None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


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


def load_pdf_documents(path: Path, render_images: bool = False, render_dir: Optional[Path] = None) -> List[PageDocument]:
    texts = extract_text_by_page(path)
    rendered_paths = []
    if render_images and render_dir is not None:
        rendered_paths = _render_pdf_pages(path, render_dir)

    pages: List[PageDocument] = []
    for page_number, page_text in enumerate(texts, start=1):
        source = f"{path.name}#page={page_number}"
        image_path = rendered_paths[page_number - 1] if rendered_paths else None
        pages.append(
            PageDocument(
                doc_id=source,
                text=page_text,
                image_path=image_path,
                page_number=page_number,
                metadata={"source": source, "page": str(page_number), "path": str(path.name)},
            )
        )
    return pages


def load_documents(data_folder: Path, render_pdf_pages: bool = False, pdf_render_dir: Optional[Path] = None) -> List[PageDocument]:
    data_folder = Path(data_folder)
    documents: List[PageDocument] = []

    for path in sorted(data_folder.rglob("*")):
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = _read_text_file(path)
            if not text:
                continue
            documents.append(
                PageDocument(
                    doc_id=str(path.relative_to(data_folder)),
                    text=text,
                    metadata={"source": str(path.relative_to(data_folder)), "path": str(path.relative_to(data_folder))},
                )
            )
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            documents.append(
                PageDocument(
                    doc_id=str(path.relative_to(data_folder)),
                    text="",
                    image_path=path,
                    metadata={"source": str(path.relative_to(data_folder)), "path": str(path.relative_to(data_folder))},
                )
            )
        elif suffix in SUPPORTED_PDF_EXTENSIONS:
            documents.extend(load_pdf_documents(path, render_images=render_pdf_pages, render_dir=pdf_render_dir or data_folder))

    return documents


def save_metadata(metadata: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def load_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))
