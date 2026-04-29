from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PdfPageLayout:
    page_number: int
    blocks: List[Dict[str, Any]]


def extract_text_by_page(pdf_path: Path) -> List[str]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. Install it with `pip install PyMuPDF`."
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    page_texts: List[str] = []
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        page_texts.append(page.get_text("text").strip())
    return page_texts


def extract_layout_by_page(pdf_path: Path) -> List[PdfPageLayout]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF layout extraction. Install it with `pip install PyMuPDF`."
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    layouts: List[PdfPageLayout] = []
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        blocks = [
            {
                "bbox": block[:4],
                "type": block[4],
                "text": block[5].strip(),
            }
            for block in page.get_text("blocks")
            if block[5].strip()
        ]
        layouts.append(PdfPageLayout(page_number=page_number + 1, blocks=blocks))
    return layouts


def render_pdf_pages(pdf_path: Path, dpi: int = 150) -> List[bytes]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF rendering. Install it with `pip install PyMuPDF`."
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    images: List[bytes] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    return images
