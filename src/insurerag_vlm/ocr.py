from pathlib import Path

from PIL import Image


class OCRUnavailableError(RuntimeError):
    """A page needs OCR but the optional local OCR backend cannot run."""


def is_blank_image(image_path: Path) -> bool:
    """Recognize only near-white empty renders; uncertain pages still need OCR."""
    with Image.open(image_path) as image:
        minimum, _ = image.convert("L").getextrema()
        return minimum >= 250


def extract_text_from_image(image_path: Path) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailableError(
            "This page has no selectable text. Local OCR requires pytesseract and the Tesseract executable."
        ) from exc

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image).strip()
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as exc:
        raise OCRUnavailableError(
            "This page has no selectable text and local OCR could not run. Install/configure Tesseract or upload a searchable PDF. "
            "The separate image-QA API was not invoked."
        ) from exc
