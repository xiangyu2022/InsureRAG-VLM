from pathlib import Path

from PIL import Image


def extract_text_from_image(image_path: Path) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required for OCR. Install it with `pip install pytesseract` "
            "and ensure Tesseract is installed on your system."
        ) from exc

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        return pytesseract.image_to_string(image).strip()
