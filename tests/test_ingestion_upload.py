import io
from pathlib import Path
from unittest.mock import Mock, patch

import fitz
import pytest

import src.insurerag_vlm.app as app
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline, DocumentTextUnavailableError
from src.insurerag_vlm.ocr import OCRUnavailableError


def pdf_bytes(*page_types):
    document = fitz.open()
    for page_type in page_types:
        page = document.new_page()
        if page_type == "text":
            page.insert_text((72, 100), "Collision deductible: $735.", fontsize=18)
        elif page_type == "scan":
            image_doc = fitz.open()
            image_page = image_doc.new_page()
            image_page.insert_text((72, 100), "Collision deductible: $735.", fontsize=18)
            page.insert_image(page.rect, stream=image_page.get_pixmap().tobytes("png"))
            image_doc.close()
    value = document.tobytes()
    document.close()
    return value


def pipeline(tmp_path, images=True):
    return DocumentRetrievalPipeline(ModelConfig(
        vlm_model="local-extractive", retrieval_model="local-hashing", corpus_source="documents",
        enable_image_signal=images, index_dir=tmp_path / "index", pdf_render_dir=tmp_path / "rendered",
    ))


def test_blank_page_does_not_require_ocr_or_fail_readable_pdf(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.pdf").write_bytes(pdf_bytes("text", "blank"))
    instance = pipeline(tmp_path)
    with patch("src.insurerag_vlm.hybrid_pipeline.extract_text_from_image") as ocr:
        instance.build_index(docs)
    ocr.assert_not_called()
    assert instance.ingestion_report["readable_pages"] == 1
    assert instance.ingestion_report["blank_pages"] == 1
    pages = instance._ensure_indices(docs)["page_meta"]
    assert len(pages) == 1 and pages[0]["source"] == "policy.pdf#page=1"


@pytest.mark.parametrize("images", [False, True])
def test_all_scanned_pdf_cannot_claim_successful_text_ingestion(tmp_path, images):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(pdf_bytes("scan"))
    instance = pipeline(tmp_path, images)
    with patch("src.insurerag_vlm.hybrid_pipeline.extract_text_from_image", side_effect=OCRUnavailableError("OCR requires Tesseract; upload a searchable PDF.")):
        with pytest.raises(DocumentTextUnavailableError, match="No readable text"):
            instance.build_index(docs)
    assert not instance._index_paths()["embedding_manifest"].exists()
    assert instance.ingestion_report["unreadable_pages"] == 1


def test_readable_pages_survive_missing_ocr_with_explicit_partial_ingestion_status(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "mixed.pdf").write_bytes(pdf_bytes("text", "scan"))
    instance = pipeline(tmp_path)
    with patch("src.insurerag_vlm.hybrid_pipeline.extract_text_from_image", side_effect=OCRUnavailableError("OCR requires Tesseract.")):
        instance.build_index(docs)
    report = instance.ingestion_report
    assert (report["total_pages"], report["readable_pages"], report["unreadable_pages"]) == (2, 1, 1)
    assert report["pages"][1]["status"] == "ocr_unavailable"
    assert report["answer_input"] == "text"
    assert instance._ensure_indices(docs)["embedding_manifest"]["ingestion_report"] == report


def post_upload(handler_type, filename, content):
    handler = object.__new__(handler_type)
    handler.path = "/api/upload"
    boundary = "test-upload-boundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n\r\n'.encode()
            + content + f"\r\n--{boundary}--\r\n".encode())
    handler.headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._send_json = Mock()
    handler.do_POST()
    return handler._send_json.call_args.args[0], handler._send_json.call_args.kwargs.get("status", 200)


def test_failed_upload_is_not_activated_or_left_to_poison_next_upload(tmp_path):
    class Handler(app.DemoHandler):
        _pipeline = None
        _data_folder = tmp_path / "public-originals"
        _index_dir = tmp_path / "original-index"
    def config(**kwargs):
        return ModelConfig(**kwargs, vlm_model="local-extractive", retrieval_model="local-hashing", enable_image_signal=False)
    with patch.object(app, "UPLOAD_DIR", tmp_path / "uploads" / "current"), patch.object(app, "ModelConfig", side_effect=config):
        bad, status = post_upload(Handler, "scan.pdf", pdf_bytes("scan"))
        assert status == 422 and not bad["ok"] and bad["active_documents_unchanged"]
        assert Handler._pipeline is None
        assert not list((tmp_path / "uploads").rglob("scan.pdf"))
        good, status = post_upload(Handler, "policy.pdf", pdf_bytes("text"))
        assert status == 200 and good["ok"]
        assert good["corpus"]["ingestion"]["readable_pages"] == 1
        original_folder, original_pipeline = Handler._data_folder, Handler._pipeline
        original_bytes = (original_folder / "policy.pdf").read_bytes()
        bad, status = post_upload(Handler, "additional-scan.pdf", pdf_bytes("scan"))
        assert status == 422 and Handler._data_folder == original_folder
        assert not list((tmp_path / "uploads").rglob("additional-scan.pdf"))
        # Replacing an active file with an unreadable PDF must also roll back.
        bad, status = post_upload(Handler, "policy.pdf", pdf_bytes("scan"))
        assert status == 422
        assert Handler._data_folder == original_folder and Handler._pipeline is original_pipeline
        assert (original_folder / "policy.pdf").read_bytes() == original_bytes
        assert len(list((tmp_path / "uploads" / "sessions").iterdir())) == 1
        # The active index's recorded paths remain valid after activation.
        assert original_pipeline._ensure_indices(original_folder)["page_meta"]
