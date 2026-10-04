import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import fitz
import pytest

import src.insurerag_vlm.app as app
from src.insurerag_vlm.config import ModelConfig


def make_pdf(amount):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 100), f"Collision deductible: ${amount}.", fontsize=18)
        return document.tobytes()


@pytest.fixture
def handler(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads" / "current")
    monkeypatch.setattr(app, "PAGE_CACHE_DIR", tmp_path / "preview-cache")
    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path)

    class Handler(app.DemoHandler):
        _pipeline = None
        _data_folder = tmp_path / "data" / "public"
        _index_dir = tmp_path / "index"
        _preview_bindings = {}

    def config(**kwargs):
        return ModelConfig(**kwargs, vlm_model="local-extractive", retrieval_model="local-hashing", enable_image_signal=False)

    monkeypatch.setattr(app, "ModelConfig", config)
    return Handler


def bind(handler, folder):
    result = {"citations": [{"source": "policy.pdf#page=1"}]}
    handler._bind_citation_previews(result, folder)
    citation = result["citations"][0]
    token = parse_qs(urlparse(citation["preview_url"]).query)["token"][0]
    return citation, token


def pixels_from_pdf(content):
    with fitz.open(stream=content, filetype="pdf") as document:
        return document[0].get_pixmap(dpi=92, annots=False).samples


def get(handler_type, url):
    request = object.__new__(handler_type)
    request.path = url
    request._send = Mock()
    request._handle_get()
    return request._send.call_args.args


def test_same_filename_replacement_preserves_old_citation_pixels(handler):
    old_bytes, new_bytes = make_pdf(735), make_pdf(1820)
    handler.activate_upload("policy.pdf", old_bytes)
    old_folder = handler._data_folder
    old_citation, old_token = bind(handler, old_folder)
    old_preview = handler._render_bound_page_image(old_token)
    assert fitz.Pixmap(str(old_preview)).samples == pixels_from_pdf(old_bytes)

    handler.activate_upload("policy.pdf", new_bytes)
    new_citation, new_token = bind(handler, handler._data_folder)
    assert old_folder != handler._data_folder and old_folder.exists()
    assert old_citation["preview_sha256"] == hashlib.sha256(old_bytes).hexdigest()
    assert new_citation["preview_sha256"] == hashlib.sha256(new_bytes).hexdigest()
    assert old_citation["preview_sha256"] != new_citation["preview_sha256"]
    # A cached old citation and a newly requested old citation both stay old.
    assert handler._render_bound_page_image(old_token) == old_preview
    assert fitz.Pixmap(str(handler._render_bound_page_image(old_token))).samples == pixels_from_pdf(old_bytes)
    assert fitz.Pixmap(str(handler._render_bound_page_image(new_token))).samples == pixels_from_pdf(new_bytes)
    delayed_citation, delayed_token = bind(handler, old_folder)
    assert delayed_citation["preview_sha256"] == old_citation["preview_sha256"]
    assert handler._render_bound_page_image(delayed_token) == old_preview


def test_chat_binds_captured_folder_when_upload_finishes_during_generation(handler):
    old_bytes, new_bytes = make_pdf(735), make_pdf(1820)
    handler.activate_upload("policy.pdf", old_bytes)
    captured_folder = handler._data_folder

    def answer(question, pipeline, data_folder):
        assert data_folder == captured_folder
        handler.activate_upload("policy.pdf", new_bytes)
        return {"source": "document", "answer": "$735", "citations": [{"source": "policy.pdf#page=1"}]}

    with patch.object(app, "build_chat_response", side_effect=answer):
        status, payload, _ = get(handler, "/api/chat?q=policy+deductible")
    assert status == 200
    citation = json.loads(payload)["citations"][0]
    assert citation["preview_sha256"] == hashlib.sha256(old_bytes).hexdigest()
    status, image_bytes, image_type = get(handler, citation["preview_url"])
    assert (status, image_type) == (200, "image/png")
    assert fitz.Pixmap(image_bytes).samples == pixels_from_pdf(old_bytes)


def test_changed_bytes_and_wrong_hash_are_rejected_even_with_cached_preview(handler):
    handler.activate_upload("policy.pdf", make_pdf(735))
    citation, token = bind(handler, handler._data_folder)
    assert handler._render_bound_page_image(token).exists()
    handler._preview_bindings[token]["sha256"] = "0" * 64
    assert handler._render_bound_page_image(token) is None
    handler._preview_bindings[token]["sha256"] = citation["preview_sha256"]
    (handler._data_folder / "policy.pdf").write_bytes(make_pdf(1820))
    assert handler._render_bound_page_image(token) is None
    assert get(handler, citation["preview_url"])[0] == 404


def test_cross_snapshot_and_cross_root_paths_cannot_be_registered_or_rendered(handler, tmp_path):
    handler.activate_upload("policy.pdf", make_pdf(735))
    active = handler._data_folder
    citation, token = bind(handler, active)
    outside = tmp_path / "private"
    outside.mkdir()
    outsider = outside / "policy.pdf"
    outsider.write_bytes(make_pdf(9999))
    handler.activate_upload("policy.pdf", make_pdf(1820))
    for source in [str(outsider), str(active / "policy.pdf"), "../../../../private/policy.pdf"]:
        result = {"citations": [{"source": source + "#page=1"}]}
        handler._bind_citation_previews(result, handler._data_folder)
        assert result["citations"][0]["preview_url"] is None
    # Defense in depth if a registry entry is malformed: a matching content
    # hash does not grant access outside the recorded snapshot or allowed roots.
    handler._preview_bindings[token].update(path=str(outsider), sha256=hashlib.sha256(outsider.read_bytes()).hexdigest())
    assert handler._render_bound_page_image(token) is None
    handler._preview_bindings[token]["root"] = str(outside)
    assert handler._render_bound_page_image(token) is None


def test_unbound_unknown_and_malformed_preview_requests_fail_closed(handler):
    handler.activate_upload("policy.pdf", make_pdf(735))
    assert get(handler, "/api/page-image?source=policy.pdf%23page%3D1")[0] == 400
    assert get(handler, "/api/page-image?token=" + "0" * 48)[0] == 404
    assert get(handler, "/api/page-image?token=../../policy.pdf")[0] == 404
    for source in ["policy.pdf#page=0", "policy.pdf#page=1junk", "policy.pdf#page=-1"]:
        result = {"citations": [{"source": source}]}
        handler._bind_citation_previews(result, handler._data_folder)
        assert result["citations"][0]["preview_url"] is None


def test_preview_registry_isolated_between_handler_classes(handler):
    handler.activate_upload("policy.pdf", make_pdf(735))
    citation, token = bind(handler, handler._data_folder)

    class OtherHandler(handler):
        pass

    assert OtherHandler._render_bound_page_image(token) is None
    assert get(handler, citation["preview_url"])[0] == 200
