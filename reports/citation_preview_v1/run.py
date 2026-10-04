"""Actual local HTTP regression; synthetic PDFs and controlled answer fixture.

Usage from the repository root:
  python reports/citation_preview_v1/run.py --output-dir reports/citation_preview_v1/retake
Refuses to overwrite an existing result directory. No answer model service runs.
Only answer generation is stubbed to isolate citation binding; upload, indexing,
HTTP dispatch, preview registration, byte validation, and rendering are real.
"""
import argparse
import ast
import hashlib
import json
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.insurerag_vlm.app as app
from src.insurerag_vlm.config import ModelConfig


def sha(content):
    return hashlib.sha256(content).hexdigest()


def pdf(amount):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 100), f"POLICY DECLARATIONS\nPolicy number: ZX-77\nCollision deductible: ${amount}.", fontsize=18)
        return document.tobytes()


def request(base, path, content=None):
    headers = {}
    if content is not None:
        boundary = "citation-preview-regression"
        content = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="policy.pdf"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
                   + content + f"\r\n--{boundary}--\r\n".encode())
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(base + path, data=content, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(ROOT / "reports/research_v1/frozen_source_v1.zip") as archive:
        old_source = archive.read("src/insurerag_vlm/app.py")
    old_tree = ast.parse(old_source)
    old_class = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == "DemoHandler")
    methods = [node for node in old_class.body if isinstance(node, ast.FunctionDef) and node.name in {"_resolve_source_pdf", "_render_source_page_image"}]
    assert len(methods) == 2
    baseline_class = ast.ClassDef(name="BaselinePreview", bases=[], keywords=[], body=methods, decorator_list=[])
    baseline_tree = ast.fix_missing_locations(ast.Module(body=[baseline_class], type_ignores=[]))
    namespace = dict(vars(app))
    exec(compile(baseline_tree, "frozen_app_preview_methods", "exec"), namespace)
    BaselinePreview = namespace["BaselinePreview"]

    class Handler(app.DemoHandler):
        _pipeline = None
        _data_folder = app.DATA_FOLDER
        _preview_bindings = {}

    def config(**kwargs):
        return ModelConfig(**kwargs, vlm_model="local-extractive", retrieval_model="local-hashing", enable_image_signal=False)

    def fixed_answer(question, pipeline, data_folder):
        with fitz.open(data_folder / "policy.pdf") as document:
            amount = re.search(r"Collision deductible: (\$[0-9]+)", document[0].get_text()).group(1)
        return {"source": "document", "answer": f"The collision deductible is {amount}.",
                "abstain": False, "generation_used": False, "backend": "controlled answer fixture (no model)",
                "citations": [{"source": "policy.pdf#page=1"}]}

    # Retained generated upload files are kept under the application's approved
    # snapshot root, independently of the compact report output directory.
    uploads = ROOT / "reports/demo_uploads/citation-preview-regression" / args.output_dir.name / "current"
    report = {"diagnostic": "synthetic same-filename PDF replacement via actual HTTP; controlled answer fixture, no model inference",
              "frozen_app_sha256": sha(old_source), "current_app_sha256": sha(Path(app.__file__).read_bytes()),
              "question": "What is the collision deductible in the uploaded policy?"}
    with patch.object(app, "UPLOAD_DIR", uploads), patch.object(app, "ModelConfig", side_effect=config), patch.object(app, "build_chat_response", side_effect=fixed_answer):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            snapshots = []
            for amount in (735, 1820):
                content = pdf(amount)
                status, body = request(base, "/api/upload", content)
                assert status == 200, body
                upload = json.loads(body)
                status, body = request(base, "/api/chat?q=" + urllib.parse.quote(report["question"]))
                answer = json.loads(body)
                assert status == 200 and not answer["abstain"], answer
                citation = answer["citations"][0]
                assert citation["preview_sha256"] == sha(content)
                status, image = request(base, citation["preview_url"])
                assert status == 200
                BaselinePreview._data_folder = Handler._data_folder
                baseline_image = BaselinePreview._render_source_page_image("policy.pdf#page=1").read_bytes()
                snapshots.append({"content": content, "folder": Handler._data_folder,
                                  "url": citation["preview_url"], "image": image, "baseline_image": baseline_image})
                report[str(amount)] = {"upload_status": upload["ok"], "answer": answer["answer"],
                    "generation_used": answer["generation_used"], "answer_backend": answer["backend"],
                    "citation_source": citation["source"], "pdf_sha256": citation["preview_sha256"],
                    "preview_sha256": sha(image), "preview_pixels_sha256": sha(fitz.Pixmap(image).samples)}
            status, old_image_after = request(base, snapshots[0]["url"])
            old_unbound_after = BaselinePreview._render_source_page_image("policy.pdf#page=1").read_bytes()
            assert status == 200 and old_image_after == snapshots[0]["image"]
            assert old_unbound_after == snapshots[1]["baseline_image"]
            assert old_unbound_after != snapshots[0]["baseline_image"]
            delayed = {"citations": [{"source": "policy.pdf#page=1"}]}
            Handler._bind_citation_previews(delayed, snapshots[0]["folder"])
            delayed_status, delayed_image = request(base, delayed["citations"][0]["preview_url"])
            assert delayed_status == 200 and delayed_image == snapshots[0]["image"]
            unbound_status, _ = request(base, "/api/page-image?source=policy.pdf%23page%3D1")
            token = urllib.parse.parse_qs(urllib.parse.urlparse(snapshots[0]["url"]).query)["token"][0]
            Handler._preview_bindings[token]["sha256"] = "0" * 64
            wrong_hash_status, _ = request(base, snapshots[0]["url"])
            report["checks"] = {
                "before_unbound_old_citation_changes_to_new_pixels": True,
                "after_bound_old_citation_preserves_old_pixels": True,
                "delayed_registration_uses_captured_old_folder": True,
                "old_and_new_preview_pixels_differ": fitz.Pixmap(snapshots[0]["image"]).samples != fitz.Pixmap(snapshots[1]["image"]).samples,
                "unbound_endpoint_status": unbound_status,
                "wrong_hash_even_with_cached_preview_status": wrong_hash_status,
            }
            assert unbound_status == 400 and wrong_hash_status == 404
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
