"""Synthetic adapter tests, separate from the actual local DOL smoke run."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.insurerag_vlm import clause_snapshot as snapshot
from src.insurerag_vlm.chroma_knowledge import ClauseStore


def fixture_snapshot(tmp_path, monkeypatch, change=None):
    rows = []
    for i in range(56, 72):
        text = "January 1, 2020\n\nFictional fixture statement about reporting. " * 40
        rows.append({"id": f"dol_html_aca-part-{i}", "text": text,
                     "text_sha256": sha256(text.encode()).hexdigest(), "html_sha256": "a" * 64,
                     "source_url": snapshot.URL_PREFIX + str(i), "rights": snapshot.RIGHTS,
                     "retrieved_utc": "2026-10-01T00:00:00Z", "answer": "MUST NOT IMPORT", "split": "test"})
    if change:
        change(rows)
    blob = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    expected = sha256(blob).hexdigest()
    monkeypatch.setattr(snapshot, "SNAPSHOT_SHA256", expected)
    (tmp_path / "records.jsonl").write_bytes(blob)
    (tmp_path / "manifest.json").write_text(json.dumps({"records": 16, "records_sha256": expected}), encoding="utf-8")
    return rows


def test_snapshot_offsets_and_no_annotation_import(tmp_path, monkeypatch):
    sources = fixture_snapshot(tmp_path, monkeypatch)
    records, receipt = snapshot.prepare_snapshot(tmp_path)
    originals = {r["id"]: r["text"] for r in sources}
    assert receipt["documents"] == 16 and receipt["qa_labels_imported"] is False
    assert len(records) > 16
    for row in records:
        assert row["text"] == originals[row["doc_id"]][row["char_start"]:row["char_end"]]
        assert len(row["text"]) <= 1600
        assert "answer" not in row and "split" not in row
        assert row["page_number"] == 0 and row["source_header_date"] == "January 1, 2020"
    assert snapshot.prepare_snapshot(tmp_path)[0] == records


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(rights="unknown"),
    lambda rows: rows[0].update(source_url="https://example.org/unapproved"),
    lambda rows: rows[0].update(text_sha256="0" * 64),
    lambda rows: rows.pop(),
])
def test_snapshot_rejects_changed_scope(tmp_path, monkeypatch, change):
    fixture_snapshot(tmp_path, monkeypatch, change)
    with pytest.raises(ValueError):
        snapshot.prepare_snapshot(tmp_path)


def test_snapshot_bytes_must_match_manifest(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path, monkeypatch)
    path = tmp_path / "records.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        snapshot.prepare_snapshot(tmp_path)


def test_project_knowledge_search_uses_chroma(tmp_path):
    db = tmp_path / "db"
    with ClauseStore(db, writable=True, collection="synthetic_cli_test") as store:
        store.upsert([{"record_id": "fixture", "doc_id": "fixture", "page_number": 0,
                       "citation": "Fictional plumbing fixture", "source_sha256": "a" * 64,
                       "text": "Fictional prescription drug reporting fixture."}])
    result = subprocess.run([sys.executable, "main.py", "knowledge-search", "prescription drug reporting",
                             "--store", str(db), "--collection", "synthetic_cli_test", "--top-k", "1"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["results"][0]["evidence"]["doc_id"] == "fixture"
    assert output["embedding"]["embedding_dim"] == 512
