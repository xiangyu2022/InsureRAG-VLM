import json

import pytest

from src.insurerag_vlm.holdout_source_state import source_state


def setup_records(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (tmp_path / "document_holds.json").write_text(json.dumps({"documents": {"c": {"status": "hold"}}}))
    (tmp_path / "document_families.json").write_text(json.dumps({"groups": [
        {"document_ids": ["a", "b"]}, {"document_ids": ["b", "c"]}]}))
    (local / "provisional_item_pool_01.jsonl").write_text(json.dumps(
        {"document_group": "c", "split": "test", "question": "private question", "answer": "private answer"}) + "\n")
    return local


def test_review_lookup_follows_family_and_redacts_private_content(tmp_path):
    local = setup_records(tmp_path)
    for n in (9, 10):
        (local / f"stage{n}_source_selection.json").write_text(json.dumps(
            {"status": "NO_DRAFTS", "document_id": "a", "reason": "private source text"}))
    (local / "stage11_targeted_review_packet.json").write_text("not parsed")
    result = source_state(tmp_path, ["a", "a"], 1)
    assert len(result["documents"]) == 1
    item = result["documents"][0]
    assert item["family_document_ids"] == ["a", "b", "c"]
    assert item["held_family_document_ids"] == ["c"]
    assert item["family_item_counts"] == {"test": 1, "dev": 0}
    assert [r["stage"] for r in item["prior_review_records"]] == [10, 9]
    assert "private" not in json.dumps(result)
    assert result["status"] == "LOOKUP_ONLY_NOT_CLEARANCE"


def test_absent_or_partial_id_is_not_treated_as_clearance(tmp_path):
    local = setup_records(tmp_path)
    (local / "stage1_source_decisions.json").write_text(json.dumps({"document_id": "abc"}))
    result = source_state(tmp_path, ["a"], 1)
    assert result["documents"][0]["prior_review_records"] == []
    assert result["documents"][0]["review_lookup"] == "no_tracked_review_found"
    assert "Missing records do not establish clearance" in result["warning"]


def test_malformed_in_scope_review_fails_instead_of_silent_clearance(tmp_path):
    local = setup_records(tmp_path)
    (local / "stage1_source_decisions.json").write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        source_state(tmp_path, ["a"], 1)
