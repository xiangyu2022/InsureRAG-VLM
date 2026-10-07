"""Read-only source review index. Mentions and item counts are not clearance."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REVIEW_NAMES = re.compile(
    r"stage(\d+)[a-z]?_(?:source_decisions|source_selection|source_selection_hold|extra_source_proof)\.json$"
)


def source_state(root: Path, document_ids: list[str], pool_number: int) -> dict:
    """Find prior review pointers without returning source bodies or item-level QA."""
    root = Path(root)
    local = root / "local"
    holds = json.loads((root / "document_holds.json").read_text(encoding="utf8"))["documents"]
    groups = json.loads((root / "document_families.json").read_text(encoding="utf8"))["groups"]
    pool = [json.loads(line) for line in
            (local / f"provisional_item_pool_{pool_number:02d}.jsonl").read_text(encoding="utf8").splitlines()
            if line.strip()]
    reviews = []
    for path in local.glob("stage*.json"):
        match = REVIEW_NAMES.fullmatch(path.name)
        if not match:
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        reviews.append((int(match[1]), path.name, raw, value))
    reviews.sort(key=lambda row: (row[0], row[1]), reverse=True)
    result = []
    for document_id in dict.fromkeys(document_ids):
        members = {document_id}
        while True:
            expanded = members | {member for group in groups
                                  if members.intersection(group["document_ids"])
                                  for member in group["document_ids"]}
            if expanded == members:
                break
            members = expanded
        pointers = []
        for stage, name, raw, value in reviews:
            # IDs are opaque tokens. A mention may occur in a key or a rationale.
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(document_id) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, json.dumps(value, ensure_ascii=False)):
                pointers.append({"stage": stage, "record": "local/" + name,
                                 "sha256": hashlib.sha256(raw).hexdigest(),
                                 "status": value.get("status", "open_record_for_disposition")})
        counts = {split: sum(row.get("split") == split and row.get("document_group") in members
                             for row in pool) for split in ("test", "dev")}
        result.append({"document_id": document_id, "family_document_ids": sorted(members),
                       "family_item_counts": counts,
                       "held_family_document_ids": sorted(members.intersection(holds)),
                       "prior_review_records": pointers,
                       "review_lookup": "mentions_found" if pointers else "no_tracked_review_found"})
    return {"status": "LOOKUP_ONLY_NOT_CLEARANCE", "pool_number": pool_number,
            "warning": "Mentions can be incidental. Read the referenced decisions and related-family records. "
                       "Missing records do not establish clearance; retained counts are not benchmark acceptance.",
            "documents": result}
