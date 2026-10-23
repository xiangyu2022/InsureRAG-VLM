"""Prepare only the existing DOL ACA historical source snapshot; never QA labels."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .chroma_knowledge import canonical, hash_value


SNAPSHOT_SHA256 = "54b596fd4228fae8e71c3bf474bcbb33a2d0e06a57193d328d4998b1f8f1f1bf"
RIGHTS = "US federal government publication; preserve source/date; not a determination of current legal status."
URL_PREFIX = "https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/resource-center/faqs/aca-part-"


def chunks(text, maximum=1600):
    """Contiguous original-text slices, ending at paragraph/word boundaries."""
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            boundary = text.rfind("\n\n", start + maximum // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + maximum // 2, end)
            if boundary >= 0:
                end = boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1)
        if text[start:end].strip():
            yield start, end, text[start:end]
        start = end


def prepare_snapshot(folder):
    source = Path(folder) / "records.jsonl"
    manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
    blob = source.read_bytes()
    snapshot_hash = sha256(blob).hexdigest()
    if snapshot_hash != SNAPSHOT_SHA256 or manifest.get("records_sha256") != snapshot_hash:
        raise ValueError("DOL source snapshot differs from the reviewed historical selection")
    sources = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    expected = {f"dol_html_aca-part-{i}" for i in range(56, 72)}
    if len(sources) != 16 or manifest.get("records") != 16 or {row.get("id") for row in sources} != expected:
        raise ValueError("Unexpected DOL source membership")
    records = []
    for row in sources:
        part = row["id"].removeprefix("dol_html_aca-part-")
        if row.get("source_url") != URL_PREFIX + part or row.get("rights") != RIGHTS:
            raise ValueError("DOL source URL or recorded use basis differs")
        text = row["text"]
        if sha256(text.encode("utf-8")).hexdigest() != row.get("text_sha256"):
            raise ValueError("DOL cached text hash mismatch")
        raw_hash = hash_value(row.get("html_sha256"), "html_sha256")
        header_date = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b", text[:3000])
        for start, end, excerpt in chunks(text):
            records.append({
                "record_id": f"{row['id']}::chars{start}-{end}", "doc_id": row["id"],
                "page_number": 0, "page_id": row["id"] + "::html",
                "source_record_id": row["id"], "text": excerpt,
                "citation": f"{row['source_url']} [cached text chars {start}:{end}]",
                "source_url": row["source_url"], "source_sha256": raw_hash,
                "source_text_sha256": row["text_sha256"], "snapshot_sha256": snapshot_hash,
                "snapshot": "dol_additional_v1@2026-10-01", "source_origin": "archived_official_guidance",
                "retrieved_utc": row["retrieved_utc"], "publisher": "US Department of Labor",
                "jurisdiction": "US-federal", "rights": RIGHTS,
                "source_header_date": header_date.group(0) if header_date else "not_extracted",
                "applicability": "historical reference; current applicability not revalidated",
                "source_hash_basis": "HTML SHA-256 recorded in acquisition; original HTML not rehashed in this run",
                "char_start": start, "char_end": end,
                "offset_basis": "cached extracted source text, Unicode code points",
            })
    receipt = {"snapshot": "dol_additional_v1@2026-10-01", "snapshot_sha256": snapshot_hash,
               "documents": len(sources), "chunks": len(records), "chunk_max_characters": 1600,
               "rights_basis": RIGHTS, "source_scope": "DOL ACA implementation parts 56-71",
               "source_bytes_modified": False, "qa_labels_imported": False,
               "text_hashes_verified": 16, "original_html_rehashed": False,
               "current_legal_applicability_validated": False}
    return records, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/research_corpus/dol_additional_v1"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Private output directory, preferably under .knowledge")
    args = parser.parse_args()
    records, receipt = prepare_snapshot(args.snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "clauses.jsonl"
    output.write_text("\n".join(canonical(row) for row in records) + "\n", encoding="utf-8", newline="\n")
    receipt["clauses_sha256"] = sha256(output.read_bytes()).hexdigest()
    (args.output_dir / "snapshot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
