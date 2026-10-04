"""Read-only independent checks of frozen splits, including document versions."""
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path("reports/local_20261004")


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def version_key(source):
    source = source.replace("\\", "/")
    parts = urlsplit(source)
    if parts.scheme:
        return re.sub(r"^www\.", "", parts.netloc.lower()) + unquote(parts.path).lower().rstrip("/")
    return unquote(source.split("#")[0]).lower()


def main():
    audit = json.loads((ROOT / "data_audit.json").read_text())
    splits = {s: [json.loads(x) for x in (ROOT / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
              for s in ("train", "dev", "test")}
    checks = {}
    for split, rows in splits.items():
        assert hashlib.sha256((ROOT / f"{split}.jsonl").read_bytes()).hexdigest() == audit["splits"][split]["sha256"]
        assert len({r["id"] for r in rows}) == len(rows)
        for row in rows:
            assert row["split"] == split == audit["document_split"][row["doc_id"]]
            assert row["doc_id"] in audit["group_members"][row["doc_group"]]
            assert hashlib.sha256(norm(row["evidence"]).encode()).hexdigest() == row["evidence_sha256"]
            if row["answerable"]:
                assert norm(row["reference_content"]) in norm(row["evidence"])
            else:
                assert row["target"] == "INSUFFICIENT_EVIDENCE"
        checks[split] = {"rows": len(rows), "integrity": "passed"}
    versions = defaultdict(list)
    for doc, split in audit["document_split"].items():
        versions[version_key(doc)].append((doc, split))
    alias_crossings = [v for v in versions.values() if len({s for _, s in v}) > 1]
    assert not alias_crossings, "Canonical URL/path aliases crossed splits"
    pairs = {}
    for a, b in combinations(splits, 2):
        for field in ("id", "doc_id", "doc_group", "evidence_sha256"):
            assert not ({r[field] for r in splits[a]} & {r[field] for r in splits[b]})
        questions_a = {norm(r["question"]) for r in splits[a]}
        questions_b = {norm(r["question"]) for r in splits[b]}
        pairs[f"{a}/{b}"] = {"forbidden_overlap": 0,
            "shared_question_templates": len(questions_a & questions_b)}
    # Recompute cross-split evidence similarities over ALL original input rows,
    # not only the capped train/dev/test sample. This checks version grouping.
    original_path = Path("data/04_curated/sft_dataset.jsonl")
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == audit["input_sha256"]
    originals = [json.loads(x) for x in original_path.read_text(encoding="utf-8").splitlines()]
    entries = {}
    for row in originals:
        doc = re.sub(r"#page=\d+$", "", row["source"].replace("\\", "/"))
        text = norm(row["evidence"])
        key = (doc, text)
        tokens = text.split()
        entries[key] = {"doc": doc, "split": audit["document_split"][doc],
                        "shingles": set(zip(*(tokens[i:] for i in range(5))))}
    entries = list(entries.values())
    inverted = defaultdict(set)
    closest = []
    violations = []
    containment_flags = []
    for i, item in enumerate(entries):
        candidates = set()
        for shingle in item["shingles"]:
            candidates.update(inverted[shingle])
        for j in candidates:
            other = entries[j]
            if item["split"] == other["split"]:
                continue
            intersection = len(item["shingles"] & other["shingles"])
            similarity = intersection / max(1, len(item["shingles"] | other["shingles"]))
            containment = intersection / max(1, min(len(item["shingles"]), len(other["shingles"])))
            record = {"documents": [item["doc"], other["doc"]], "splits": [item["split"], other["split"]],
                      "jaccard": similarity, "shorter_span_containment": containment}
            closest.append(record)
            if similarity >= .85:
                violations.append(record)
            if containment >= .8:
                containment_flags.append(record)
        for shingle in item["shingles"]:
            inverted[shingle].add(i)
    assert not violations, "Near-duplicate evidence crossed splits under frozen criterion"
    closest.sort(key=lambda r: r["jaccard"], reverse=True)
    result = {"status": "passed_under_frozen_criteria", "split_integrity": checks,
              "pair_checks": pairs, "canonical_alias_crossings": alias_crossings,
              "cross_split_jaccard_at_least_0_85": len(violations),
              "cross_split_shorter_span_containment_at_least_0_8": len(containment_flags),
              "highest_cross_split_similarities": closest[:10],
              "containment_examples": containment_flags[:10],
              "limitations": ["Canonical keys only cover URL scheme/www/query/fragment/case aliases; unrelated publication-version names are not proof of independent documents.",
                              "Shared question templates and shorter copied clauses can remain across document groups.",
                              "No semantic duplicate oracle or pretrained-model contamination audit was performed."]}
    (ROOT / "independent_split_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("highest_cross_split_similarities", "containment_examples")}, indent=2))


if __name__ == "__main__":
    main()
