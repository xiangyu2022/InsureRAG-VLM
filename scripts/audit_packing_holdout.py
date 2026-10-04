"""Read-only feasibility audit; never relabel exposed examples as a new holdout."""
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path("reports/packing_20261004")


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def key(source):
    source = re.sub(r"#page=\d+$", "", source.replace("\\", "/"))
    url = urlsplit(source)
    return re.sub(r"^www\.", "", url.netloc.lower()) + unquote(url.path).lower().rstrip("/") if url.scheme else source.lower()


def shingles(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(zip(*(tokens[i:] for i in range(5))))


def main():
    old = read("data/04_curated/sft_dataset.jsonl")
    used = {key(r["source"]) for r in old}
    used.add(key("nc_travel_insurance_guide.pdf"))  # Prior CLI output inspected.
    pages = read("data/04_curated/rag_pages.jsonl")
    candidates = defaultdict(list)
    for row in pages:
        source = key(row["citation"])
        if source not in used:
            candidates[source].append(row)
    old_evidence = {r["evidence"] for r in old}
    old_shingles = [shingles(text) for text in old_evidence]
    summaries = []
    for source, records in sorted(candidates.items()):
        max_jaccard, max_containment = 0.0, 0.0
        for row in records:
            current = shingles(row["text"])
            for prior in old_shingles:
                intersection = len(current & prior)
                max_jaccard = max(max_jaccard, intersection / max(1, len(current | prior)))
                max_containment = max(max_containment, intersection / max(1, min(len(current), len(prior))))
        summaries.append({"source": source, "pages": len(records), "max_jaccard_to_prior_sft": max_jaccard,
                          "max_shorter_span_containment": max_containment,
                          "duplicate_or_shared_clause_flag": max_jaccard >= .85 or max_containment >= .8})
    eligible = [r for r in summaries if not r["duplicate_or_shared_clause_flag"]]
    result = {"prior_sft_source_identifiers": len(used)-1,
              "prior_cli_document_also_excluded": "nc_travel_insurance_guide.pdf",
              "candidate_document_identifiers": len(candidates), "candidates": summaries,
              "unflagged_document_identifiers": len(eligible),
              "decision": "No new confirmatory holdout created. Candidate sources have no independent held-out answerability/reference labels; unflagged lexical similarity is not proof of publication-version independence. This stage reports dev and explicitly exposed-test regression only.",
              "limitations": "This is a conservative source feasibility screen, not semantic duplicate detection. No new labels were generated or selected from observed test errors."}
    (ROOT / "holdout_feasibility.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
