#!/usr/bin/env python3
"""Rebuild scoped manifests from committed labels without models or raw PDFs."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.insurerag_vlm.retrieval_manifest import scope_and_merge
from src.insurerag_vlm.retrieval_metrics import gold_groups


def main():
    root = REPO_ROOT / "reports/retrieval_eval"
    paths = ["clean_exact_source.jsonl", "expanded_targeted/valid.jsonl", "expanded_targeted/test.jsonl",
             "external_official/valid.jsonl", "external_official/test.jsonl"]
    audit = {"metric_version": "binary_page_v2", "primary_metric": "recall_at_5",
             "benchmark_scope": "document_scoped_synthetic", "model_scores": "not_run",
             "note": "Existing positives only; document IDs are query context, page IDs and answers are not. Human relevance review remains necessary.",
             "manifests": {}}
    by_path = {}
    for relative in paths:
        old = [json.loads(line) for line in (root / relative).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = scope_and_merge(old)
        by_path[relative] = rows
        dest = root / "v2" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        audit["manifests"][relative] = {"original_rows": len(old), "queries": len(rows),
            "multi_gold_queries": sum(len(gold_groups(row)) > 1 for row in rows),
            "unique_documents": len({row["source_doc_id"] for row in rows})}
    for name in ("expanded_targeted", "external_official"):
        valid = {row["source_doc_id"] for row in by_path[f"{name}/valid.jsonl"]}
        test = {row["source_doc_id"] for row in by_path[f"{name}/test.jsonl"]}
        if valid & test:
            raise ValueError(f"Document leakage between {name} validation/test: {valid & test}")
    (root / "v2/manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
