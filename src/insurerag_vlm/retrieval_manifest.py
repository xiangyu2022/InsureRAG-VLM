"""Explicit document-scoped synthetic retrieval manifests, with traceable qrels."""
import hashlib
from typing import Any, Dict, List

from .retrieval_metrics import METRIC_VERSION, PRIMARY_METRIC, gold_groups, validate_queries


def document_id(source: str) -> str:
    return source.split("#page=", 1)[0].split("#chunk=", 1)[0].split("::p", 1)[0]


def scope_and_merge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add document context, never a gold page number or answer to the query.

    Only existing labeled positives are merged. This is a scoped synthetic
    benchmark, not new human relevance judgments or unrestricted user QA.
    """
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        if not row.get("answerable", True):
            raise ValueError("Scoped retrieval manifests only accept answerable examples.")
        sources = row.get("evidence_sources") or row.get("citations") or []
        docs = {document_id(source) for source in sources}
        if len(docs) != 1:
            raise ValueError("Scoping requires evidence from exactly one document per input row.")
        doc = next(iter(docs))
        question = " ".join(str(row["question"]).split())
        if not doc or not question:
            raise ValueError("Document scope and question must not be empty.")
        key = (doc, question.casefold())
        if key not in grouped:
            grouped[key] = {
                "qa_id": "scoped::" + hashlib.sha256((doc + "\n" + key[1]).encode()).hexdigest()[:16],
                "question": f"In the document {doc}, {question[0].lower() + question[1:]}",
                "answerable": True,
                "source_doc_id": doc,
                "question_type": "document_scoped_synthetic",
                "gold_evidence": [],
                "provenance": [],
                "metric_version": METRIC_VERSION,
                "primary_metric": PRIMARY_METRIC,
            }
        target = grouped[key]
        keys = row.get("gold_page_keys") or []
        ids = row.get("evidence_page_ids") or []
        gold_groups(row)  # Reject misaligned source/key lists before converting.
        pages = row.get("gold_evidence") or [
            {"source": source, **({"page_key": keys[i]} if keys else {}),
             **({"page_id": ids[i]} if ids else {})}
            for i, source in enumerate(sources)
        ]
        for page in pages:
            if page not in target["gold_evidence"]:
                target["gold_evidence"].append(page)
        target["provenance"].append({"qa_id": row.get("qa_id"), "original_question": row["question"],
                                     "answer": row.get("answer"), "evidence_text": row.get("evidence_text")})
    result = list(grouped.values())
    for row in result:
        row["evidence_sources"] = list(dict.fromkeys(page["source"] for page in row["gold_evidence"]))
        row["citations"] = row["evidence_sources"]
    validate_queries(result, strict=True)
    return result
