"""Shared, binary-relevance retrieval metrics over unique evidence pages.

Recall@5 is the primary retrieval metric. Page IDs are labels, never retrieval
inputs. Duplicate predictions consume a rank but cannot earn a second hit.
"""
import math
import re
import warnings
from typing import Any, Dict, Iterable, List

PRIMARY_METRIC = "recall_at_5"
METRIC_VERSION = "binary_page_v2"
METRIC_NAMES = ("recall_at_1", "recall_at_5", "hit_at_5", "mrr_at_10", "ndcg_at_10")


def evaluation_depth(top_k: int) -> int:
    if top_k < 10:
        raise ValueError("Retrieval evaluation reports MRR@10/nDCG@10; use top_k >= 10 (Recall still uses only the first 5).")
    return top_k


def canonical_page(value: Any, page_number: Any = None) -> str:
    value = str(value or "").strip().replace("\\", "/")
    if not value:
        return ""
    match = re.search(r"(?:#page=|::p)(\d+)$", value)
    if match:
        return f"{value[:match.start()]}::p{int(match.group(1)):04d}"
    # Text chunks are distinct indexed evidence units, not physical PDF pages.
    if "#chunk=" in value:
        return value
    return f"{value}::p{int(page_number or 1):04d}"


def page_aliases(page: Dict[str, Any]) -> set[str]:
    aliases = set()
    page_number = page.get("page_number")
    key_page = re.search(r"(?:#page=|::p)(\d+)$", str(page.get("page_key") or ""))
    if page_number is None and key_page:
        page_number = int(key_page.group(1))
    for field in ("source", "page_key"):
        key = canonical_page(page.get(field), page_number)
        if key:
            aliases.add("page:" + key)
    if page.get("page_id"):
        aliases.add("id:" + str(page["page_id"]))
    return aliases


def gold_groups(example: Dict[str, Any]) -> List[set[str]]:
    """Each aligned source/key/ID describes ONE page, even when IDs differ."""
    pages = example.get("gold_evidence")
    if pages is None:
        columns = {
            "source": example.get("evidence_sources") or example.get("citations") or [],
            "page_key": example.get("gold_page_keys") or [],
            "page_id": example.get("evidence_page_ids") or [],
        }
        lengths = {len(values) for values in columns.values() if values}
        if len(lengths) > 1:
            raise ValueError("Gold source/key/ID lists must be aligned; use gold_evidence for explicit page aliases.")
        pages = [{field: values[i] for field, values in columns.items() if values}
                 for i in range(max(lengths, default=0))]
    groups: List[set[str]] = []
    for page in pages:
        aliases = page_aliases(page)
        if not aliases:
            continue
        overlapping = [group for group in groups if group & aliases]
        for group in overlapping:
            aliases |= group
            groups.remove(group)
        groups.append(aliases)
    if not groups:
        raise ValueError(f"Answerable query {example.get('qa_id', example.get('question', ''))!r} has no gold evidence.")
    return groups


def score_ranking(example: Dict[str, Any], ranked: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    groups = gold_groups(example)
    found = set()
    positions = []
    for rank, candidate in enumerate(ranked, 1):
        if rank > 10:
            break
        aliases = page_aliases(candidate)
        matches = {i for i, group in enumerate(groups) if aliases & group}
        if len(matches) > 1:
            raise ValueError("A retrieved page matches multiple gold pages; check conflicting source/key aliases.")
        new = matches - found
        if new:
            found |= new
            positions.append(rank)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(groups)) + 1))
    return {
        "recall_at_1": sum(rank <= 1 for rank in positions) / len(groups),
        "recall_at_5": sum(rank <= 5 for rank in positions) / len(groups),
        "hit_at_5": float(any(rank <= 5 for rank in positions)),
        "mrr_at_10": 1.0 / positions[0] if positions else 0.0,
        "ndcg_at_10": sum(1.0 / math.log2(rank + 1) for rank in positions) / ideal,
    }


def validate_queries(examples: List[Dict[str, Any]], strict: bool = False) -> None:
    seen = set()
    duplicates = 0
    for example in examples:
        gold_groups(example)
        key = " ".join(str(example["question"]).split()).casefold()
        if not key:
            raise ValueError("Retrieval questions must not be empty.")
        duplicates += key in seen
        seen.add(key)
    if duplicates:
        message = (f"{duplicates} duplicate retrieval questions: repeated templates can weight queries multiple times "
                   "or assign conflicting gold evidence. Scope questions and consolidate judged positives; "
                   "see docs/retrieval_evaluation.md.")
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def mean_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {name: sum(row[name] for row in rows) / len(rows) if rows else 0.0 for name in METRIC_NAMES}
