#!/usr/bin/env python3
import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.data import PageDocument, load_documents
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import compute_retrieval_metrics

EXCLUDE_PATTERNS = (
    "skip to main content",
    "file a complaint",
    "contact us",
    "submit",
    "search",
    "menu",
    "online consumer complaint",
    "public records request",
    "fax:",
    "email (choose one)",
    "po box",
    "keep for your records",
    "view more:",
    "market share reports",
    "consumer complaint study",
    "annual report of the commissioner",
    "reports & publications",
    "newsroom",
    "virtual viewing room",
)

QUESTION_TEMPLATES = {
    "actual cash value": "How does this source explain actual cash value?",
    "replacement cost": "How does this source explain replacement cost?",
    "deductible": "What does this source say about deductibles?",
    "premium": "What does this source explain about insurance premiums?",
    "declarations": "What does this source explain about the declarations page?",
    "endorsement": "How does this source describe endorsements?",
    "liability": "What liability coverage concept is explained here?",
    "limit": "What does this source say about coverage limits?",
    "limits": "What does this source say about coverage limits?",
    "coverage": "What coverage concept is explained by this source?",
}

SELECTED_SOURCES: List[Dict[str, Any]] = [
    {
        "source_path": Path("data/00_raw/external/public_docs/naic_glossary_insurance_terms.txt"),
        "dest_path": Path("public_docs/naic_glossary_insurance_terms.txt"),
        "preferred_terms": ["actual cash value", "deductible", "endorsement", "liability", "premium"],
        "source_family": "naic",
        "document_kind": "glossary",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/naic_auto_insurance_topic.txt"),
        "dest_path": Path("public_docs/naic_auto_insurance_topic.txt"),
        "preferred_terms": ["deductible", "liability", "premium", "coverage"],
        "source_family": "naic",
        "document_kind": "topic_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/naic_homeowners_insurance_topic.txt"),
        "dest_path": Path("public_docs/naic_homeowners_insurance_topic.txt"),
        "preferred_terms": ["replacement cost", "actual cash value", "deductible", "coverage"],
        "source_family": "naic",
        "document_kind": "topic_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/md_homeowners_declarations_page.pdf"),
        "dest_path": Path("public_docs/md_homeowners_declarations_page.pdf"),
        "preferred_terms": ["declarations", "deductible", "premium", "limit"],
        "source_family": "state_public_doc",
        "document_kind": "declarations_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/md_auto_insurance_guide.pdf"),
        "dest_path": Path("public_docs/md_auto_insurance_guide.pdf"),
        "preferred_terms": ["liability", "deductible", "premium", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/md_homeowners_insurance_guide.pdf"),
        "dest_path": Path("public_docs/md_homeowners_insurance_guide.pdf"),
        "preferred_terms": ["replacement cost", "actual cash value", "deductible", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/ca_auto_insurance_terms.txt"),
        "dest_path": Path("public_docs/ca_auto_insurance_terms.txt"),
        "preferred_terms": ["deductible", "liability", "limit", "endorsement", "premium"],
        "source_family": "state_public_doc",
        "document_kind": "glossary",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/ca_auto_insurance_guide.txt"),
        "dest_path": Path("public_docs/ca_auto_insurance_guide.txt"),
        "preferred_terms": ["deductible", "liability", "premium", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/wa_consumer_insurance_glossary.txt"),
        "dest_path": Path("public_docs/wa_consumer_insurance_glossary.txt"),
        "preferred_terms": ["actual cash value", "replacement cost", "deductible", "endorsement"],
        "source_family": "state_public_doc",
        "document_kind": "glossary",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/tx_auto_insurance_glossary.txt"),
        "dest_path": Path("public_docs/tx_auto_insurance_glossary.txt"),
        "preferred_terms": ["deductible", "premium", "actual cash value", "endorsement", "liability"],
        "source_family": "state_public_doc",
        "document_kind": "glossary",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/tx_home_insurance_guide.txt"),
        "dest_path": Path("public_docs/tx_home_insurance_guide.txt"),
        "preferred_terms": ["replacement cost", "actual cash value", "deductible", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/de_homeowners_guide.pdf"),
        "dest_path": Path("public_docs/de_homeowners_guide.pdf"),
        "preferred_terms": ["replacement cost", "actual cash value", "deductible", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/de_auto_insurance_guide.pdf"),
        "dest_path": Path("public_docs/de_auto_insurance_guide.pdf"),
        "preferred_terms": ["liability", "deductible", "premium", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/ny_homeowners_basic_coverage.txt"),
        "dest_path": Path("public_docs/ny_homeowners_basic_coverage.txt"),
        "preferred_terms": ["coverage", "deductible", "limit", "replacement cost"],
        "source_family": "state_public_doc",
        "document_kind": "coverage_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/ny_home_insurance_replacement_cost.txt"),
        "dest_path": Path("public_docs/ny_home_insurance_replacement_cost.txt"),
        "preferred_terms": ["replacement cost", "limit", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "coverage_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/sc_understanding_your_deductible.txt"),
        "dest_path": Path("public_docs/sc_understanding_your_deductible.txt"),
        "preferred_terms": ["deductible", "premium"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/public_docs/sc_purchasing_home_insurance.txt"),
        "dest_path": Path("public_docs/sc_purchasing_home_insurance.txt"),
        "preferred_terms": ["replacement cost", "actual cash value", "deductible", "coverage"],
        "source_family": "state_public_doc",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/state_doi_docs/louisiana/louisiana_consumers_guide_to_auto_insurance.pdf"),
        "dest_path": Path("state_doi_docs/louisiana_consumers_guide_to_auto_insurance.pdf"),
        "preferred_terms": ["liability", "deductible", "premium", "coverage"],
        "source_family": "state_doi",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/state_doi_docs/mississippi/mississippi_consumer_quick_guide_home_insurance.pdf"),
        "dest_path": Path("state_doi_docs/mississippi_consumer_quick_guide_home_insurance.pdf"),
        "preferred_terms": ["replacement cost", "deductible", "coverage", "actual cash value"],
        "source_family": "state_doi",
        "document_kind": "consumer_guide",
    },
    {
        "source_path": Path("data/00_raw/external/state_doi_docs/vermont/vermont_consumer_advisory_rising_premiums.txt"),
        "dest_path": Path("state_doi_docs/vermont_consumer_advisory_rising_premiums.txt"),
        "preferred_terms": ["premium"],
        "source_family": "state_doi",
        "document_kind": "consumer_advisory",
    },
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_segments(text: str) -> List[str]:
    text = text.replace("\r", "\n")
    text = re.sub(r"(?m)^(Title|Category|Source|URL):.*$", "", text)
    raw_segments = re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    segments: List[str] = []
    for raw in raw_segments:
        seg = _normalize_ws(raw)
        if len(seg) < 50 or len(seg) > 700:
            continue
        lowered = seg.lower()
        if any(pattern in lowered for pattern in EXCLUDE_PATTERNS):
            continue
        if seg.count(" | ") > 8:
            continue
        segments.append(seg)
    return segments


def _question_for_term(term: str, document_kind: str) -> str:
    if term == "coverage" and document_kind == "declarations_guide":
        return "What does this source explain about the declarations page?"
    return QUESTION_TEMPLATES.get(term, f"What does this source explain about {term}?")


def _source_doc_id(source: str) -> str:
    return source.split("#page=", 1)[0].split("#chunk=", 1)[0]


def _copy_corpus(selected_sources: List[Dict[str, Any]], corpus_root: Path) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for source in selected_sources:
        src = Path(source["source_path"])
        if not src.exists():
            continue
        dest = corpus_root / source["dest_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(source)
    return copied


def _build_doc_config_lookup(selected_sources: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for source in selected_sources:
        dest = Path(source["dest_path"])
        lookup[str(dest)] = source
        lookup[dest.name] = source
    return lookup


def _score_segment(segment: str, term: str) -> int:
    lowered = segment.lower()
    score = 0
    score += 200 if term in lowered else 0
    score += 40 if "means" in lowered or "refers to" in lowered else 0
    score += 30 if "coverage" in lowered or "policy" in lowered else 0
    score += 20 if ":" in segment or " - " in segment else 0
    score -= abs(len(segment) - 180) // 10
    return score


def _segment_is_valid_for_term(segment: str, term: str) -> bool:
    lowered = segment.lower()
    if term not in lowered:
        return False
    if any(pattern in lowered for pattern in EXCLUDE_PATTERNS):
        return False

    if term == "deductible":
        return any(
            token in lowered
            for token in ("deductible is", "deductible means", "out-of-pocket", "percentage-based deductible", "pay out-of-pocket")
        )
    if term == "premium":
        return any(
            token in lowered
            for token in ("premium is", "premium means", "amount you pay", "increase in insurance premiums", "insurance premiums")
        )
    if term == "liability":
        return any(
            token in lowered
            for token in ("liability coverage", "liability insurance", "liability limit", "covers medical expenses")
        )
    if term == "coverage":
        return any(
            token in lowered
            for token in ("coverage includes", "covers", "insurance coverage", "basic coverage", "homeowner and tenant")
        ) and "coverage study" not in lowered
    if term == "limit" or term == "limits":
        return any(
            token in lowered
            for token in ("coverage limit", "coverage limits", "liability limits", "limit of liability", "coverage amount")
        )
    if term == "declarations":
        return "declarations page" in lowered
    if term == "endorsement":
        return "endorsement" in lowered and any(
            token in lowered for token in ("endorsement is", "endorsement means", "policy endorsement")
        )
    if term == "actual cash value":
        return "actual cash value" in lowered
    if term == "replacement cost":
        return "replacement cost" in lowered
    return True


def _choose_segment(segments: List[str], term: str) -> Optional[str]:
    candidates = [segment for segment in segments if _segment_is_valid_for_term(segment, term)]
    if not candidates:
        return None
    return max(candidates, key=lambda segment: _score_segment(segment, term))


def _build_examples(
    corpus_root: Path,
    selected_sources: List[Dict[str, Any]],
    max_questions_per_doc: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config_lookup = _build_doc_config_lookup(selected_sources)
    rows: List[Dict[str, Any]] = []
    docs = load_documents(corpus_root)
    family_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for doc in docs:
        doc_path = str(doc.metadata.get("path") or "")
        cfg = config_lookup.get(doc_path) or config_lookup.get(Path(doc_path).name)
        if not cfg:
            continue
        preferred_terms = cfg["preferred_terms"]
        segments = _split_segments(doc.text)
        if not segments:
            continue

        used_terms = 0
        for term in preferred_terms:
            segment = _choose_segment(segments, term)
            if not segment:
                continue
            answer = segment[:400].rstrip()
            question = _question_for_term(term, cfg["document_kind"])
            source = str(doc.metadata.get("source") or doc.doc_id)
            dedupe_key = (_source_doc_id(source), term, segment)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "qa_id": f"external::{doc_path}::{term}::{len(rows):04d}",
                    "question": question,
                    "answer": answer,
                    "evidence_sources": [source],
                    "citations": [source],
                    "gold_page_keys": [],
                    "evidence_text": segment,
                    "answerable": True,
                    "question_type": f"external_{term.replace(' ', '_')}",
                    "source_family": cfg["source_family"],
                    "document_kind": cfg["document_kind"],
                    "preferred_term": term,
                    "source_doc_id": _source_doc_id(source),
                }
            )
            family_counter[cfg["source_family"]] += 1
            kind_counter[cfg["document_kind"]] += 1
            term_counter[term] += 1
            used_terms += 1
            if used_terms >= max_questions_per_doc:
                break

    metadata = {
        "count": len(rows),
        "source_families": dict(family_counter),
        "document_kinds": dict(kind_counter),
        "preferred_terms": dict(term_counter),
        "selected_documents": len(selected_sources),
    }
    return rows, metadata


def _split_doc_ids(doc_ids: List[str], seed: int, valid_ratio: float, min_valid_examples: int = 40) -> Dict[str, str]:
    unique_doc_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    rng = random.Random(seed)
    rng.shuffle(unique_doc_ids)
    split_map: Dict[str, str] = {}
    counts = Counter(doc_ids)
    total_examples = sum(counts.values())
    target_valid = max(int(total_examples * valid_ratio), min_valid_examples)
    target_valid = min(target_valid, max(1, total_examples - 10))

    valid_examples = 0
    remaining_docs = len(unique_doc_ids)
    for doc_id in unique_doc_ids:
        remaining_docs -= 1
        doc_count = counts.get(doc_id, 0)
        if valid_examples < target_valid and remaining_docs >= 1:
            split_map[doc_id] = "valid"
            valid_examples += doc_count
        else:
            split_map[doc_id] = "test"
    return split_map


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _metrics_to_dict(metrics) -> Dict[str, float]:
    return {
        "evaluated_count": metrics.evaluated_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_5": metrics.recall_at_5,
        "mrr_at_10": metrics.mrr_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
    }


def _run_eval(
    data_folder: Path,
    qa_path: Path,
    index_dir: Path,
    retrieval_model: str,
    retrieval_mode: str,
    enable_image_signal: bool,
    top_k: int,
) -> Dict[str, float]:
    pipeline = DocumentRetrievalPipeline(
        ModelConfig(
            retrieval_model=retrieval_model,
            retrieval_mode=retrieval_mode,
            corpus_source="documents",
            enable_image_signal=enable_image_signal,
            index_dir=index_dir,
        )
    )
    metrics = compute_retrieval_metrics(pipeline, data_folder, qa_path, top_k=top_k)
    return _metrics_to_dict(metrics)


def _write_markdown(output_md: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# External Official Hybrid vs Dense Evaluation",
        "",
        f"- Corpus root: `{payload['corpus_root']}`",
        f"- Manifest root: `{payload['manifest_root']}`",
        f"- Retrieval model: `{payload['retrieval_model']}`",
        f"- Hybrid mode: `{payload['hybrid_mode']}`",
        f"- Dense mode: `{payload['dense_mode']}`",
        f"- Top-k: `{payload['top_k']}`",
        "",
        "## Dataset",
        "",
        f"- Total examples: `{payload['manifest_metadata']['count']}`",
        f"- Selected documents: `{payload['manifest_metadata']['selected_documents']}`",
        f"- Source families: `{payload['manifest_metadata']['source_families']}`",
        f"- Document kinds: `{payload['manifest_metadata']['document_kinds']}`",
        f"- Preferred terms: `{payload['manifest_metadata']['preferred_terms']}`",
        "",
    ]
    if not payload.get("results"):
        lines.extend(
            [
                "## Status",
                "",
                "Dataset prepared. Retrieval evaluation was skipped in `prepare-only` mode.",
                "",
            ]
        )
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    for split_name in ("valid", "test"):
        dense = payload["results"][split_name]["dense"]
        hybrid = payload["results"][split_name]["hybrid"]
        delta = payload["results"][split_name]["delta"]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- QA file: `{payload['results'][split_name]['qa_path']}`",
                "",
                "| metric | dense_only | hybrid | delta |",
                "| --- | ---: | ---: | ---: |",
                f"| evaluated_count | {dense['evaluated_count']} | {hybrid['evaluated_count']} | {delta['evaluated_count']} |",
                f"| recall_at_1 | {dense['recall_at_1']:.4f} | {hybrid['recall_at_1']:.4f} | {delta['recall_at_1']:.4f} |",
                f"| recall_at_5 | {dense['recall_at_5']:.4f} | {hybrid['recall_at_5']:.4f} | {delta['recall_at_5']:.4f} |",
                f"| mrr_at_10 | {dense['mrr_at_10']:.4f} | {hybrid['mrr_at_10']:.4f} | {delta['mrr_at_10']:.4f} |",
                f"| ndcg_at_10 | {dense['ndcg_at_10']:.4f} | {hybrid['ndcg_at_10']:.4f} | {delta['ndcg_at_10']:.4f} |",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a low-noise external official benchmark and compare hybrid vs dense retrieval.")
    parser.add_argument("--corpus-root", type=Path, default=Path("reports/retrieval_eval/external_official/corpus"))
    parser.add_argument("--manifest-root", type=Path, default=Path("reports/retrieval_eval/external_official"))
    parser.add_argument("--index-dir", type=Path, default=Path("reports/retrieval_eval/external_official/index"))
    parser.add_argument("--retrieval-model", type=str, default="models/retrieval/bge-base-insurerag")
    parser.add_argument("--hybrid-mode", type=str, default="hybrid_text")
    parser.add_argument("--dense-mode", type=str, default="dense_only")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.25)
    parser.add_argument("--min-valid-examples", type=int, default=40)
    parser.add_argument("--max-questions-per-doc", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("reports/retrieval_eval/external_official_hybrid_vs_dense.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/retrieval_eval/external_official_hybrid_vs_dense.md"))
    args = parser.parse_args()

    if args.corpus_root.exists():
        shutil.rmtree(args.corpus_root)
    copied_sources = _copy_corpus(SELECTED_SOURCES, args.corpus_root)
    rows, metadata = _build_examples(args.corpus_root, copied_sources, max_questions_per_doc=args.max_questions_per_doc)
    split_map = _split_doc_ids(
        [row["source_doc_id"] for row in rows],
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        min_valid_examples=args.min_valid_examples,
    )

    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for row in rows:
        split = split_map.get(row["source_doc_id"], "test")
        clean_row = {key: value for key, value in row.items() if key != "source_doc_id"}
        if split == "valid":
            valid_rows.append(clean_row)
        else:
            test_rows.append(clean_row)

    valid_path = args.manifest_root / "valid.jsonl"
    test_path = args.manifest_root / "test.jsonl"
    _write_jsonl(valid_rows, valid_path)
    _write_jsonl(test_rows, test_path)

    payload: Dict[str, Any] = {
        "corpus_root": str(args.corpus_root),
        "manifest_root": str(args.manifest_root),
        "manifest_metadata": {
            **metadata,
            "valid_count": len(valid_rows),
            "test_count": len(test_rows),
        },
        "retrieval_model": args.retrieval_model,
        "hybrid_mode": args.hybrid_mode,
        "dense_mode": args.dense_mode,
        "top_k": args.top_k,
        "results": {},
    }

    if not args.prepare_only:
        results: Dict[str, Any] = {}
        for split_name, qa_path in (("valid", valid_path), ("test", test_path)):
            dense = _run_eval(
                args.corpus_root,
                qa_path,
                args.index_dir,
                args.retrieval_model,
                args.dense_mode,
                False,
                args.top_k,
            )
            hybrid = _run_eval(
                args.corpus_root,
                qa_path,
                args.index_dir,
                args.retrieval_model,
                args.hybrid_mode,
                False,
                args.top_k,
            )
            delta = {key: hybrid[key] - dense[key] for key in dense.keys()}
            results[split_name] = {
                "qa_path": str(qa_path),
                "dense": dense,
                "hybrid": hybrid,
                "delta": delta,
            }
        payload["results"] = results

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, payload)


if __name__ == "__main__":
    main()
