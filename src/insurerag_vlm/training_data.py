import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import ModelConfig
from .hybrid_pipeline import DocumentRetrievalPipeline
from .qa import generate_policy_qa_pairs
from .query_understanding import understand_query
from .sft import read_sft_records


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _source_doc_id(source: str) -> str:
    return str(source or "").split("#page=", 1)[0]


def _source_page_key(source: str) -> str:
    doc_id, _, page_part = str(source or "").partition("#page=")
    if page_part.isdigit():
        return f"{doc_id}::p{int(page_part):04d}"
    return str(source or "")


def _split_record_lists(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"train": [], "valid": [], "test": []}
    for record in records:
        split = str(record.get("split") or "train")
        if split not in grouped:
            split = "train"
        grouped[split].append(record)
    return grouped


def _split_doc_ids(doc_ids: List[str], seed: int = 42, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> Dict[str, str]:
    doc_ids = sorted({doc_id for doc_id in doc_ids if doc_id})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    total = len(doc_ids)
    if total <= 1:
        return {doc_id: "train" for doc_id in doc_ids}
    train_n = max(1, int(total * train_ratio))
    valid_n = max(1, int(total * valid_ratio)) if total >= 3 else 0
    if train_n + valid_n >= total:
        train_n = max(1, total - 1)
        valid_n = 0 if total == 2 else 1
    split_names = ["train"] * train_n + ["valid"] * valid_n
    while len(split_names) < total:
        split_names.append("test")
    return {doc_id: split for doc_id, split in zip(doc_ids, split_names)}


@dataclass
class TrainingCorpusBuildConfig:
    data_folder: Path
    output_dir: Path = Path("reports/training_data")
    qa_path: Optional[Path] = None
    hard_negatives_path: Optional[Path] = None
    sft_dataset_path: Path = Path("data/04_curated/sft_dataset.jsonl")
    index_dir: Path = Path("data/train_index")
    retrieval_model: str = "local-hashing"
    retrieval_mode: str = "hybrid_multimodal"
    corpus_source: str = "auto"
    enable_image_signal: bool = True
    target_qa_count: int = 300
    unsupported_count: int = 50
    top_k: int = 5
    max_negatives: int = 4
    max_sft_pages: int = 3

    def __post_init__(self) -> None:
        self.data_folder = Path(self.data_folder)
        self.output_dir = Path(self.output_dir)
        self.sft_dataset_path = Path(self.sft_dataset_path)
        self.index_dir = Path(self.index_dir)
        if self.qa_path is not None:
            self.qa_path = Path(self.qa_path)
        if self.hard_negatives_path is not None:
            self.hard_negatives_path = Path(self.hard_negatives_path)


def _pick_positive_snippets(
    page_record: Dict[str, Any],
    snippets_by_page: Dict[str, List[Dict[str, Any]]],
    question: str,
    evidence_text: str,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    candidates = snippets_by_page.get(str(page_record.get("page_key")), [])
    question_terms = set(DocumentRetrievalPipeline._key_terms(question))
    evidence_terms = set(DocumentRetrievalPipeline._terms(evidence_text, min_len=3))
    scored = []
    for snippet in candidates:
        text = str(snippet.get("text", ""))
        text_terms = set(DocumentRetrievalPipeline._terms(text, min_len=3))
        score = len(question_terms & text_terms) + len(evidence_terms & text_terms)
        if evidence_text and evidence_text[:80] in text:
            score += 5
        if score > 0:
            scored.append((score, snippet))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [snippet for _, snippet in scored[:max_items]]


def _ensure_seed_qa(config: TrainingCorpusBuildConfig) -> tuple[Path, Path]:
    if config.qa_path and config.hard_negatives_path:
        return config.qa_path, config.hard_negatives_path
    sft_records = read_sft_records(config.sft_dataset_path)
    if sft_records:
        seed_output_dir = config.output_dir / "seed_qa_from_sft"
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        doc_to_split = _split_doc_ids([_source_doc_id(str(record.get("source") or "")) for record in sft_records])
        qa_rows = []
        for idx, record in enumerate(sft_records):
            source = str(record.get("source") or "")
            doc_id = _source_doc_id(source)
            qa_rows.append(
                {
                    "qa_id": str(record.get("record_id") or f"sft_seed_{idx}"),
                    "question": str(record.get("question") or ""),
                    "answer": str(record.get("answer") or ""),
                    "evidence_sources": [source] if source and bool(record.get("answerable", True)) else [],
                    "evidence_text": str(record.get("evidence") or ""),
                    "answerable": bool(record.get("answerable", True)),
                    "split": doc_to_split.get(doc_id, "train" if bool(record.get("answerable", True)) else "test"),
                }
            )
        qa_path = _write_jsonl(qa_rows, seed_output_dir / "qa_pairs.jsonl")
        hard_negatives_path = _write_jsonl([], seed_output_dir / "hard_negatives.jsonl")
        return qa_path, hard_negatives_path
    seed_output_dir = config.output_dir / "seed_qa"
    result = generate_policy_qa_pairs(
        config.data_folder,
        output_dir=seed_output_dir,
        target_count=config.target_qa_count,
        unsupported_count=config.unsupported_count,
    )
    return result.qa_path, result.hard_negatives_path


def build_training_corpora(config: TrainingCorpusBuildConfig) -> Dict[str, Any]:
    qa_path, hard_negatives_path = _ensure_seed_qa(config)
    qa_records = _read_jsonl(qa_path)
    hard_negative_records = _read_jsonl(hard_negatives_path)
    sft_records = read_sft_records(config.sft_dataset_path)

    pipeline = DocumentRetrievalPipeline(
        ModelConfig(
            retrieval_model=config.retrieval_model,
            retrieval_mode=config.retrieval_mode,
            corpus_source=config.corpus_source,
            enable_image_signal=config.enable_image_signal,
            index_dir=config.index_dir,
        )
    )
    pipeline.build_index(config.data_folder)
    corpus = pipeline._load_hybrid_corpus(config.data_folder, include_images=config.enable_image_signal)  # noqa: SLF001
    page_by_source = {str(page.get("source")): page for page in corpus["pages"]}
    snippets_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for snippet in corpus["snippets"]:
        snippets_by_page.setdefault(str(snippet.get("page_key")), []).append(snippet)

    negatives_by_qa: Dict[str, List[Dict[str, Any]]] = {}
    for record in hard_negative_records:
        negatives_by_qa.setdefault(str(record.get("qa_id")), []).append(record)

    retrieval_records: List[Dict[str, Any]] = []
    calibration_records: List[Dict[str, Any]] = []
    for qa in qa_records:
        split = str(qa.get("split") or "train")
        understanding = understand_query(str(qa.get("question") or ""))
        answerable = bool(qa.get("answerable", True))
        positive_sources = list(qa.get("evidence_sources") or qa.get("citations") or [])
        hard_negative_sources = [
            str(item.get("negative_source"))
            for item in negatives_by_qa.get(str(qa.get("qa_id")), [])
            if str(item.get("negative_source"))
        ]
        ranked_sources = [
            str(page.get("source"))
            for page in pipeline.rank_pages(str(qa.get("question")), config.data_folder, top_k=max(config.top_k + 2, 8))
        ]
        for source in ranked_sources:
            if source not in positive_sources and source not in hard_negative_sources:
                hard_negative_sources.append(source)
            if len(hard_negative_sources) >= config.max_negatives:
                break

        if answerable and positive_sources:
            first_positive = page_by_source.get(positive_sources[0])
            positive_snippets = _pick_positive_snippets(
                first_positive or {},
                snippets_by_page,
                question=str(qa.get("question") or ""),
                evidence_text=str(qa.get("evidence_text") or qa.get("evidence") or ""),
            ) if first_positive else []
            retrieval_records.append(
                {
                    "record_id": qa.get("qa_id"),
                    "question": qa.get("question"),
                    "split": split,
                    "question_type": understanding.intent,
                    "coverage_tags": understanding.target_coverages,
                    "gold_sources": positive_sources,
                    "gold_page_keys": [_source_page_key(source) for source in positive_sources],
                    "gold_snippet_ids": [snippet.get("record_id") for snippet in positive_snippets],
                    "positive_page_texts": [page_by_source[source]["text"] for source in positive_sources if source in page_by_source],
                    "positive_snippet_texts": [snippet.get("text") for snippet in positive_snippets],
                    "hard_negative_sources": hard_negative_sources[: config.max_negatives],
                    "hard_negative_texts": [
                        page_by_source[source]["text"]
                        for source in hard_negative_sources[: config.max_negatives]
                        if source in page_by_source
                    ],
                    "evidence_text": qa.get("evidence_text") or qa.get("evidence") or "",
                    "has_table_signal": bool(first_positive and first_positive.get("primary_clause_type") in {"limit", "deductible", "premium"}),
                    "has_graph_signal": bool(
                        first_positive and (
                            first_positive.get("document_type") == "endorsement"
                            or first_positive.get("primary_clause_type") in {"exclusion", "definition"}
                        )
                    ),
                    "document_type": first_positive.get("document_type") if first_positive else None,
                    "section_anchor": first_positive.get("section_anchor") if first_positive else None,
                }
            )

        if split in {"valid", "test"}:
            calibration_records.append(
                {
                    "qa_id": qa.get("qa_id"),
                    "question": qa.get("question"),
                    "answer": qa.get("answer", ""),
                    "answerable": answerable,
                    "evidence_sources": positive_sources,
                    "split": split,
                }
            )

    split_by_source: Dict[str, str] = {}
    for qa in qa_records:
        source_candidates = list(qa.get("evidence_sources") or qa.get("citations") or [])
        if not source_candidates and qa.get("source"):
            source_candidates = [str(qa.get("source"))]
        if source_candidates:
            split_by_source[_source_doc_id(source_candidates[0])] = str(qa.get("split") or "train")

    rag_sft_records: List[Dict[str, Any]] = []
    for record in sft_records:
        question = str(record.get("question") or "")
        source = str(record.get("source") or "")
        split = split_by_source.get(_source_doc_id(source), "train" if record.get("answerable", True) else "test")
        ranked_pages = pipeline.rank_pages(question, config.data_folder, top_k=max(config.max_sft_pages + 2, 6))
        selected_pages: List[Dict[str, Any]] = []
        if record.get("answerable", True) and source in page_by_source:
            selected_pages.append(page_by_source[source])
        for page in ranked_pages:
            if any(str(existing.get("source")) == str(page.get("source")) for existing in selected_pages):
                continue
            selected_pages.append(page)
            if len(selected_pages) >= config.max_sft_pages:
                break
        if not selected_pages:
            selected_pages = ranked_pages[: config.max_sft_pages]
        understanding = understand_query(question)
        packed_evidence = pipeline.pack_long_context(selected_pages, config.max_sft_pages, understanding=understanding)
        rag_sft_records.append(
            {
                **record,
                "evidence": packed_evidence,
                "split": split,
                "dataset_variant": "retrieval_conditioned",
                "retrieval_context_sources": [page.get("source") for page in selected_pages],
                "gold_in_context": bool(source and any(str(page.get("source")) == source for page in selected_pages)),
            }
        )

    retrieval_splits = _split_record_lists(retrieval_records)
    rag_sft_splits = _split_record_lists(rag_sft_records)
    calibration_splits = _split_record_lists(calibration_records)

    outputs = {
        "seed_qa_path": str(qa_path),
        "seed_hard_negatives_path": str(hard_negatives_path),
        "retrieval_train_path": str(_write_jsonl(retrieval_splits["train"], config.output_dir / "retrieval_train.jsonl")),
        "retrieval_dev_path": str(_write_jsonl(retrieval_splits["valid"], config.output_dir / "retrieval_dev.jsonl")),
        "retrieval_test_path": str(_write_jsonl(retrieval_splits["test"], config.output_dir / "retrieval_test.jsonl")),
        "rag_sft_train_path": str(_write_jsonl(rag_sft_splits["train"], config.output_dir / "rag_sft_train.jsonl")),
        "rag_sft_dev_path": str(_write_jsonl(rag_sft_splits["valid"], config.output_dir / "rag_sft_dev.jsonl")),
        "rag_sft_test_path": str(_write_jsonl(rag_sft_splits["test"], config.output_dir / "rag_sft_test.jsonl")),
        "calibration_dev_path": str(_write_jsonl(calibration_splits["valid"], config.output_dir / "calibration_dev.jsonl")),
        "calibration_test_path": str(_write_jsonl(calibration_splits["test"], config.output_dir / "calibration_test.jsonl")),
        "retrieval_counts": {split: len(records) for split, records in retrieval_splits.items()},
        "rag_sft_counts": {split: len(records) for split, records in rag_sft_splits.items()},
        "calibration_counts": {split: len(records) for split, records in calibration_splits.items()},
    }
    summary_path = config.output_dir / "training_corpora_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary_path"] = str(summary_path)
    outputs["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    return outputs
