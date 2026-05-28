import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import ModelConfig
from .data import PageDocument, load_documents
from .evaluation import evaluate_predictions, load_evaluation_examples
from .graph import build_document_graph, build_graph_adjacency, expand_candidate_page_keys
from .insurance_structure import (
    extract_coverage_tags,
    extract_section_metadata,
    infer_clause_types,
    infer_document_type,
    normalize_coverage_labels,
    primary_clause_type,
)
from .ocr import extract_text_from_image
from .query_understanding import QueryUnderstanding, understand_query
from .retriever import EmbeddingRetriever, SparseRetriever, load_index, load_sparse_index
from .tables import build_table_records, serialize_table_record
from .visual import build_lightweight_page_image_embeddings, score_lightweight_page_image_query
from .vlm import VLMClient, format_prompt


_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d+/\d+/\d+\b")


class DocumentRetrievalPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._documents_cache: Dict[tuple[str, bool, str], List[PageDocument]] = {}
        self._hybrid_corpus_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._index_cache: Dict[str, Dict[str, Any]] = {}
        self.retriever = EmbeddingRetriever(
            config.retrieval_model,
            use_hf_api=config.use_hf_api,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
        )
        self.sparse_retriever = SparseRetriever()
        self.vlm_client = VLMClient(
            model_name=config.vlm_model,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
            anthropic_api_key=getattr(config, "anthropic_api_key", None),
            use_hf_api=config.use_hf_api,
        )

    def _index_paths(self) -> Dict[str, Path]:
        base = Path(self.config.index_dir)
        return {
            "snippet_dense": base / "hybrid_snippets_dense.npy",
            "snippet_sparse": base / "hybrid_snippets_sparse.json",
            "snippet_meta": base / "hybrid_snippets.jsonl",
            "page_dense": base / "hybrid_pages_dense.npy",
            "page_sparse": base / "hybrid_pages_sparse.json",
            "page_meta": base / "hybrid_pages.jsonl",
            "table_sparse": base / "hybrid_tables_sparse.json",
            "table_meta": base / "hybrid_tables.jsonl",
            "graph_meta": base / "hybrid_graph.jsonl",
            "page_image": base / "hybrid_page_image.npy",
            "page_image_meta": base / "hybrid_page_image_pages.jsonl",
        }

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _page_key(doc_id: str, page_number: Optional[int]) -> str:
        page_number = int(page_number or 0)
        return f"{doc_id}::p{page_number:04d}"

    @staticmethod
    def _source_to_page_id(source: str) -> str:
        return str(source).replace("/", "_").replace("#page=", "_p")

    @staticmethod
    def _document_priority(document_role: str) -> int:
        priorities = {
            "declarations": 0,
            "schedule": 1,
            "endorsement": 2,
            "base_policy": 3,
            "definition": 4,
            "claim_form": 5,
            "billing": 6,
        }
        return priorities.get(str(document_role or ""), 7)

    @staticmethod
    def _has_metadata_value(value: Any) -> bool:
        return value is not None and value != ""

    def _augment_record_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = str(record.get("text") or "")
        source = str(record.get("source") or record.get("record_id") or "")
        updated = dict(record)
        coverage_tags = extract_coverage_tags(text)
        clause_types = infer_clause_types(text)
        section_meta = extract_section_metadata(text)
        document_type = infer_document_type(source, text)
        updated["coverage_tags"] = coverage_tags
        updated["clause_types"] = clause_types
        updated["primary_clause_type"] = primary_clause_type(text)
        updated["document_type"] = document_type
        updated["document_role"] = str(record.get("document_role") or document_type)
        updated["packet_id"] = str(record.get("packet_id") or record.get("policy_family_id") or record.get("doc_id") or source)
        try:
            updated["document_priority"] = int(record.get("document_priority"))
        except (TypeError, ValueError):
            updated["document_priority"] = self._document_priority(updated["document_role"])
        updated["section_titles"] = section_meta["section_titles"]
        updated["section_path"] = section_meta["section_path"]
        updated["section_anchor"] = section_meta["section_anchor"]
        updated["section_tokens"] = section_meta["section_tokens"]
        updated["form_codes"] = section_meta["form_codes"]
        explicit_form_codes = [
            str(value).replace(" ", "-")
            for value in [record.get("form_code"), record.get("endorsement_code")]
            if str(value or "").strip()
        ]
        if explicit_form_codes:
            updated["form_codes"] = sorted(set(updated["form_codes"]) | set(explicit_form_codes))
        for key in [
            "effective_date",
            "form_code",
            "endorsement_code",
            "sequence_order",
            "source_origin",
            "source_name",
            "source_url",
            "source_authority",
            "authority",
            "content_type",
            "source_file",
            "policy_family_id",
            "version_id",
            "policy_number",
        ]:
            if self._has_metadata_value(record.get(key)):
                updated[key] = record.get(key)
        return updated

    @staticmethod
    def _inherit_page_structure(record: Dict[str, Any], page_record: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(record)
        for key in [
            "section_titles",
            "section_path",
            "section_anchor",
            "section_tokens",
            "form_codes",
            "packet_id",
            "document_role",
            "document_priority",
            "effective_date",
            "form_code",
            "endorsement_code",
            "sequence_order",
            "source_origin",
            "source_name",
            "source_url",
            "source_authority",
            "authority",
            "content_type",
            "source_file",
            "policy_family_id",
            "version_id",
            "policy_number",
        ]:
            if not updated.get(key):
                updated[key] = list(page_record.get(key, [])) if isinstance(page_record.get(key), list) else page_record.get(key)
        if not updated.get("coverage_tags"):
            updated["coverage_tags"] = list(page_record.get("coverage_tags", []) or [])
        return updated

    def _curated_paths(self) -> Dict[str, Path]:
        base = Path(self.config.curated_dataset_dir)
        return {
            "pages": base / "rag_pages.jsonl",
            "snippets": base / "rag_snippets.jsonl",
        }

    def _has_curated_corpus(self) -> bool:
        paths = self._curated_paths()
        return paths["pages"].exists() and paths["snippets"].exists()

    def _load_documents(self, data_folder: Path, render_pdf_pages: Optional[bool] = None) -> List[PageDocument]:
        render_pdf_pages = self.config.render_pdf_pages if render_pdf_pages is None else render_pdf_pages
        cache_key = (
            str(Path(data_folder).resolve()),
            bool(render_pdf_pages),
            str(Path(self.config.pdf_render_dir).resolve()) if self.config.pdf_render_dir else "",
        )
        if cache_key in self._documents_cache:
            return self._documents_cache[cache_key]
        documents = load_documents(
            data_folder,
            render_pdf_pages=render_pdf_pages,
            pdf_render_dir=self.config.pdf_render_dir,
        )
        for doc in documents:
            if doc.image_path and not doc.text:
                doc.text = extract_text_from_image(doc.image_path)
        self._documents_cache[cache_key] = documents
        return documents

    def _expanded_qa_chunks(self, text: str, max_chars: int = 420) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|(?<=:)\s+", cleaned)
            if sentence.strip()
        ]
        chunks = []
        current = ""
        for sentence in sentences:
            if len(sentence) < 35:
                continue
            if len(sentence) > max_chars:
                sentence = sentence[:max_chars].rsplit(" ", 1)[0].strip()
            if current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        if current:
            chunks.append(current)
        return chunks

    def _load_curated_corpus(self) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = str(Path(self.config.curated_dataset_dir).resolve())
        cached = self._hybrid_corpus_cache.get(cache_key)
        if cached:
            return cached
        paths = self._curated_paths()
        pages: List[Dict[str, Any]] = []
        snippets: List[Dict[str, Any]] = []
        for record in self._read_jsonl(paths["pages"]):
            page_number = int(record.get("page") or 0)
            doc_id = str(record.get("doc_id") or "")
            source = str(record.get("citation") or record.get("source_file") or record.get("record_id"))
            page_key = self._page_key(doc_id, page_number)
            pages.append(
                self._augment_record_metadata(
                    {
                        "record_id": record.get("record_id") or page_key,
                        "record_type": "page",
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "page_key": page_key,
                        "parent_page_id": page_key,
                        "source": source,
                        "text": str(record.get("text") or ""),
                        "image_path": None,
                        "packet_id": record.get("packet_id") or record.get("doc_id"),
                        "document_role": record.get("document_role"),
                        "source_origin": record.get("source_origin") or "curated_real_official_document",
                        "source_name": record.get("name") or record.get("source_file") or record.get("doc_id"),
                        "source_url": record.get("source_url"),
                        "source_authority": record.get("authority"),
                        "authority": record.get("authority"),
                        "content_type": record.get("content_type"),
                        "source_file": record.get("source_file"),
                    }
                )
            )
        page_by_key = {str(page["page_key"]): page for page in pages}
        for record in self._read_jsonl(paths["snippets"]):
            page_number = int(record.get("page") or 0)
            doc_id = str(record.get("doc_id") or "")
            source = str(record.get("citation") or record.get("source_file") or record.get("record_id"))
            page_key = str(record.get("parent_page_id") or self._page_key(doc_id, page_number))
            snippet_record = self._augment_record_metadata(
                {
                    "record_id": record.get("record_id") or f"{page_key}::snippet",
                    "record_type": "snippet",
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "page_key": page_key,
                    "parent_page_id": page_key,
                    "source": source,
                    "text": str(record.get("text") or ""),
                    "image_path": None,
                    "packet_id": record.get("packet_id") or record.get("doc_id"),
                    "document_role": record.get("document_role"),
                    "source_origin": record.get("source_origin") or "curated_real_official_document",
                    "source_name": record.get("name") or record.get("source_file") or record.get("doc_id"),
                    "source_url": record.get("source_url"),
                    "source_authority": record.get("authority"),
                    "authority": record.get("authority"),
                    "content_type": record.get("content_type"),
                    "source_file": record.get("source_file"),
                }
            )
            snippets.append(self._inherit_page_structure(snippet_record, page_by_key.get(page_key, {})))
        corpus = {"pages": pages, "snippets": snippets}
        self._hybrid_corpus_cache[cache_key] = corpus
        return corpus

    def _load_document_corpus(self, data_folder: Path, include_images: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        render_pdf_pages = include_images and self.config.enable_image_signal
        documents = self._load_documents(data_folder, render_pdf_pages=render_pdf_pages)
        pages: List[Dict[str, Any]] = []
        snippets: List[Dict[str, Any]] = []
        for doc in documents:
            source = str(doc.metadata.get("source", doc.doc_id))
            doc_id = str(doc.metadata.get("path") or doc.doc_id.split("#page=", 1)[0])
            page_key = self._page_key(doc_id, doc.page_number)
            document_metadata = {
                key: value
                for key, value in doc.metadata.items()
                if key not in {"source", "page", "path"} and self._has_metadata_value(value)
            }
            page_record = self._augment_record_metadata(
                {
                    "record_id": page_key,
                    "record_type": "page",
                    "doc_id": doc_id,
                    "page_number": int(doc.page_number or 0),
                    "page_key": page_key,
                    "parent_page_id": page_key,
                    "source": source,
                    "text": doc.text or "",
                    "image_path": str(doc.image_path) if doc.image_path else None,
                    **document_metadata,
                }
            )
            pages.append(page_record)
            for chunk_idx, chunk in enumerate(self._expanded_qa_chunks(doc.text or "")):
                snippet_record = self._augment_record_metadata(
                    {
                        "record_id": f"{page_key}::c{chunk_idx + 1:03d}",
                        "record_type": "snippet",
                        "doc_id": doc_id,
                        "page_number": int(doc.page_number or 0),
                        "page_key": page_key,
                        "parent_page_id": page_key,
                        "source": source,
                        "text": chunk,
                        "image_path": str(doc.image_path) if doc.image_path else None,
                        **document_metadata,
                    }
                )
                snippets.append(self._inherit_page_structure(snippet_record, page_record))
        return {"pages": pages, "snippets": snippets}

    def _attach_image_paths_from_documents(
        self,
        data_folder: Path,
        page_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        documents = self._load_documents(data_folder, render_pdf_pages=True)
        image_by_source = {
            str(doc.metadata.get("source", doc.doc_id)): str(doc.image_path)
            for doc in documents
            if doc.image_path
        }
        enriched = []
        for record in page_records:
            updated = dict(record)
            updated["image_path"] = image_by_source.get(str(record.get("source")), record.get("image_path"))
            enriched.append(updated)
        return enriched

    def _load_hybrid_corpus(
        self,
        data_folder: Path,
        include_images: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = f"{Path(data_folder).resolve()}::{self.config.corpus_source}::{include_images}"
        cached = self._hybrid_corpus_cache.get(cache_key)
        if cached:
            return cached

        source_mode = self.config.corpus_source
        if source_mode == "curated" or (source_mode == "auto" and self._has_curated_corpus()):
            corpus = self._load_curated_corpus()
            pages = [dict(record) for record in corpus["pages"]]
            snippets = [dict(record) for record in corpus["snippets"]]
            if include_images and self.config.enable_image_signal:
                pages = self._attach_image_paths_from_documents(data_folder, pages)
            corpus = {"pages": pages, "snippets": snippets}
        else:
            corpus = self._load_document_corpus(data_folder, include_images=include_images)
        self._hybrid_corpus_cache[cache_key] = corpus
        return corpus

    def build_index(self, data_folder: Path) -> None:
        corpus = self._load_hybrid_corpus(data_folder, include_images=self.config.enable_image_signal)
        paths = self._index_paths()
        paths["snippet_dense"].parent.mkdir(parents=True, exist_ok=True)
        table_records = build_table_records(corpus["pages"])
        graph_edges = build_document_graph(corpus["pages"], table_records)

        snippet_dense = self.retriever.embed_texts([record["text"] for record in corpus["snippets"]])
        page_dense = self.retriever.embed_texts([record["text"] for record in corpus["pages"]])
        np.save(paths["snippet_dense"], snippet_dense)
        np.save(paths["page_dense"], page_dense)
        self.sparse_retriever.build_index([record["text"] for record in corpus["snippets"]], paths["snippet_sparse"])
        self.sparse_retriever.build_index([record["text"] for record in corpus["pages"]], paths["page_sparse"])
        self.sparse_retriever.build_index([serialize_table_record(record) for record in table_records], paths["table_sparse"])
        self._write_jsonl(corpus["snippets"], paths["snippet_meta"])
        self._write_jsonl(corpus["pages"], paths["page_meta"])
        self._write_jsonl(table_records, paths["table_meta"])
        self._write_jsonl(graph_edges, paths["graph_meta"])

        # Preserve the legacy page-dense index path for older scripts that still read these files.
        np.save(self.config.index_path, page_dense)
        self.config.metadata_path.write_text(
            json.dumps(
                [
                    {
                        "source": record["source"],
                        "path": record["doc_id"],
                        "page": record["page_number"],
                    }
                    for record in corpus["pages"]
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if self.config.enable_image_signal:
            image_embeddings, image_pages = build_lightweight_page_image_embeddings(corpus["pages"])
            np.save(paths["page_image"], image_embeddings)
            self._write_jsonl(image_pages, paths["page_image_meta"])
        self._index_cache.clear()

    def _ensure_indices(self, data_folder: Path) -> Dict[str, Any]:
        paths = self._index_paths()
        required = [
            paths["snippet_dense"],
            paths["snippet_sparse"],
            paths["page_dense"],
            paths["page_sparse"],
            paths["snippet_meta"],
            paths["page_meta"],
            paths["table_sparse"],
            paths["table_meta"],
            paths["graph_meta"],
        ]
        if not all(path.exists() for path in required):
            self.build_index(data_folder)

        cache_key = str(Path(self.config.index_dir).resolve())
        cached = self._index_cache.get(cache_key)
        if cached:
            return cached

        indices: Dict[str, Any] = {
            "snippet_dense": load_index(paths["snippet_dense"]),
            "snippet_sparse": load_sparse_index(paths["snippet_sparse"]),
            "snippet_meta": self._read_jsonl(paths["snippet_meta"]),
            "page_dense": load_index(paths["page_dense"]),
            "page_sparse": load_sparse_index(paths["page_sparse"]),
            "page_meta": self._read_jsonl(paths["page_meta"]),
            "table_sparse": load_sparse_index(paths["table_sparse"]),
            "table_meta": self._read_jsonl(paths["table_meta"]),
            "graph_edges": self._read_jsonl(paths["graph_meta"]),
        }
        indices["graph_adjacency"] = build_graph_adjacency(indices["graph_edges"])
        if paths["page_image"].exists() and paths["page_image_meta"].exists():
            indices["page_image"] = np.load(paths["page_image"])
            indices["page_image_meta"] = self._read_jsonl(paths["page_image_meta"])
        self._index_cache[cache_key] = indices
        return indices

    @staticmethod
    def _metadata_match_score(record: Dict[str, Any], understanding: QueryUnderstanding) -> float:
        score = 0.0
        coverage_tags = set(record.get("coverage_tags", []) or [])
        target_coverages = set(understanding.target_coverages or [])
        if target_coverages and coverage_tags:
            overlap = len(target_coverages & coverage_tags)
            if overlap:
                score += 0.30 + (0.10 * overlap)

        document_type = str(record.get("document_type", ""))
        if understanding.preferred_document_types and document_type in set(understanding.preferred_document_types):
            score += 0.24

        clause_types = set(record.get("clause_types", []) or [])
        primary_clause_type = str(record.get("primary_clause_type", ""))
        if understanding.preferred_clause_types:
            preferred_clause_types = set(understanding.preferred_clause_types)
            if primary_clause_type in preferred_clause_types or clause_types & preferred_clause_types:
                score += 0.22

        field_type = str(record.get("field_type", ""))
        if understanding.preferred_field_types and field_type in set(understanding.preferred_field_types):
            score += 0.22

        if understanding.preferred_sections:
            section_titles = " ".join(record.get("section_titles", []) or [])
            section_anchor = str(record.get("section_anchor", "") or "")
            section_haystack = f"{section_titles} {section_anchor}".lower()
            for preferred_section in understanding.preferred_sections:
                preferred_lower = preferred_section.lower()
                if preferred_lower and preferred_lower in section_haystack:
                    score += 0.18
                    break

        if understanding.needs_declarations and document_type == "base_policy" and field_type in {"premium", "deductible"}:
            score -= 0.05
        return round(score, 6)

    def _prioritize_ranked_rows(
        self,
        rows: List[Dict[str, Any]],
        understanding: QueryUnderstanding,
        target_k: int,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        enriched = []
        for row in rows:
            metadata_match_score = self._metadata_match_score(row["record"], understanding)
            enriched.append({**row, "metadata_match_score": metadata_match_score})
        enriched.sort(
            key=lambda item: (
                float(item.get("metadata_match_score", 0.0)),
                float(item.get("raw_score", 0.0)),
            ),
            reverse=True,
        )
        keep = max(target_k, min(len(enriched), target_k + max(2, target_k // 2)))
        return enriched[:keep]

    def _merge_ranked_lists(
        self,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        rrf_k = max(1, int(self.config.rrf_k))
        for list_name, rows in ranked_lists.items():
            for rank, row in enumerate(rows, start=1):
                record_id = str(row["record"]["record_id"])
                merged_row = merged.setdefault(
                    record_id,
                    {
                        **row["record"],
                        "rrf_score": 0.0,
                        "retrieval_score": 0.0,
                        "metadata_match_score": 0.0,
                        "dense_rank": None,
                        "sparse_rank": None,
                        "page_rank": None,
                        "snippet_rank": None,
                    },
                )
                merged_row["rrf_score"] += 1.0 / (rrf_k + rank)
                merged_row["retrieval_score"] = max(float(merged_row["retrieval_score"]), float(row.get("raw_score", 0.0)))
                merged_row["metadata_match_score"] = max(
                    float(merged_row["metadata_match_score"]),
                    float(row.get("metadata_match_score", 0.0)),
                )
                if "dense" in list_name:
                    merged_row["dense_rank"] = min(rank, merged_row["dense_rank"] or rank)
                if "sparse" in list_name:
                    merged_row["sparse_rank"] = min(rank, merged_row["sparse_rank"] or rank)
                if "page" in list_name:
                    merged_row["page_rank"] = min(rank, merged_row["page_rank"] or rank)
                if "snippet" in list_name:
                    merged_row["snippet_rank"] = min(rank, merged_row["snippet_rank"] or rank)
        merged_rows = list(merged.values())
        merged_rows.sort(key=lambda item: float(item.get("rrf_score", 0.0)), reverse=True)
        return merged_rows[: max(top_k, self.config.candidate_pool_size)]

    def _retrieve_ranked_lists(
        self,
        question: str,
        data_folder: Path,
        understanding: QueryUnderstanding,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        target_k = max(top_k or self.config.max_retrievals, self.config.page_top_k)
        indices = self._ensure_indices(data_folder)
        mode = self.config.retrieval_mode
        ranked_lists: Dict[str, List[Dict[str, Any]]] = {}

        if mode in {"hybrid_multimodal", "hybrid_text", "dense_only", "visual"}:
            snippet_rows = self.retriever.search(question, indices["snippet_dense"], top_k=self.config.snippet_top_k, return_scores=True)
            page_rows = self.retriever.search(question, indices["page_dense"], top_k=self.config.page_top_k, return_scores=True)
            ranked_lists["snippet_dense"] = self._prioritize_ranked_rows(
                [{"record": indices["snippet_meta"][idx], "raw_score": score} for idx, score in snippet_rows],
                understanding,
                target_k=self.config.snippet_top_k,
            )
            ranked_lists["page_dense"] = self._prioritize_ranked_rows(
                [{"record": indices["page_meta"][idx], "raw_score": score} for idx, score in page_rows],
                understanding,
                target_k=target_k,
            )
        if mode in {"hybrid_multimodal", "hybrid_text", "sparse_only"}:
            snippet_rows = self.sparse_retriever.search(question, indices["snippet_sparse"], top_k=self.config.snippet_top_k, return_scores=True)
            page_rows = self.sparse_retriever.search(question, indices["page_sparse"], top_k=self.config.page_top_k, return_scores=True)
            ranked_lists["snippet_sparse"] = self._prioritize_ranked_rows(
                [{"record": indices["snippet_meta"][idx], "raw_score": score} for idx, score in snippet_rows],
                understanding,
                target_k=self.config.snippet_top_k,
            )
            ranked_lists["page_sparse"] = self._prioritize_ranked_rows(
                [{"record": indices["page_meta"][idx], "raw_score": score} for idx, score in page_rows],
                understanding,
                target_k=target_k,
            )
        if understanding.needs_table_lookup:
            table_rows = self.sparse_retriever.search(question, indices["table_sparse"], top_k=max(6, self.config.page_top_k), return_scores=True)
            ranked_lists["table_sparse"] = self._prioritize_ranked_rows(
                [{"record": indices["table_meta"][idx], "raw_score": score} for idx, score in table_rows],
                understanding,
                target_k=max(6, self.config.page_top_k),
            )
        return ranked_lists

    def retrieve_text_candidates(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.max_retrievals
        candidate_pool = max(top_k, self.config.candidate_pool_size)
        understanding = understand_query(question)
        ranked_lists = self._retrieve_ranked_lists(question, data_folder, understanding, top_k=top_k)
        text_only_lists = {
            key: value
            for key, value in ranked_lists.items()
            if not key.startswith("table_")
        }
        return self._merge_ranked_lists(text_only_lists, candidate_pool)

    def retrieve_image_candidates(
        self,
        question: str,
        data_folder: Path,
        page_sources: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        if not self.config.enable_image_signal or self.config.retrieval_mode in {"dense_only", "sparse_only"}:
            return {}
        indices = self._ensure_indices(data_folder)
        embeddings = indices.get("page_image")
        page_meta = indices.get("page_image_meta")
        if embeddings is None or page_meta is None:
            return {}
        scores = score_lightweight_page_image_query(question, embeddings)
        score_map: Dict[str, float] = {}
        for idx, page in enumerate(page_meta):
            source = str(page.get("source") or "")
            if page_sources and source not in page_sources:
                continue
            score_map[source] = float(scores[idx])
        return score_map

    def _add_graph_expansion_candidates(
        self,
        merged_candidates: List[Dict[str, Any]],
        indices: Dict[str, Any],
        understanding: QueryUnderstanding,
    ) -> List[Dict[str, Any]]:
        if not understanding.needs_graph_expansion:
            return merged_candidates

        page_by_key = {str(record.get("page_key")): record for record in indices["page_meta"]}
        seed_page_keys = {str(candidate.get("page_key") or candidate.get("parent_page_id")) for candidate in merged_candidates}
        expansions = expand_candidate_page_keys(
            seed_page_keys=seed_page_keys,
            adjacency=indices.get("graph_adjacency", {}),
            needs_endorsement_check=understanding.needs_endorsement_check,
            needs_declarations=understanding.needs_declarations,
            needs_definition=understanding.needs_definition,
            needs_exclusion_review=understanding.needs_exclusion_review,
        )
        if not expansions:
            return merged_candidates

        existing_ids = {str(candidate.get("record_id")) for candidate in merged_candidates}
        for expansion in expansions:
            page_key = str(expansion.get("page_key"))
            page_record = page_by_key.get(page_key)
            if not page_record:
                continue
            record_id = f"{page_key}::graph"
            if record_id in existing_ids:
                continue
            existing_ids.add(record_id)
            merged_candidates.append(
                {
                    **page_record,
                    "record_id": record_id,
                    "record_type": "graph_page",
                    "rrf_score": 0.08,
                    "retrieval_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "page_rank": None,
                    "snippet_rank": None,
                    "graph_relation": expansion.get("relation"),
                    "graph_source_page_key": expansion.get("source_page_key"),
                    "graph_confidence": expansion.get("confidence"),
                    "graph_reason": expansion.get("reason"),
                    "graph_shared_coverages": expansion.get("shared_coverages", []),
                    "graph_shared_sections": expansion.get("shared_sections", []),
                    "graph_source_section_title": expansion.get("source_section_title"),
                    "graph_target_section_title": expansion.get("target_section_title"),
                    "graph_source_form_codes": expansion.get("source_form_codes", []),
                }
            )
        return merged_candidates

    def merge_candidates(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.max_retrievals
        candidate_pool = max(top_k, self.config.candidate_pool_size)
        understanding = understand_query(question)
        indices = self._ensure_indices(data_folder)
        ranked_lists = self._retrieve_ranked_lists(question, data_folder, understanding, top_k=top_k)
        merged = self._merge_ranked_lists(ranked_lists, candidate_pool)
        return self._add_graph_expansion_candidates(merged, indices, understanding)

    def rerank_multimodal_candidates(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        understanding: Optional[QueryUnderstanding] = None,
        image_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        understanding = understanding or understand_query(question)
        page_type_counts: Dict[str, Set[str]] = defaultdict(set)
        question_key_terms = self._key_terms(question)
        for candidate in candidates:
            page_type_counts[str(candidate.get("source", ""))].add(str(candidate.get("record_type", "")))

        reranked = []
        for candidate in candidates:
            text = str(candidate.get("text", ""))
            source = str(candidate.get("source", ""))
            page_number = candidate.get("page_number")
            dense_rank = candidate.get("dense_rank")
            sparse_rank = candidate.get("sparse_rank")
            rrf_score = float(candidate.get("rrf_score", 0.0))
            metadata_match_score = float(candidate.get("metadata_match_score", 0.0))
            rule_score = self._score_insurance_evidence(question, text, source, page_number)
            overlap_score = 0.02 * len(question_key_terms & self._terms(text, min_len=3))
            dense_prior = 0.12 / (1 + dense_rank) if dense_rank else 0.0
            sparse_prior = 0.10 / (1 + sparse_rank) if sparse_rank else 0.0
            image_score = float((image_scores or {}).get(source, 0.0))
            consistency_boost = 0.05 if page_type_counts[source] >= {"page", "snippet"} else 0.0
            if candidate.get("record_type") == "snippet" and candidate.get("snippet_rank"):
                consistency_boost += 0.02
            document_type = str(candidate.get("document_type", ""))
            clause_types = set(candidate.get("clause_types", []) or [])
            field_type = str(candidate.get("field_type", ""))
            coverage_tags = set(candidate.get("coverage_tags", []) or [])
            target_coverage_overlap = len(coverage_tags & set(understanding.target_coverages or []))
            insurance_logic_boost = 0.04 * target_coverage_overlap
            if understanding.preferred_sections:
                section_haystack = " ".join(candidate.get("section_titles", []) or []).lower()
                if any(preferred.lower() in section_haystack for preferred in understanding.preferred_sections):
                    insurance_logic_boost += 0.14
            if understanding.needs_declarations and document_type == "declarations":
                insurance_logic_boost += 0.20
            if understanding.needs_limit and field_type in {"limit", "deductible", "premium"}:
                insurance_logic_boost += 0.18
            if understanding.needs_endorsement_check and (document_type == "endorsement" or "endorsement" in clause_types):
                insurance_logic_boost += 0.18
            if understanding.needs_definition and "definition" in clause_types:
                insurance_logic_boost += 0.14
            if understanding.needs_exclusion_review and "exclusion" in clause_types:
                insurance_logic_boost += 0.14
            graph_relation = str(candidate.get("graph_relation", ""))
            graph_confidence = float(candidate.get("graph_confidence") or 0.0)
            if graph_relation.endswith("overridden_by") or graph_relation == "overridden_by":
                insurance_logic_boost += 0.10
            if graph_relation.endswith("defines_limit_for") or graph_relation == "defines_limit_for":
                insurance_logic_boost += 0.08
            if graph_relation.endswith("qualified_by") or graph_relation == "qualified_by":
                insurance_logic_boost += 0.08
            if graph_relation.endswith("limited_by") or graph_relation == "limited_by":
                insurance_logic_boost += 0.06
            insurance_logic_boost += min(0.10, graph_confidence * 0.08)
            final_score = (
                rrf_score
                + metadata_match_score
                + dense_prior
                + sparse_prior
                + rule_score
                + overlap_score
                + consistency_boost
                + insurance_logic_boost
                + (self.config.image_signal_weight * image_score)
            )
            reranked.append(
                {
                    **candidate,
                    "score": round(final_score, 6),
                    "rerank_score": round(final_score - float(candidate.get("retrieval_score", 0.0)), 6),
                    "image_score": round(image_score, 6),
                }
            )
        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return reranked

    def rollup_candidates_to_pages(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        top_k = top_k or self.config.max_retrievals
        grouped: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            source = str(candidate.get("source", ""))
            page_entry = grouped.setdefault(
                source,
                {
                    "source": source,
                    "page_number": candidate.get("page_number"),
                    "score": float(candidate.get("score", 0.0)),
                    "retrieval_score": float(candidate.get("retrieval_score", 0.0)),
                    "rerank_score": float(candidate.get("rerank_score", 0.0)),
                    "image_score": float(candidate.get("image_score", 0.0)),
                    "dense_rank": candidate.get("dense_rank"),
                    "sparse_rank": candidate.get("sparse_rank"),
                    "document_type": candidate.get("document_type"),
                    "document_role": candidate.get("document_role"),
                    "packet_id": candidate.get("packet_id"),
                    "document_priority": candidate.get("document_priority"),
                    "primary_clause_type": candidate.get("primary_clause_type"),
                    "coverage_tags": set(candidate.get("coverage_tags", []) or []),
                    "section_titles": list(candidate.get("section_titles", []) or []),
                    "section_path": list(candidate.get("section_path", []) or []),
                    "section_anchor": candidate.get("section_anchor"),
                    "form_codes": set(candidate.get("form_codes", []) or []),
                    "source_origin": candidate.get("source_origin"),
                    "source_name": candidate.get("source_name"),
                    "source_url": candidate.get("source_url"),
                    "source_authority": candidate.get("source_authority") or candidate.get("authority"),
                    "graph_relations": set(),
                    "graph_details": [],
                    "record_types": set(),
                    "snippets": [],
                    "table_fields": [],
                    "page_text": "",
                },
            )
            page_entry["score"] = max(page_entry["score"], float(candidate.get("score", 0.0)))
            page_entry["retrieval_score"] = max(page_entry["retrieval_score"], float(candidate.get("retrieval_score", 0.0)))
            page_entry["rerank_score"] = max(page_entry["rerank_score"], float(candidate.get("rerank_score", 0.0)))
            page_entry["image_score"] = max(page_entry["image_score"], float(candidate.get("image_score", 0.0)))
            if candidate.get("dense_rank") is not None:
                page_entry["dense_rank"] = min(candidate.get("dense_rank"), page_entry["dense_rank"] or candidate.get("dense_rank"))
            if candidate.get("sparse_rank") is not None:
                page_entry["sparse_rank"] = min(candidate.get("sparse_rank"), page_entry["sparse_rank"] or candidate.get("sparse_rank"))
            page_entry["record_types"].add(str(candidate.get("record_type", "")))
            if candidate.get("record_type") == "page":
                page_entry["page_text"] = str(candidate.get("text", ""))
            if candidate.get("text"):
                page_entry["snippets"].append((float(candidate.get("score", 0.0)), str(candidate.get("text", ""))))
            page_entry["coverage_tags"].update(candidate.get("coverage_tags", []) or [])
            page_entry["form_codes"].update(candidate.get("form_codes", []) or [])
            if candidate.get("field_type"):
                page_entry["table_fields"].append(
                    {
                        "field_name": candidate.get("field_name"),
                        "field_value": candidate.get("field_value"),
                        "normalized_field_name": candidate.get("normalized_field_name"),
                        "normalized_field_value": candidate.get("normalized_field_value"),
                        "field_type": candidate.get("field_type"),
                        "numeric_value": candidate.get("numeric_value"),
                        "value_unit": candidate.get("value_unit"),
                        "coverage_tags": list(candidate.get("coverage_tags", []) or []),
                    }
                )
            if candidate.get("graph_relation"):
                page_entry["graph_relations"].add(str(candidate.get("graph_relation")))
                page_entry["graph_details"].append(
                    {
                        "relation": candidate.get("graph_relation"),
                        "confidence": candidate.get("graph_confidence"),
                        "reason": candidate.get("graph_reason"),
                        "shared_coverages": candidate.get("graph_shared_coverages", []),
                        "shared_sections": candidate.get("graph_shared_sections", []),
                        "source_page_key": candidate.get("graph_source_page_key"),
                        "source_section_title": candidate.get("graph_source_section_title"),
                        "target_section_title": candidate.get("graph_target_section_title"),
                        "source_form_codes": candidate.get("graph_source_form_codes", []),
                    }
                )

        ranked_pages: List[Dict[str, object]] = []
        for page_entry in grouped.values():
            snippets = []
            seen_texts = set()
            for _, snippet in sorted(page_entry["snippets"], key=lambda item: item[0], reverse=True):
                if snippet in seen_texts:
                    continue
                seen_texts.add(snippet)
                snippets.append(snippet)
                if len(snippets) >= 3:
                    break
            support_text = " ".join(snippets) or page_entry["page_text"]
            score = float(page_entry["score"])
            if page_entry["record_types"] >= {"page", "snippet"}:
                score += 0.04
            ranked_pages.append(
                {
                    "source": page_entry["source"],
                    "score": round(score, 6),
                    "retrieval_score": round(float(page_entry["retrieval_score"]), 6),
                    "rerank_score": round(float(page_entry["rerank_score"]), 6),
                    "page_number": page_entry["page_number"],
                    "text_snippet": self._select_evidence_snippet(question, support_text, max_chars=self.config.max_page_chars),
                    "snippet_support": snippets,
                    "image_score": round(float(page_entry["image_score"]), 6),
                    "dense_rank": page_entry["dense_rank"],
                    "sparse_rank": page_entry["sparse_rank"],
                    "document_type": page_entry["document_type"],
                    "document_role": page_entry["document_role"],
                    "packet_id": page_entry["packet_id"],
                    "document_priority": page_entry["document_priority"],
                    "primary_clause_type": page_entry["primary_clause_type"],
                    "coverage_tags": sorted(page_entry["coverage_tags"]),
                    "section_titles": page_entry["section_titles"][:3],
                    "section_path": page_entry["section_path"][:3],
                    "section_anchor": page_entry["section_anchor"],
                    "form_codes": sorted(page_entry["form_codes"]),
                    "source_origin": page_entry["source_origin"],
                    "source_name": page_entry["source_name"],
                    "source_url": page_entry["source_url"],
                    "source_authority": page_entry["source_authority"],
                    "table_fields": page_entry["table_fields"][:3],
                    "graph_relations": sorted(page_entry["graph_relations"]),
                    "graph_details": page_entry["graph_details"][:3],
                }
            )
        ranked_pages.sort(key=lambda page: float(page.get("score", 0.0)), reverse=True)
        return ranked_pages[:top_k]

    def _page_order_bucket(self, page: Dict[str, object], understanding: QueryUnderstanding) -> int:
        document_type = str(page.get("document_type", ""))
        primary_clause_type = str(page.get("primary_clause_type", ""))
        if understanding.needs_declarations and document_type == "declarations":
            return 0
        if page.get("table_fields") and understanding.needs_table_lookup:
            return 1
        if primary_clause_type == "coverage":
            return 2
        if primary_clause_type == "exclusion":
            return 3
        if primary_clause_type == "exception":
            return 4
        if document_type == "endorsement" or primary_clause_type == "endorsement":
            return 5
        if primary_clause_type == "definition":
            return 6
        return 7

    def pack_long_context(
        self,
        ranked_pages: List[Dict[str, object]],
        answer_top_k: int,
        understanding: Optional[QueryUnderstanding] = None,
    ) -> str:
        understanding = understanding or QueryUnderstanding(
            intent="document_qa",
            target_coverages=[],
            needs_limit=False,
            needs_endorsement_check=False,
            needs_table_lookup=False,
            needs_definition=False,
            needs_exclusion_review=False,
            needs_declarations=False,
            needs_graph_expansion=False,
            preferred_document_types=[],
            preferred_clause_types=[],
            preferred_field_types=[],
            preferred_sections=[],
        )
        context_parts = []
        ordered_pages = sorted(
            ranked_pages,
            key=lambda page: (self._page_order_bucket(page, understanding), -float(page.get("score", 0.0))),
        )
        for page in ordered_pages[:answer_top_k]:
            snippets = page.get("snippet_support") or [page.get("text_snippet", "")]
            table_lines = []
            for field in page.get("table_fields", []) or []:
                field_name = str(field.get("normalized_field_name") or field.get("field_name") or "").strip()
                field_value = str(field.get("normalized_field_value") or field.get("field_value") or "").strip()
                if field_name or field_value:
                    table_lines.append(f"- TABLE: {field_name}: {field_value}".strip(": "))
            bullet_text = "\n".join(f"- {snippet}" for snippet in snippets[:3] if snippet)
            role_header = f"ROLE: {page.get('document_type', 'page')} / {page.get('primary_clause_type', 'general')}"
            block_parts = [f"SOURCE: {page['source']}", role_header]
            if page.get("section_anchor"):
                block_parts.append(f"SECTION: {page.get('section_anchor')}")
            if table_lines:
                block_parts.extend(table_lines[:2])
            if bullet_text:
                block_parts.append(bullet_text)
            context_parts.append("\n".join(block_parts) + "\n")
        return self._trim_context("\n---\n".join(context_parts), max_chars=self.config.max_context_chars)

    def _tool_router_stub(self, question: str, ranked_pages: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "tool_decision": "none",
            "tool_name": None,
            "reason": "tool_use_stub_v1",
            "question": question,
            "candidate_count": len(ranked_pages),
        }

    def query(self, question: str, data_folder: Path, top_k: Optional[int] = None) -> str:
        return str(self.query_with_ranking(question, data_folder, top_k=top_k)["answer"])

    def query_with_ranking(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
        force_extractive: bool = False,
    ) -> Dict[str, object]:
        top_k = top_k or self.config.max_retrievals
        answer_top_k = min(top_k, getattr(self.config, "max_answer_pages", top_k))
        understanding = understand_query(question)
        merged_candidates = self.merge_candidates(question, data_folder, top_k=top_k)
        if not merged_candidates:
            return {
                "answer": "I cannot support an answer from the retrieved evidence. SOURCE: insufficient_evidence",
                "source_ranking": [],
                "tool_router": self._tool_router_stub(question, []),
                "query_understanding": understanding.to_dict(),
            }
        page_sources = {str(candidate.get("source", "")) for candidate in merged_candidates}
        image_scores = self.retrieve_image_candidates(question, data_folder, page_sources=page_sources)
        reranked = self.rerank_multimodal_candidates(question, merged_candidates, understanding=understanding, image_scores=image_scores)
        ranked_pages = self.rollup_candidates_to_pages(question, reranked, top_k=top_k)
        combined_context = self.pack_long_context(ranked_pages, answer_top_k, understanding=understanding)
        prompt = format_prompt(combined_context, question, self.config.prompt_template)
        answer = self.vlm_client.generate_extractive(prompt) if force_extractive else self.vlm_client.generate(prompt)
        return {
            "answer": answer,
            "source_ranking": ranked_pages,
            "tool_router": self._tool_router_stub(question, ranked_pages),
            "query_understanding": understanding.to_dict(),
        }

    @staticmethod
    def _first_amount(text: str) -> Optional[str]:
        match = _AMOUNT_RE.search(text or "")
        return match.group(0) if match else None

    def _extract_structured_limit(
        self,
        question: str,
        cited_page: Optional[Dict[str, object]],
    ) -> Optional[str]:
        if not cited_page or not self._question_requires_numeric_evidence(question):
            return None
        lowered = question.lower()
        preferred_field_type = None
        if "deductible" in lowered:
            preferred_field_type = "deductible"
        elif "premium" in lowered:
            preferred_field_type = "premium"
        elif any(term in lowered for term in ["limit", "limits", "sublimit", "retention", "coinsurance"]):
            preferred_field_type = "limit"

        target_coverages = set(understand_query(question).target_coverages or [])
        candidate_fields = list(cited_page.get("table_fields", []) or [])
        if target_coverages:
            targeted_fields = [
                field for field in candidate_fields
                if target_coverages & set(field.get("coverage_tags", []) or [])
            ]
            if targeted_fields:
                candidate_fields = targeted_fields

        for field in candidate_fields:
            field_type = str(field.get("field_type", ""))
            if preferred_field_type and field_type != preferred_field_type:
                continue
            field_value = str(field.get("normalized_field_value") or field.get("field_value") or "").strip()
            if field_value:
                return field_value
        for field in candidate_fields:
            field_value = str(field.get("normalized_field_value") or field.get("field_value") or "").strip()
            if field_value:
                return field_value
        return self._first_amount(str(cited_page.get("text_snippet", "")))

    def _extract_structured_coverage(
        self,
        understanding: QueryUnderstanding,
        cited_page: Optional[Dict[str, object]],
    ) -> Optional[object]:
        coverage_tags = list(understanding.target_coverages or [])
        if not coverage_tags and cited_page:
            for field in cited_page.get("table_fields", []) or []:
                field_coverages = list(field.get("coverage_tags", []) or [])
                if field_coverages:
                    coverage_tags = field_coverages
                    break
        if not coverage_tags and cited_page:
            coverage_tags = list(cited_page.get("coverage_tags", []) or [])
        coverage_labels = normalize_coverage_labels(coverage_tags)
        if not coverage_labels:
            return None
        return coverage_labels[0] if len(coverage_labels) == 1 else coverage_labels

    @staticmethod
    def _shared_coverage_labels(pages: List[Dict[str, object]]) -> List[str]:
        if not pages:
            return []
        shared = set(pages[0].get("coverage_tags", []) or [])
        for page in pages[1:]:
            shared &= set(page.get("coverage_tags", []) or [])
        return normalize_coverage_labels(sorted(shared))

    @staticmethod
    def _page_sections(page: Dict[str, object]) -> List[str]:
        return list(page.get("section_path", []) or page.get("section_titles", []) or [])

    @staticmethod
    def _question_mentions_exception(question: str) -> bool:
        lowered = (question or "").lower()
        return any(term in lowered for term in ["exception", "except", "does not apply", "carve out"])

    def _resolve_conflicts(
        self,
        question: str,
        understanding: QueryUnderstanding,
        ranked_pages: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        needs_exception_review = self._question_mentions_exception(question)
        top_pages = ranked_pages[:5]
        declarations_pages = [page for page in top_pages if page.get("document_type") == "declarations"]
        endorsement_pages = [
            page for page in top_pages
            if page.get("document_type") == "endorsement" or page.get("primary_clause_type") == "endorsement"
        ]
        exclusion_pages = [page for page in top_pages if page.get("primary_clause_type") == "exclusion"]
        exception_pages = [page for page in top_pages if page.get("primary_clause_type") == "exception"]
        conflicts: List[Dict[str, object]] = []

        if understanding.needs_endorsement_check and endorsement_pages and exclusion_pages:
            shared_labels = self._shared_coverage_labels([endorsement_pages[0], exclusion_pages[0]])
            shared_sections = sorted(set(self._page_sections(endorsement_pages[0])) & set(self._page_sections(exclusion_pages[0])))
            conflicts.append(
                {
                    "type": "endorsement_override",
                    "status": "possible_override",
                    "severity": "review",
                    "message": (
                        f"Base-policy exclusion and endorsement evidence were both retrieved for {', '.join(shared_labels)}."
                        if shared_labels else
                        "Base-policy exclusion and endorsement evidence were both retrieved and should be reconciled."
                    ),
                    "sources": [exclusion_pages[0].get("source"), endorsement_pages[0].get("source")],
                    "coverages": shared_labels,
                    "sections": shared_sections,
                    "endorsement_forms": endorsement_pages[0].get("form_codes", []),
                }
            )
        elif understanding.needs_endorsement_check and exclusion_pages and not endorsement_pages:
            conflicts.append(
                {
                    "type": "missing_endorsement_evidence",
                    "status": "insufficient_counterevidence",
                    "severity": "blocking",
                    "message": "Exclusion evidence was retrieved without a matching endorsement page; final coverage should not be decided from the exclusion alone.",
                    "sources": [exclusion_pages[0].get("source")],
                    "coverages": normalize_coverage_labels(exclusion_pages[0].get("coverage_tags", []) or []),
                    "sections": self._page_sections(exclusion_pages[0]),
                    "endorsement_forms": [],
                }
            )

        if needs_exception_review and exclusion_pages and exception_pages:
            shared_labels = self._shared_coverage_labels([exception_pages[0], exclusion_pages[0]])
            shared_sections = sorted(set(self._page_sections(exception_pages[0])) & set(self._page_sections(exclusion_pages[0])))
            conflicts.append(
                {
                    "type": "exception_qualifies_exclusion",
                    "status": "possible_exception_carveout",
                    "severity": "review",
                    "message": "Exclusion and exception evidence were both retrieved and should be read together before deciding coverage.",
                    "sources": [exclusion_pages[0].get("source"), exception_pages[0].get("source")],
                    "coverages": shared_labels,
                    "sections": shared_sections,
                    "endorsement_forms": exception_pages[0].get("form_codes", []),
                }
            )
        elif needs_exception_review and exclusion_pages and not exception_pages:
            conflicts.append(
                {
                    "type": "missing_exception_evidence",
                    "status": "insufficient_counterevidence",
                    "severity": "blocking",
                    "message": "Exclusion evidence was retrieved without a matching exception page; final coverage should not be decided from the exclusion alone.",
                    "sources": [exclusion_pages[0].get("source")],
                    "coverages": normalize_coverage_labels(exclusion_pages[0].get("coverage_tags", []) or []),
                    "sections": self._page_sections(exclusion_pages[0]),
                    "endorsement_forms": [],
                }
            )

        if understanding.needs_limit and declarations_pages and any(page.get("table_fields") for page in declarations_pages):
            non_declaration_numeric_pages = [
                page for page in top_pages
                if page.get("document_type") != "declarations" and page.get("table_fields")
            ]
            if non_declaration_numeric_pages:
                conflicts.append(
                    {
                        "type": "numeric_source_conflict",
                        "status": "review_numeric_source",
                        "severity": "review",
                        "message": "Numeric evidence appears on both declarations-style and non-declarations pages; review source context.",
                        "sources": [declarations_pages[0].get("source"), non_declaration_numeric_pages[0].get("source")],
                        "coverages": self._shared_coverage_labels([declarations_pages[0], non_declaration_numeric_pages[0]]),
                        "sections": sorted(set(self._page_sections(declarations_pages[0])) & set(self._page_sections(non_declaration_numeric_pages[0]))),
                        "endorsement_forms": non_declaration_numeric_pages[0].get("form_codes", []),
                    }
                )

        for page in top_pages:
            for detail in page.get("graph_details", []) or []:
                relation = str(detail.get("relation", ""))
                if "overridden_by" in relation:
                    conflicts.append(
                        {
                            "type": "graph_override_relation",
                            "status": "graph_supported_override",
                            "severity": "info",
                            "message": "Graph expansion identified endorsement override relationships in the retrieved evidence.",
                            "sources": [detail.get("source_page_key"), page.get("source")],
                            "coverages": normalize_coverage_labels(detail.get("shared_coverages", []) or []),
                            "sections": detail.get("shared_sections", []) or [],
                            "endorsement_forms": detail.get("source_form_codes", []) or [],
                        }
                    )
                if "qualified_by" in relation:
                    conflicts.append(
                        {
                            "type": "graph_exception_relation",
                            "status": "graph_supported_exception",
                            "severity": "info",
                            "message": "Graph expansion identified exception-to-exclusion relationships in the retrieved evidence.",
                            "sources": [detail.get("source_page_key"), page.get("source")],
                            "coverages": normalize_coverage_labels(detail.get("shared_coverages", []) or []),
                            "sections": detail.get("shared_sections", []) or [],
                            "endorsement_forms": detail.get("source_form_codes", []) or [],
                        }
                    )

        deduped: List[Dict[str, object]] = []
        seen = set()
        for conflict in conflicts:
            key = (
                conflict.get("type"),
                tuple(conflict.get("sources", []) or []),
                tuple(conflict.get("sections", []) or []),
                tuple(conflict.get("coverages", []) or []),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(conflict)
        return deduped

    def _resolve_conflict_notes(
        self,
        question: str,
        understanding: QueryUnderstanding,
        ranked_pages: List[Dict[str, object]],
    ) -> List[str]:
        return [str(conflict.get("message")) for conflict in self._resolve_conflicts(question, understanding, ranked_pages)]

    def _structured_answer_fields(
        self,
        question: str,
        understanding: QueryUnderstanding,
        cited_page: Optional[Dict[str, object]],
        ranked_pages: List[Dict[str, object]],
    ) -> Dict[str, object]:
        conflicts = self._resolve_conflicts(question, understanding, ranked_pages)
        blocking_conflicts = [conflict for conflict in conflicts if conflict.get("severity") == "blocking"]
        return {
            "coverage": self._extract_structured_coverage(understanding, cited_page),
            "limit": self._extract_structured_limit(question, cited_page),
            "evidence_role": (
                f"{cited_page.get('document_type', 'page')}:{cited_page.get('primary_clause_type', 'general')}"
                if cited_page else None
            ),
            "policy_logic_status": "blocked" if blocking_conflicts else "review" if conflicts else "clear",
            "conflict_notes": [str(conflict.get("message")) for conflict in conflicts],
            "conflicts": conflicts,
            "override_summary": next((conflict for conflict in conflicts if conflict.get("type") == "endorsement_override"), None),
        }

    def query_structured(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
        force_extractive: bool = False,
    ) -> Dict[str, object]:
        understanding = understand_query(question)
        result = self.query_with_ranking(question, data_folder, top_k=top_k, force_extractive=force_extractive)
        answer = str(result["answer"]).strip()
        ranked_pages = result["source_ranking"]
        citations = []
        cited_source = self._extract_answer_source(answer, ranked_pages)
        clean_answer = re.sub(r"\n+\s*SOURCE:\s*.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        cited_page = None
        if cited_source:
            cited_page = next((page for page in ranked_pages if page["source"] == cited_source), None)
        if cited_page is None:
            cited_page = self._choose_cited_page(question, clean_answer, ranked_pages)
        if cited_page:
            repaired_answer = self._repair_answer_from_evidence(question, clean_answer, cited_page)
            if not repaired_answer and ("insufficient_evidence" in answer.lower() or not clean_answer):
                repaired_answer = self._best_sentence_from_evidence(question, str(cited_page.get("text_snippet", "")))
            if repaired_answer:
                clean_answer = repaired_answer
            citations.append(
                {
                    "source": cited_page["source"],
                    "page_id": self._source_to_page_id(str(cited_page["source"])),
                    "evidence_text": cited_page.get("text_snippet", ""),
                    "document_type": cited_page.get("document_type"),
                    "document_role": cited_page.get("document_role"),
                    "packet_id": cited_page.get("packet_id"),
                    "primary_clause_type": cited_page.get("primary_clause_type"),
                    "section_anchor": cited_page.get("section_anchor"),
                    "form_codes": cited_page.get("form_codes", []),
                    "source_origin": cited_page.get("source_origin"),
                    "source_name": cited_page.get("source_name"),
                    "source_url": cited_page.get("source_url"),
                    "source_authority": cited_page.get("source_authority"),
                }
            )

        confidence = self._estimate_confidence(question, clean_answer, ranked_pages)
        supported, support_reason = self._citation_support_details(
            question,
            clean_answer,
            citations,
            min_overlap=getattr(self.config, "citation_min_overlap", 0.20),
        )
        structured_fields = self._structured_answer_fields(question, understanding, cited_page, ranked_pages)
        if not supported:
            confidence = min(confidence, 0.19)
        blocking_conflicts = [conflict for conflict in structured_fields.get("conflicts", []) if conflict.get("severity") == "blocking"]
        if blocking_conflicts:
            confidence = min(confidence, 0.15)
        threshold = getattr(self.config, "abstain_threshold", 0.20)
        abstain = confidence < threshold or "insufficient_evidence" in answer.lower() or not supported or bool(blocking_conflicts)
        caveats: List[str] = []
        if citations and citations[0].get("document_type") != "declarations" and self._question_requires_numeric_evidence(question):
            caveats.append("Numeric answer was not cited from a declarations-style page.")
        if ranked_pages and any("overridden_by" in relation for relation in ranked_pages[0].get("graph_relations", []) or []):
            caveats.append("Retrieved evidence includes endorsement override relationships that should be reviewed.")
        if ranked_pages and any("qualified_by" in relation for relation in ranked_pages[0].get("graph_relations", []) or []):
            caveats.append("Retrieved evidence includes exception relationships that should be reviewed with the underlying exclusion.")
        if blocking_conflicts:
            caveats.extend(str(conflict.get("message")) for conflict in blocking_conflicts)
        return {
            "answer": "" if abstain else clean_answer,
            "citations": [] if abstain else citations,
            "confidence": confidence,
            "abstain": abstain,
            "abstain_reason": "missing_policy_packet_counterevidence" if blocking_conflicts else "insufficient_retrieved_evidence" if abstain else None,
            "citation_support": supported,
            "citation_support_reason": support_reason,
            "source_ranking": ranked_pages,
            "tool_router": result.get("tool_router"),
            "query_understanding": result.get("query_understanding"),
            "caveats": caveats,
            "coverage": None if abstain else structured_fields.get("coverage"),
            "limit": None if abstain else structured_fields.get("limit"),
            "evidence_role": None if abstain else structured_fields.get("evidence_role"),
            "policy_logic_status": structured_fields.get("policy_logic_status"),
            "conflict_notes": structured_fields.get("conflict_notes", []),
            "conflicts": structured_fields.get("conflicts", []),
            "override_summary": structured_fields.get("override_summary"),
        }

    @staticmethod
    def _terms(text: str, min_len: int = 2) -> Set[str]:
        return {term for term in re.findall(r"[a-zA-Z0-9$%]+", text.lower()) if len(term) >= min_len}

    @staticmethod
    def _generic_terms() -> Set[str]:
        return {
            "what", "which", "does", "this", "that", "the", "policy", "coverage",
            "include", "includes", "provide", "provides", "listed", "insurance",
            "from", "your", "about", "page", "evidence", "mentioning", "explains",
            "summarize", "guidance", "consumer", "know", "passage", "related",
            "described", "find", "amount", "numeric", "detail", "stated", "for",
            "and", "after", "before", "with", "into", "under", "document", "guide",
            "pdf", "apply", "applies",
        }

    @classmethod
    def _key_terms(cls, text: str) -> Set[str]:
        return cls._terms(text, min_len=3) - cls._generic_terms()

    @staticmethod
    def _question_requires_numeric_evidence(question: str) -> bool:
        lowered = question.lower()
        numeric_intents = {
            "amount", "limit", "limits", "deductible", "premium", "sublimit",
            "coinsurance", "retention", "per person", "per accident", "per day",
            "maximum", "minimum", "how much", "dollar", "percent", "percentage",
        }
        return any(intent in lowered for intent in numeric_intents)

    @staticmethod
    def _insurance_field_groups(question: str) -> List[Set[str]]:
        lowered = question.lower()
        groups: List[Set[str]] = []
        if "liability" in lowered:
            groups.append({"liability"})
        if "comprehensive" in lowered:
            groups.append({"comprehensive"})
        if "collision" in lowered:
            groups.append({"collision"})
        if "deductible" in lowered:
            groups.append({"deductible"})
        if "limit" in lowered or "limits" in lowered or "sublimit" in lowered:
            groups.append({"limit", "limits", "sublimit"})
        if "premium" in lowered:
            groups.append({"premium"})
        if "medical payments" in lowered:
            groups.append({"medical", "payments"})
        if "endorsement" in lowered or "rider" in lowered:
            groups.append({"endorsement", "rider"})
        if "exclusion" in lowered or "excludes" in lowered:
            groups.append({"exclusion", "exclusions", "excludes"})
        if "duties" in lowered or "after a loss" in lowered:
            groups.append({"duties", "loss", "notify", "cooperate"})
        return groups

    @staticmethod
    def _candidate_sentences(text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=:)\s+|\n+", normalized) if s.strip()]

    @classmethod
    def _score_insurance_evidence(
        cls,
        question: str,
        text: str,
        source: object = "",
        page_number: object = None,
    ) -> float:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return 0.0
        lowered = cleaned.lower()
        text_terms = cls._terms(cleaned, min_len=2)
        question_terms = cls._key_terms(question)
        field_groups = cls._insurance_field_groups(question)
        requires_numeric = cls._question_requires_numeric_evidence(question)

        score = 0.025 * len(question_terms & text_terms)
        for group in field_groups:
            score += 0.12 if group & text_terms else -0.08

        if requires_numeric:
            has_amount = bool(_AMOUNT_RE.search(cleaned))
            score += 0.25 if has_amount else -0.25
            if ("declaration" in lowered or "declarations" in lowered) and has_amount:
                score += 0.18
            if re.search(r"(limit|deductible|premium|sublimit)\s*:", lowered):
                score += 0.20
            if page_number in {1, "1"} and has_amount:
                score += 0.05

        for sentence in cls._candidate_sentences(cleaned):
            sentence_terms = cls._terms(sentence, min_len=2)
            if requires_numeric and _AMOUNT_RE.search(sentence):
                covered_groups = sum(1 for group in field_groups if group & sentence_terms)
                if covered_groups:
                    score += 0.18 * covered_groups
                if question_terms and len(question_terms & sentence_terms) >= min(2, len(question_terms)):
                    score += 0.10

        return round(score, 6)

    @classmethod
    def _repair_answer_from_evidence(
        cls,
        question: str,
        answer: str,
        cited_page: Dict[str, object],
    ) -> Optional[str]:
        if not cls._question_requires_numeric_evidence(question):
            return None
        if set(_AMOUNT_RE.findall(answer or "")):
            return None
        evidence = str(cited_page.get("text_snippet", ""))
        field_groups = cls._insurance_field_groups(question)
        best_sentence = ""
        best_score = 0
        for sentence in cls._candidate_sentences(evidence):
            if not _AMOUNT_RE.search(sentence):
                continue
            sentence_terms = cls._terms(sentence, min_len=2)
            score = sum(1 for group in field_groups if group & sentence_terms)
            if "limit" in sentence_terms or "deductible" in sentence_terms:
                score += 1
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence if best_score > 0 else None

    @classmethod
    def _best_sentence_from_evidence(
        cls,
        question: str,
        evidence: str,
    ) -> Optional[str]:
        question_terms = cls._key_terms(question)
        if not question_terms:
            return None
        best_sentence = ""
        best_score = 0.0
        for sentence in cls._candidate_sentences(evidence):
            sentence_terms = cls._terms(sentence, min_len=3)
            overlap = question_terms & sentence_terms
            if not overlap:
                continue
            score = float(len(overlap))
            if _AMOUNT_RE.search(sentence):
                score += 1.0
            if len(sentence) < 40:
                score -= 0.5
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence if best_score >= 2.0 else None

    @classmethod
    def _citation_support_details(
        cls,
        question: str,
        answer: str,
        citations: List[Dict[str, object]],
        min_overlap: float = 0.20,
    ) -> tuple[bool, str]:
        if "insufficient_evidence" in (answer or "").lower():
            return False, "model_reported_insufficient_evidence"
        if not citations:
            return False, "missing_citation"
        evidence = " ".join(str(citation.get("evidence_text", "")) for citation in citations)
        if not evidence.strip():
            return False, "missing_citation_evidence"

        answer_amounts = set(_AMOUNT_RE.findall(answer or ""))
        evidence_amounts = set(_AMOUNT_RE.findall(evidence))
        if answer_amounts and not answer_amounts <= evidence_amounts:
            return False, "answer_amount_not_in_citation"
        if cls._question_requires_numeric_evidence(question) and not answer_amounts:
            return False, "numeric_question_without_answer_amount"

        key_terms = cls._key_terms(question)
        evidence_terms = cls._terms(evidence, min_len=3)
        if key_terms and not (key_terms & evidence_terms):
            return False, "question_terms_not_in_citation"
        broad_value_terms = {
            "coverage", "cover", "include", "policy", "insurance", "limit", "limits",
            "sublimit", "deductible", "endorsement", "reimbursement", "provision",
            "amount", "liability", "property", "loss", "use", "auto", "automobile",
            "guide", "document", "pdf", "apply", "applies",
        }
        specific_terms = {term for term in key_terms if len(term) >= 4 and term not in broad_value_terms}
        if specific_terms and not specific_terms <= evidence_terms:
            return False, "specific_question_terms_not_in_citation"

        answer_terms = cls._key_terms(answer)
        answer_specific_terms = {
            term for term in answer_terms
            if len(term) >= 4 and term not in broad_value_terms
        }
        if answer_specific_terms:
            overlap_ratio = len(answer_specific_terms & evidence_terms) / max(1, len(answer_specific_terms))
            if overlap_ratio < min_overlap:
                return False, "answer_terms_not_supported_by_citation"
        return True, "supported"

    @staticmethod
    def _extract_answer_source(answer: str, ranked_pages: Optional[List[Dict[str, object]]] = None) -> Optional[str]:
        match = re.search(r"SOURCE:\s*(.+)", answer, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        for page in ranked_pages or []:
            source = str(page.get("source", ""))
            if source and source in answer:
                return source
        return None

    @staticmethod
    def _choose_cited_page(
        question: str,
        answer: str,
        ranked_pages: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not ranked_pages:
            return None
        answer_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", answer.lower()) if len(term) > 2}
        question_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower()) if len(term) > 2}
        answer_amounts = set(_AMOUNT_RE.findall(answer))
        best_page = ranked_pages[0]
        best_score = -1.0
        for page in ranked_pages:
            text = str(page.get("text_snippet", ""))
            text_terms = set(re.findall(r"[a-zA-Z0-9$%]+", text.lower()))
            text_amounts = set(_AMOUNT_RE.findall(text))
            score = float(len(answer_terms & text_terms))
            score += 0.5 * len(question_terms & text_terms)
            score += 4.0 * len(answer_amounts & text_amounts)
            score += 0.1 * float(page.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_page = page
        return best_page

    @staticmethod
    def _estimate_confidence(question: str, answer: str, ranked_pages: List[Dict[str, object]]) -> float:
        if not ranked_pages:
            return 0.0
        top_score = max(0.0, min(1.0, float(ranked_pages[0].get("score", 0.0))))
        question_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower()) if len(term) > 2}
        answer_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", answer.lower()) if len(term) > 2}
        generic_terms = DocumentRetrievalPipeline._generic_terms()
        key_terms = question_terms - generic_terms
        evidence_terms = {
            term
            for page in ranked_pages[:3]
            for term in re.findall(r"[a-zA-Z0-9$%]+", str(page.get("text_snippet", "")).lower())
            if len(term) > 2
        }
        overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
        evidence_overlap = len(key_terms & evidence_terms) / max(1, len(key_terms))
        confidence = 0.55 * top_score + 0.20 * overlap + 0.25 * evidence_overlap
        missing_specific_terms = [term for term in key_terms if len(term) >= 7 and term not in evidence_terms]
        if DocumentRetrievalPipeline._question_requires_numeric_evidence(question):
            top_evidence = " ".join(str(page.get("text_snippet", "")) for page in ranked_pages[:3])
            answer_amounts = set(_AMOUNT_RE.findall(answer))
            evidence_amounts = set(_AMOUNT_RE.findall(top_evidence))
            if not answer_amounts or not answer_amounts <= evidence_amounts:
                confidence = min(confidence, 0.19)
            else:
                confidence = max(confidence, 0.42)
        if missing_specific_terms and evidence_overlap <= 0.50:
            confidence = min(confidence, 0.19)
        elif key_terms and evidence_overlap < 0.25:
            confidence = min(confidence, 0.19)
        return round(confidence, 4)

    @staticmethod
    def _select_evidence_snippet(question: str, text: str, max_chars: int = 900) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text

        question_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower()) if len(term) > 2}
        generic_terms = {
            "what", "which", "does", "the", "this", "that", "policy", "coverage",
            "insured", "provide", "provides", "apply", "applies", "listed",
        }
        key_terms = question_terms - generic_terms
        sentences = DocumentRetrievalPipeline._candidate_sentences(text)
        scored = []
        for position, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"[a-zA-Z0-9$%]+", sentence.lower()))
            score = len(question_terms & sentence_terms)
            if _AMOUNT_RE.search(sentence):
                score += 1
            if DocumentRetrievalPipeline._question_requires_numeric_evidence(question) and _AMOUNT_RE.search(sentence):
                score += 2
            if "deductible" in question_terms and "deductible" in sentence_terms:
                score += 2
            if {"limit", "limits"} & question_terms and {"limit", "limits"} & sentence_terms:
                score += 2
            if key_terms and not (key_terms & sentence_terms):
                score -= 1
            scored.append((score, position, sentence))

        selected = []
        total_chars = 0
        for score, position, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score <= 0 and selected:
                break
            if total_chars + len(sentence) + 1 > max_chars:
                continue
            selected.append((position, sentence))
            total_chars += len(sentence) + 1
            if total_chars >= max_chars * 0.75:
                break

        if not selected:
            return text[:max_chars].rstrip()
        selected_text = " ".join(sentence for _, sentence in sorted(selected))
        return selected_text[:max_chars].rstrip()

    @staticmethod
    def _trim_context(context: str, max_chars: int = 2400) -> str:
        if len(context) <= max_chars:
            return context
        return context[:max_chars].rsplit("\n---\n", 1)[0].strip() or context[:max_chars].rstrip()

    def rank_pages(self, question: str, data_folder: Path, top_k: Optional[int] = None) -> List[Dict[str, object]]:
        top_k = top_k or self.config.max_retrievals
        merged_candidates = self.merge_candidates(question, data_folder, top_k=top_k)
        if not merged_candidates:
            return []
        page_sources = {str(candidate.get("source", "")) for candidate in merged_candidates}
        image_scores = self.retrieve_image_candidates(question, data_folder, page_sources=page_sources)
        reranked = self.rerank_multimodal_candidates(question, merged_candidates, image_scores=image_scores)
        return self.rollup_candidates_to_pages(question, reranked, top_k=top_k)

    def evaluate(self, data_folder: Path, examples_path: Path, top_k: Optional[int] = None):
        self._ensure_indices(data_folder)
        examples = load_evaluation_examples(examples_path)
        predictions = []
        for example in examples:
            prediction = self.query(example.question, data_folder, top_k=top_k)
            predictions.append((example.question, prediction))
        return evaluate_predictions(predictions, examples)

    def quick_demo(self, data_folder: Path) -> None:
        print("Building or reusing the index from:", self.config.index_dir)
        self._ensure_indices(data_folder)
        answer = self.query("What coverage limits are described?", data_folder)
        print("\n=== ANSWER ===\n", answer)
