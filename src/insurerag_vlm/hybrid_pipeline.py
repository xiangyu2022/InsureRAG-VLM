import json
import re
from hashlib import sha256
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import ModelConfig
from .answer_safety import is_explicit_abstention
from .data import (
    PACKET_MANIFEST_FILENAMES, SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS, SUPPORTED_TEXT_EXTENSIONS, PageDocument, load_documents,
)
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
from .ocr import OCRUnavailableError, extract_text_from_image, is_blank_image
from .query_understanding import QueryUnderstanding, understand_query
from .retriever import EmbeddingRetriever, SparseRetriever, load_index, load_sparse_index
from .tables import build_table_records, serialize_table_record
from .visual import build_lightweight_page_image_embeddings, score_lightweight_page_image_query
from .vlm import VLMClient, format_prompt


_AMOUNT_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d+/\d+/\d+\b")


class DocumentTextUnavailableError(ValueError):
    """The text-RAG corpus contains no readable text."""


class DocumentRetrievalPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._documents_cache: Dict[tuple[str, bool, str], List[PageDocument]] = {}
        self._hybrid_corpus_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._index_cache: Dict[str, Dict[str, Any]] = {}
        self._source_hash_cache: Dict[str, tuple[tuple[int, int], str]] = {}
        self.ingestion_report: Dict[str, Any] = {}
        self.retriever = EmbeddingRetriever(
            config.retrieval_model,
            use_hf_api=config.use_hf_api,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
            pooling=config.retrieval_pooling,
            query_instruction=config.retrieval_query_instruction,
            max_length=config.retrieval_max_length,
        )
        self.sparse_retriever = SparseRetriever()
        self.vlm_client = VLMClient(
            model_name=config.vlm_model,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
            anthropic_api_key=getattr(config, "anthropic_api_key", None),
            use_hf_api=config.use_hf_api,
            provider=config.vlm_provider,
            ollama_base_url=config.ollama_base_url,
            generation_options=config.ollama_generation_options,
            thinking=config.vlm_thinking,
            request_timeout=config.vlm_request_timeout,
            expected_model_digest=config.vlm_expected_digest,
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
            "embedding_manifest": base / "hybrid_embedding_manifest.json",
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

    def _curated_paths(self, folder: Optional[Path] = None) -> Dict[str, Path]:
        base = Path(folder if folder is not None else self.config.curated_dataset_dir)
        return {
            "pages": base / "rag_pages.jsonl",
            "snippets": base / "rag_snippets.jsonl",
        }

    def _has_curated_corpus(self) -> bool:
        paths = self._curated_paths()
        return paths["pages"].exists() and paths["snippets"].exists()

    def _resolve_corpus_source(self, data_folder: Path) -> tuple[str, Path]:
        mode = self.config.corpus_source
        if mode not in {"auto", "curated", "documents"}:
            raise ValueError(f"Unknown corpus_source {mode!r}; choose auto, curated, or documents.")
        root = Path(self.config.curated_dataset_dir if mode == "curated" else data_folder).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Requested corpus folder does not exist: {root}")
        paths = self._curated_paths(root)
        present = [path.is_file() for path in paths.values()]
        if mode == "curated" or (mode == "auto" and any(present)):
            if not all(present):
                raise ValueError(f"Curated corpus requires both rag_pages.jsonl and rag_snippets.jsonl in {root}")
            return "curated", root
        return "documents", root

    def _corpus_source_identity(self, data_folder: Path) -> Dict[str, Any]:
        mode, root = self._resolve_corpus_source(data_folder)
        if mode == "curated":
            files = sorted(self._curated_paths(root).values())
        else:
            extensions = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS
            files = sorted(path for path in root.rglob("*") if path.is_file() and (
                path.suffix.lower() in extensions or path.relative_to(root).as_posix() in PACKET_MANIFEST_FILENAMES
            ))
        hashes = {}
        for path in files:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            cached = self._source_hash_cache.get(str(path))
            if cached is None or cached[0] != signature:
                digest = sha256(path.read_bytes()).hexdigest()
                self._source_hash_cache[str(path)] = (signature, digest)
            else:
                digest = cached[1]
            hashes[path.relative_to(root).as_posix()] = digest
        return {"mode": mode, "root": str(root), "files_sha256": hashes}

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
            if doc.text.strip():
                doc.metadata["text_extraction_status"] = "readable_text"
            elif doc.image_path:
                if is_blank_image(doc.image_path):
                    doc.metadata["text_extraction_status"] = "blank_page"
                else:
                    try:
                        doc.text = extract_text_from_image(doc.image_path)
                        doc.metadata["text_extraction_status"] = "ocr_text" if doc.text.strip() else "ocr_no_text"
                    except OCRUnavailableError as exc:
                        doc.metadata["text_extraction_status"] = "ocr_unavailable"
                        doc.metadata["text_extraction_warning"] = str(exc)
            else:
                doc.metadata["text_extraction_status"] = "no_selectable_text"
                doc.metadata["text_extraction_warning"] = "No selectable text was found and page rendering/OCR is disabled. Upload a searchable PDF or enable local OCR."
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

    def _load_curated_corpus(self, folder: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        paths = self._curated_paths(folder)
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
                        "policy_number": record.get("policy_number"),
                        "source_references": record.get("source_references", []),
                        "printed_page_label": record.get("printed_page_label"),
                    }
                )
            )
        page_by_key = {str(page["page_key"]): page for page in pages}
        for record in self._read_jsonl(paths["snippets"]):
            page_number = int(record.get("page") or 0)
            doc_id = str(record.get("doc_id") or "")
            source = str(record.get("citation") or record.get("source_file") or record.get("record_id"))
            # External parent IDs are dataset IDs, not necessarily index page keys.
            page_key = self._page_key(doc_id, page_number)
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
        return corpus

    def _load_document_corpus(self, data_folder: Path, include_images: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        render_pdf_pages = include_images and self.config.enable_image_signal
        documents = self._load_documents(data_folder, render_pdf_pages=render_pdf_pages)
        statuses = [
            {"source": str(doc.metadata.get("source", doc.doc_id)),
             "status": doc.metadata.get("text_extraction_status", "readable_text"),
             "text_characters": len(doc.text.strip()),
             "warning": doc.metadata.get("text_extraction_warning")}
            for doc in documents
        ]
        self.ingestion_report = {
            "answer_input": "text", "total_pages": len(documents),
            "readable_pages": sum(bool(doc.text.strip()) for doc in documents),
            "blank_pages": sum(row["status"] == "blank_page" for row in statuses),
            "unreadable_pages": sum(not row["text_characters"] and row["status"] != "blank_page" for row in statuses),
            "pages": statuses,
        }
        if not self.ingestion_report["readable_pages"]:
            details = next((row["warning"] for row in statuses if row["warning"]), "The document contains no readable text.")
            raise DocumentTextUnavailableError(f"No readable text was found in {len(documents)} document page(s). {details} This upload was not activated.")
        pages: List[Dict[str, Any]] = []
        snippets: List[Dict[str, Any]] = []
        for doc in documents:
            if not doc.text.strip():
                continue
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
        source_identity = self._corpus_source_identity(data_folder)
        cache_key = json.dumps(source_identity, sort_keys=True) + f"::{include_images}"
        cached = self._hybrid_corpus_cache.get(cache_key)
        if cached:
            return cached

        source_mode, source_root = source_identity["mode"], Path(source_identity["root"])
        if source_mode == "curated":
            corpus = self._load_curated_corpus(source_root)
            pages = [dict(record) for record in corpus["pages"]]
            snippets = [dict(record) for record in corpus["snippets"]]
            if include_images and self.config.enable_image_signal:
                pages = self._attach_image_paths_from_documents(data_folder, pages)
            corpus = {"pages": pages, "snippets": snippets}
        else:
            # Source edits invalidate document extraction as well as this cache.
            self._documents_cache.clear()
            corpus = self._load_document_corpus(source_root, include_images=include_images)
        self._hybrid_corpus_cache[cache_key] = corpus
        return corpus

    def build_index(self, data_folder: Path) -> None:
        # An explicit rebuild must see source edits rather than an earlier cache.
        self._hybrid_corpus_cache.clear()
        self._documents_cache.clear()
        source_identity = self._corpus_source_identity(data_folder)
        corpus = self._load_hybrid_corpus(data_folder, include_images=self.config.enable_image_signal)
        paths = self._index_paths()
        paths["snippet_dense"].parent.mkdir(parents=True, exist_ok=True)
        # Invalidate a previous successful build before replacing any artifacts.
        paths["embedding_manifest"].unlink(missing_ok=True)
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
        manifest = {
            "schema_version": 4,
            "embedding_fingerprint": self.retriever.index_fingerprint(),
            "corpus_root": str(Path(data_folder).resolve()),
            "corpus_source_identity": source_identity,
            "corpus_sha256": sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "snippet_dense_shape": list(snippet_dense.shape),
            "page_dense_shape": list(page_dense.shape),
            "ingestion_report": self.ingestion_report,
        }
        # Write last, so a partially failed rebuild cannot be treated as valid.
        paths["embedding_manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.config.index_path.with_suffix(".manifest.json").write_text(
            json.dumps({"embedding_fingerprint": manifest["embedding_fingerprint"]}, indent=2), encoding="utf-8"
        )
        self._index_cache.clear()

    def _validate_embedding_manifest(self, data_folder: Path) -> Dict[str, Any]:
        path = self._index_paths()["embedding_manifest"]
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Dense index identity is missing or invalid at {path}; rebuild this index with the configured retriever.") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 4:
            raise ValueError("Unsupported dense index manifest; rebuild this index with the configured retriever.")
        if manifest.get("embedding_fingerprint") != self.retriever.index_fingerprint():
            raise ValueError("Dense index embedding model, checkpoint, pooling, or query settings changed; rebuild this index before searching.")
        if manifest.get("corpus_root") != str(Path(data_folder).resolve()):
            raise ValueError("Dense index belongs to a different corpus folder; select the correct index or rebuild it.")
        if manifest.get("corpus_source_identity") != self._corpus_source_identity(data_folder):
            raise ValueError("Dense index corpus source, files, or contents changed; rebuild this index before searching.")
        return manifest

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

        manifest = self._validate_embedding_manifest(data_folder)
        manifest_key = sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        cache_key = str(Path(self.config.index_dir).resolve()) + ":" + manifest_key
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
        for name, metadata_name in (("snippet_dense", "snippet_meta"), ("page_dense", "page_meta")):
            if list(indices[name].shape) != manifest.get(name + "_shape") or len(indices[name]) != len(indices[metadata_name]):
                raise ValueError("Dense index array shape does not match its manifest/metadata; rebuild the index.")
        indices["embedding_manifest"] = manifest
        self.ingestion_report = manifest.get("ingestion_report", {})
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
        if self.config.graph_mode == "off":
            return merged_candidates

        page_by_key = {str(record.get("page_key")): record for record in indices["page_meta"]}
        seed_order = list(dict.fromkeys(str(candidate.get("page_key") or candidate.get("parent_page_id"))
                                       for candidate in merged_candidates))
        seed_page_keys = set(seed_order[:self.config.graph_seed_pages])
        expansions = expand_candidate_page_keys(
            seed_page_keys=seed_page_keys,
            adjacency=indices.get("graph_adjacency", {}),
            needs_endorsement_check=understanding.needs_endorsement_check,
            needs_declarations=understanding.needs_declarations,
            needs_definition=understanding.needs_definition,
            needs_exclusion_review=understanding.needs_exclusion_review,
            max_hops=self.config.graph_max_hops,
            max_expansions=self.config.graph_max_expansions,
            explicit_only=self.config.graph_mode == "explicit" or not understanding.needs_graph_expansion,
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
                    "graph_relation_status": expansion.get("relation_status", "candidate"),
                    "graph_evidence_span": expansion.get("evidence_span"),
                    "graph_path": expansion.get("path", []),
                    "graph_hop": expansion.get("hop"),
                    "graph_supports_precedence": False,
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
                        "relation_status": candidate.get("graph_relation_status", "candidate"),
                        "evidence_span": candidate.get("graph_evidence_span"),
                        "path": candidate.get("graph_path", []),
                        "hop": candidate.get("graph_hop"),
                        "supports_precedence": False,
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

    def select_graph_evidence_bundle(
        self, ranked_pages: List[Dict[str, Any]], indices: Dict[str, Any], top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reserve context for explicit dependencies of a highly ranked page.

        Merely adding graph candidates can leave all dependencies below the final
        cutoff. Complete the bounded neighborhood of the first connected page in
        the initial top-k; heuristic overlap edges never reserve context slots.
        This is a retrieval policy, not a conclusion that one clause overrides another.
        """
        if self.config.graph_mode == "off" or top_k < 2:
            return ranked_pages[:top_k]
        by_source = {str(p["source"]): p for p in ranked_pages}
        key_to_source = {str(p["page_key"]): str(p["source"]) for p in indices["page_meta"]}
        source_to_key = {v: k for k, v in key_to_source.items()}
        for anchor in ranked_pages[:top_k]:
            key = source_to_key.get(str(anchor["source"]))
            if not key:
                continue
            expansion = expand_candidate_page_keys({key}, indices.get("graph_adjacency", {}),
                True, True, True, True, max_hops=self.config.graph_max_hops,
                max_expansions=self.config.graph_max_expansions, explicit_only=True)
            connected = [(row, key_to_source.get(row["page_key"])) for row in expansion]
            connected = [(row, source) for row, source in connected if source in by_source]
            if not connected:
                continue
            selected = [{**anchor, "context_selection": "explicit_reference_anchor"}]
            for row, source in connected:
                selected.append({**by_source[source], "context_selection": "explicit_reference_dependency",
                                 "context_graph_path": row["path"]})
                if len(selected) == top_k:
                    return selected
            selected_sources = {p["source"] for p in selected}
            selected.extend(p for p in ranked_pages if p["source"] not in selected_sources)
            return selected[:top_k]
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

    @staticmethod
    def _deduplicate_context_snippets(snippets: List[str]) -> List[str]:
        """Keep earlier, query-focused evidence when later snippets overlap it."""
        kept: List[str] = []
        normalized: List[str] = []
        for value in snippets:
            text = " ".join(str(value or "").split())
            if not text:
                continue
            key = text.casefold()
            if any(key in existing or existing in key for existing in normalized):
                continue
            kept.append(text)
            normalized.append(key)
        return kept

    @staticmethod
    def _clip_context_evidence(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        if limit == 1:
            return "…"
        prefix = text[:limit - 1].rstrip()
        # Avoid leaving half of a word when a useful word boundary is available.
        if " " in prefix and not text[limit - 1].isspace():
            prefix = prefix.rsplit(" ", 1)[0].rstrip()
        return prefix + "…"

    def pack_long_context(
        self,
        ranked_pages: List[Dict[str, object]],
        answer_top_k: int,
        understanding: Optional[QueryUnderstanding] = None,
    ) -> str:
        """Preserve retrieval order and reserve complete headers for each page.

        Role/section metadata remains visible to the answer model, but does not
        override relevance ranking. Fair per-page budgets prevent one long page
        from evicting another selected page's evidence.
        """
        budget = max(0, int(self.config.max_context_chars))
        if not ranked_pages or answer_top_k <= 0 or budget == 0:
            return ""
        separator = "\n---\n"
        prepared = []
        seen_sources = set()
        for page in ranked_pages:
            source = str(page.get("source") or "").strip()
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            # The query-focused excerpt comes first, followed by distinct
            # retrieval snippets. Larger snippets containing it add no duplicate.
            snippets = self._deduplicate_context_snippets([
                str(page.get("text_snippet") or ""),
                *(page.get("snippet_support") or []),
            ])[:3]
            table_lines = []
            for field in page.get("table_fields", []) or []:
                field_name = str(field.get("normalized_field_name") or field.get("field_name") or "").strip()
                field_value = str(field.get("normalized_field_value") or field.get("field_value") or "").strip()
                if field_name or field_value:
                    table_lines.append(f"- TABLE: {field_name}: {field_value}".strip(": "))
            body = "\n".join([*(f"- {snippet}" for snippet in snippets), *table_lines[:2]])
            if not body:
                continue
            body = self._clip_context_evidence(body, max(1, int(self.config.max_page_chars)))
            header_lines = [
                f"SOURCE: {source}",
                f"ROLE: {page.get('document_type', 'page')} / {page.get('primary_clause_type', 'general')}",
            ]
            if page.get("section_anchor"):
                header_lines.append("SECTION: " + self._clip_context_evidence(str(page["section_anchor"]), 160))
            header = "\n".join(header_lines) + "\n"
            prepared.append((header, body))
            if len(prepared) >= answer_top_k:
                break

        # If the budget cannot fit all headers plus a little evidence, retain
        # higher-ranked pages only. Never produce a partial SOURCE header.
        while prepared:
            overhead = sum(len(header) for header, _ in prepared) + len(separator) * (len(prepared) - 1)
            minimum_evidence = sum(min(32, len(body)) for _, body in prepared)
            if overhead + minimum_evidence <= budget:
                break
            prepared.pop()
        if not prepared:
            return ""
        available = budget - overhead
        allowances = [0] * len(prepared)
        active = list(range(len(prepared)))
        while available > 0 and active:
            share = max(1, available // len(active))
            for index in active:
                add = min(share, len(prepared[index][1]) - allowances[index], available)
                allowances[index] += add
                available -= add
            active = [index for index in active if allowances[index] < len(prepared[index][1])]
        return separator.join(
            header + self._clip_context_evidence(body, allowances[index])
            for index, (header, body) in enumerate(prepared)
        )

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
                **self.vlm_client.answer_trace(invoked=False),
            }
        page_sources = {str(candidate.get("source", "")) for candidate in merged_candidates}
        image_scores = self.retrieve_image_candidates(question, data_folder, page_sources=page_sources)
        reranked = self.rerank_multimodal_candidates(question, merged_candidates, understanding=understanding, image_scores=image_scores)
        ranked_pages = self.rollup_candidates_to_pages(question, reranked, top_k=max(top_k, self.config.candidate_pool_size))
        ranked_pages = self.select_graph_evidence_bundle(ranked_pages, self._ensure_indices(data_folder), top_k)
        combined_context = self.pack_long_context(ranked_pages, answer_top_k, understanding=understanding)
        prompt = format_prompt(combined_context, question, self.config.prompt_template)
        answer = self.vlm_client.generate_extractive(prompt) if force_extractive else self.vlm_client.generate(prompt)
        return {
            "answer": answer,
            "retrieval_context": combined_context,
            "source_ranking": ranked_pages,
            "tool_router": self._tool_router_stub(question, ranked_pages),
            "query_understanding": understanding.to_dict(),
            **self.vlm_client.answer_trace(invoked=True, force_extractive=force_extractive),
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
        if preferred_field_type is None:
            return None

        target_coverages = set(understand_query(question).target_coverages or [])
        candidate_fields = list(cited_page.get("table_fields", []) or [])
        if target_coverages:
            candidate_fields = [
                field for field in candidate_fields
                if not field.get("coverage_tags") or target_coverages & set(field.get("coverage_tags", []) or [])
            ]

        for field in candidate_fields:
            field_type = str(field.get("field_type", ""))
            if field_type != preferred_field_type:
                continue
            field_value = str(field.get("normalized_field_value") or field.get("field_value") or "").strip()
            if field_value:
                return field_value
        # An unrelated table value or illustrative amount is not the requested
        # policy field. Absence stays null instead of guessing from the page.
        return None

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
                            "status": "candidate_relation_requires_review",
                            "severity": "review",
                            "message": "Metadata overlap suggests a related endorsement. This does not establish an override or which version controls.",
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
                            "status": "candidate_relation_requires_review",
                            "severity": "review",
                            "message": "Metadata overlap suggests a related exception. Its applicability requires the underlying clause text.",
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
        answer_repaired = False
        explicit_abstention = is_explicit_abstention(answer)
        generation_truncated = bool(((result.get("backend_metadata") or {}).get("last_generation") or {}).get("truncated"))
        ranked_pages = result["source_ranking"]
        citations = []
        cited_source = self._extract_answer_source(answer, ranked_pages)
        clean_answer = re.sub(
            r"(?im)^\s*(?:[-*]\s*)?[*_`]*sources?[*_`]*\s*:[*_`]*[^\n]*(?:\n|$)", "", answer,
        ).strip()
        cited_page = None
        citation_origin = None
        if cited_source:
            cited_page = next((page for page in ranked_pages if page["source"] == cited_source), None)
            if cited_page:
                citation_origin = "model_source"
        # An explicit unknown citation is a validation failure, not permission
        # to silently replace it with a different retrieved source.
        if cited_page is None and cited_source is None and not explicit_abstention and not generation_truncated:
            cited_page = self._choose_cited_page(
                question, clean_answer, ranked_pages,
                retrieval_context=str(result["retrieval_context"]) if "retrieval_context" in result else None,
                min_overlap=getattr(self.config, "citation_min_overlap", 0.20),
            )
            if cited_page:
                citation_origin = "evidence_selection"
        if cited_page and not explicit_abstention and not generation_truncated:
            repaired_answer = self._repair_answer_from_evidence(question, clean_answer, cited_page)
            if not repaired_answer and ("insufficient_evidence" in answer.lower() or not clean_answer):
                repaired_answer = self._best_sentence_from_evidence(question, str(cited_page.get("text_snippet", "")))
            if repaired_answer:
                clean_answer = repaired_answer
                answer_repaired = True
            evidence_text = str(cited_page.get("text_snippet", ""))
            if "retrieval_context" in result:
                evidence_text = self._packed_source_evidence(str(result["retrieval_context"]), str(cited_page["source"]))
            citations.append(
                {
                    "source": cited_page["source"],
                    "origin": citation_origin,
                    "page_id": self._source_to_page_id(str(cited_page["source"])),
                    "evidence_text": evidence_text,
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
                    "source_scope_text": self._source_scope_text(cited_page),
                }
            )

        supported, support_reason = self._citation_support_details(
            question,
            clean_answer,
            citations,
            min_overlap=getattr(self.config, "citation_min_overlap", 0.20),
        )
        # A multi-paragraph answer can give a supported primary fact followed
        # by an uncited comparison amount. Serve the complete first paragraph
        # only when it independently passes the same evidence checks. Preserve
        # the raw answer and mark this deterministic reduction as a repair.
        if not supported and (support_reason == "answer_amount_not_in_citation" or citation_origin == "evidence_selection") and not explicit_abstention and not generation_truncated and self._allows_primary_paragraph_reduction(question):
            primary_claim = re.split(r"\n\s*\n", clean_answer, maxsplit=1)[0].strip()
            if primary_claim and primary_claim != clean_answer:
                primary_supported, primary_reason = self._citation_support_details(
                    question, primary_claim, citations,
                    min_overlap=getattr(self.config, "citation_min_overlap", 0.20),
                )
                if primary_supported:
                    clean_answer = primary_claim
                    answer_repaired = True
                    supported, support_reason = primary_supported, primary_reason
        confidence = self._estimate_confidence(question, clean_answer, ranked_pages)
        structured_fields = self._structured_answer_fields(question, understanding, cited_page, ranked_pages)
        if not supported:
            confidence = min(confidence, 0.19)
        blocking_conflicts = [conflict for conflict in structured_fields.get("conflicts", []) if conflict.get("severity") == "blocking"]
        if blocking_conflicts:
            confidence = min(confidence, 0.15)
        threshold = getattr(self.config, "abstain_threshold", 0.20)
        abstain = confidence < threshold or explicit_abstention or generation_truncated or not supported or bool(blocking_conflicts)
        caveats: List[str] = []
        if citations and citations[0].get("document_type") != "declarations" and self._question_requires_numeric_evidence(question):
            caveats.append("Numeric answer was not cited from a declarations-style page.")
        if ranked_pages and any("overridden_by" in relation for relation in ranked_pages[0].get("graph_relations", []) or []):
            caveats.append("Graph metadata suggests an endorsement relationship; it does not establish precedence.")
        if ranked_pages and any("qualified_by" in relation for relation in ranked_pages[0].get("graph_relations", []) or []):
            caveats.append("Graph metadata suggests an exception relationship; verify applicability in the clause text.")
        if blocking_conflicts:
            caveats.extend(str(conflict.get("message")) for conflict in blocking_conflicts)
        return {
            "answer": "" if abstain else clean_answer,
            "raw_answer": answer,
            "answer_repaired": answer_repaired,
            "explicit_abstention": explicit_abstention,
            "generation_truncated": generation_truncated,
            "generation_used": result.get("generation_used", False),
            "answer_backend": "deterministic-evidence-repair" if answer_repaired and not abstain else result.get("answer_backend"),
            "backend_metadata": result.get("backend_metadata"),
            "citations": [] if abstain else citations,
            "citation_origin": None if abstain else citation_origin,
            "confidence": confidence,
            "abstain": abstain,
            "abstain_reason": "generation_truncated" if generation_truncated else "missing_policy_packet_counterevidence" if blocking_conflicts else "insufficient_retrieved_evidence" if abstain else None,
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
        # A complete, unqualified definition request does not ask for a policy
        # amount. Otherwise the numeric repair step replaces a valid definition
        # with an arbitrary dollar example from the cited FAQ. Full matching
        # keeps personal-policy and mixed definition/amount requests numeric.
        term = r"(?:deductible|premium|sublimit|coinsurance|retention|coverage limit|policy limit)"
        concept = rf"(?:(?:insurance|health insurance|auto insurance)\s+)?{term}"
        if re.fullmatch(
            rf"\s*(?:what\s+is\s+(?:a|an)\s+{concept}|what\s+does\s+(?:a\s+|an\s+)?{concept}\s+mean|define\s+(?:a\s+|an\s+)?{concept})\s*[?.!]?\s*",
            lowered,
        ):
            return False
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
        if is_explicit_abstention(answer):
            return None
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
    def _allows_primary_paragraph_reduction(cls, question: str) -> bool:
        # A partial answer must never replace a requested comparison or list of
        # facts. Ambiguous multi-part wording retains the original abstention.
        if question.count("?") > 1 or re.search(r"\b(?:and|or|both|each|respectively|versus|vs)\b|;|\b[A-Za-z]+\s*,\s*[A-Za-z]+\b", question, re.I):
            return False
        return len(understand_query(question).target_coverages) <= 1

    @classmethod
    def _support_terms(cls, text: str) -> Set[str]:
        terms = set()
        for term in cls._terms(text, min_len=3):
            term = {"included": "include", "includes": "include", "provided": "provide", "provides": "provide"}.get(term, term)
            if len(term) > 5 and term.endswith("ies"):
                term = term[:-3] + "y"
            elif len(term) > 5 and term.endswith(("sses", "xes", "ches", "shes")):
                term = term[:-2]
            elif len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
                term = term[:-1]
            terms.add(term)
        if re.search(r"\b(?:up to|at most|no more than|not more than)\b", text, re.I):
            terms.add("maximum")
        if re.search(r"\b(?:at least|no less than|not less than)\b", text, re.I):
            terms.add("minimum")
        return terms

    @staticmethod
    def _document_lookup_scope(question: str) -> tuple[str, str]:
        match = re.match(
            r"^\s*(?:using|according to|based on|from|in)\s+(.+?\b(?:guides?|documents?|pdf|polic(?:y|ies)|files?))\s*[,;:]\s*", question, re.I,
        )
        return (question[match.end():], match.group(1)) if match else (question, "")

    @staticmethod
    def _source_scope_text(page: Dict[str, object]) -> str:
        # Full-page metadata may establish document identity, while factual
        # amounts still have to appear in the actual packed citation evidence.
        return " ".join([str(page.get("source", "")), str(page.get("source_name", "")),
                         str(page.get("text_snippet", "")), *(str(x) for x in page.get("snippet_support", []) or [])])

    @classmethod
    def _citation_support_details(
        cls,
        question: str,
        answer: str,
        citations: List[Dict[str, object]],
        min_overlap: float = 0.20,
    ) -> tuple[bool, str]:
        if is_explicit_abstention(answer):
            return False, "model_reported_insufficient_evidence"
        if not citations:
            return False, "missing_citation"
        evidence = " ".join(str(citation.get("evidence_text", "")) for citation in citations)
        if not evidence.strip():
            return False, "missing_citation_evidence"
        factual_input, document_scope = cls._document_lookup_scope(question)
        if document_scope:
            scope_ignored = cls._generic_terms() | {"using", "uploaded", "attached", "provided", "supplied", "available", "following", "our", "own", "the"}
            required_scope = cls._support_terms(document_scope) - scope_ignored
            scope_evidence = " ".join(str(c.get("source_scope_text", c.get("evidence_text", ""))) for c in citations)
            if not required_scope <= cls._support_terms(scope_evidence):
                return False, "document_scope_not_in_citation"

        # Exact identity constraints must survive removal of conversational
        # lookup words. Token overlap alone drops short ID components and can
        # confuse policies or versions that differ by one character.
        def identifiers(text: str) -> Set[str]:
            tokens = {match.lower() for match in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", text)
                      if any(char.isdigit() for char in match) or match.isupper()}
            tokens.update(match.lower() for match in re.findall(
                r"\b(?:policy|claim|endorsement|report|identifier)\s*(?:(?:number|identifier|report)\s*)?[:#]?\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b", text, re.I
            ))
            return tokens
        evidence_identifiers = {match.lower() for match in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", evidence)}
        if not identifiers(question) <= evidence_identifiers or not identifiers(answer) <= evidence_identifiers:
            return False, "identifier_not_in_citation"
        factual_question = re.sub(r"\bwithout\s+(?:deciding|determining)\b[^,;?]*[,;]", "", factual_input, flags=re.I)
        requested_version = re.search(r"\bversion\s+(?:label\s+)?([A-Za-z0-9]+)\b", factual_question, re.I)
        evidence_version = re.search(r"\bversion\s+(?:label\s+)?([A-Za-z0-9]+)\b", evidence, re.I)
        if requested_version and len(requested_version.group(1)) <= 3 and (
            not evidence_version or requested_version.group(1).lower() != evidence_version.group(1).lower()
        ):
            return False, "requested_version_not_in_citation"

        contrast = re.search(r"\b(?:rather than|instead of)\s+([^,;?]+)", factual_question, re.I)
        if contrast:
            # A requested contrast is not a requirement that the rejected
            # alternative also appear in the primary citation.
            factual_question = factual_question[:contrast.start()] + factual_question[contrast.end():]
            if "example" in contrast.group(1).lower() and all(c.get("document_type") != "declarations" for c in citations) and re.search(
                r"\b(?:educational|illustrative|worked)\b.{0,30}\bexample\b|\bin this example only\b", evidence, re.I
            ):
                return False, "requested_policy_value_is_only_example"
        if "example" not in factual_question.lower() and (identifiers(question) or re.search(r"\b(?:my|our|own|actual|actually|declared)\b", question, re.I)) and re.search(
            r"\b(?:educational|illustrative|worked)\b.{0,30}\bexample\b|\bin this example only\b", evidence, re.I
        ) and all(c.get("document_type") != "declarations" for c in citations):
            return False, "requested_policy_value_is_only_example"
        if re.search(r"\b(?:controls?|controlling|currently|latest|issued later|precedence)\b", factual_question, re.I) and re.search(
            r"\b(?:which\s+version\s+controls|controlling\s+version)\b.{0,35}\bnot\s+(?:established|stated|specified|known)\b", evidence, re.I
        ):
            return False, "controlling_version_not_established"
        if re.search(r"\b(?:approved|settlement)\b", question, re.I) and re.search(
            r"\bno\b[^.;]{0,70}\bapproved\b[^.;]{0,45}\b(?:payment|amount)\b|\bdoes not\s+(?:list|state|establish)\s+an?\s+approved\b", evidence, re.I
        ):
            return False, "approved_payment_not_established"

        answer_amounts = set(_AMOUNT_RE.findall(answer or ""))
        evidence_amounts = set(_AMOUNT_RE.findall(evidence))
        if answer_amounts and not answer_amounts <= evidence_amounts:
            return False, "answer_amount_not_in_citation"
        if cls._question_requires_numeric_evidence(question) and not answer_amounts:
            return False, "numeric_question_without_answer_amount"
        date_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b"
        date_keys = lambda text: {re.sub(r"[,\s]+", " ", value.lower()).strip() for value in re.findall(date_pattern, text, re.I)}
        if not date_keys(answer) <= date_keys(evidence):
            return False, "answer_date_not_in_citation"

        key_terms = cls._support_terms(factual_question) - cls._generic_terms()
        evidence_terms = cls._support_terms(evidence)
        if key_terms and not (key_terms & evidence_terms):
            return False, "question_terms_not_in_citation"
        broad_value_terms = {
            "coverage", "cover", "include", "policy", "insurance", "limit", "limits",
            "sublimit", "deductible", "endorsement", "reimbursement", "provision",
            "amount", "liability", "property", "loss", "use", "auto", "automobile",
            "guide", "document", "pdf", "apply", "applies",
            "actual", "actually", "declared", "printed", "stated", "shown", "date",
            "benefit", "benefits",
        }
        specific_terms = {term for term in key_terms if len(term) >= 4 and term not in broad_value_terms}
        if specific_terms and not specific_terms <= evidence_terms:
            return False, "specific_question_terms_not_in_citation"

        answer_terms = cls._support_terms(answer) - cls._generic_terms()
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
        explicit = re.search(r"sources?[*_`]*\s*:[*_`]*\s*(.+)", answer, flags=re.IGNORECASE)
        cited_text = explicit.group(1) if explicit else answer
        matches = []
        for page in ranked_pages or []:
            source = str(page.get("source", ""))
            # Require a delimiter after the complete source, so p1 cannot match
            # p10. Markdown wrapping and explanatory text remain outside it.
            if source:
                match = re.search(r"(?<![\w./\\-])" + re.escape(source) + r"(?=$|[\s),.;:*_`\]])", cited_text, re.I)
                if match:
                    matches.append((match.start(), source))
        if matches:
            return min(matches)[1]
        if explicit:
            return cited_text.strip()
        return None

    @staticmethod
    def _packed_source_evidence(context: str, source: str) -> str:
        sections = re.split(r"(?m)^SOURCE:\s*([^\n]+)\n", context)
        for index in range(1, len(sections), 2):
            if sections[index].strip() == source:
                return sections[index + 1].strip()
        return ""

    @classmethod
    def _choose_cited_page(
        cls,
        question: str,
        answer: str,
        ranked_pages: List[Dict[str, object]],
        retrieval_context: Optional[str] = None,
        min_overlap: float = 0.20,
    ) -> Optional[Dict[str, object]]:
        """Select only a page that validates a complete served claim.

        Retrieval rank breaks ties only after evidence support. This is an
        explicit postprocessing citation, not a citation emitted by the model.
        """
        if is_explicit_abstention(answer):
            return None
        primary_claim = re.split(r"\n\s*\n", answer, maxsplit=1)[0].strip()
        can_reduce = primary_claim != answer and cls._allows_primary_paragraph_reduction(question)
        primary_candidates = []
        for page in ranked_pages:
            evidence = (cls._packed_source_evidence(retrieval_context, str(page.get("source", "")))
                        if retrieval_context is not None else str(page.get("text_snippet", "")))
            citation = [{"evidence_text": evidence, "document_type": page.get("document_type"),
                         "source_scope_text": cls._source_scope_text(page)}]
            if cls._citation_support_details(question, answer, citation, min_overlap)[0]:
                return page
            if can_reduce and cls._citation_support_details(question, primary_claim, citation, min_overlap)[0]:
                primary_candidates.append(page)
        return primary_candidates[0] if primary_candidates else None

    @staticmethod
    def _estimate_confidence(question: str, answer: str, ranked_pages: List[Dict[str, object]]) -> float:
        if not ranked_pages:
            return 0.0
        # Use the same factual scope as citation validation; lookup scaffolding
        # must not independently cap a well-supported answer below threshold.
        question = DocumentRetrievalPipeline._document_lookup_scope(question)[0]
        question = re.sub(r"\bwithout\s+(?:deciding|determining)\b[^,;?]*[,;]", "", question, flags=re.I)
        question = re.sub(r"\b(?:rather than|instead of)\s+[^,;?]+", "", question, flags=re.I)
        top_score = max(0.0, min(1.0, float(ranked_pages[0].get("score", 0.0))))
        question_terms = DocumentRetrievalPipeline._support_terms(question)
        answer_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", answer.lower()) if len(term) > 2}
        generic_terms = DocumentRetrievalPipeline._generic_terms() | {
            "actual", "actually", "declared", "printed", "stated", "shown", "date",
            "benefit", "benefits",
        }
        key_terms = question_terms - generic_terms
        evidence_terms = {
            term
            for page in ranked_pages[:3]
            for term in DocumentRetrievalPipeline._support_terms(str(page.get("text_snippet", "")))
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
