"""Local persistent clause store. Import is operator-only; queries never open sources."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np

from .retriever import EmbeddingRetriever, SparseRetriever


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def bounded_string(value, name, maximum=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    return value


def hash_value(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def normalize_record(record):
    """Preserve original fields; never infer a raw-source hash from snippet text."""
    row = dict(record)
    row["record_id"] = bounded_string(row.get("record_id"), "record_id", 2000)
    row["doc_id"] = bounded_string(row.get("doc_id"), "doc_id", 2000)
    row["text"] = bounded_string(row.get("text"), "text", 100000)
    page = row.get("page_number", row.get("page"))
    if type(page) is not int or page < 0:
        raise ValueError("page_number must be a nonnegative integer (0 means unspecified)")
    row["page_number"] = page
    row["page_id"] = bounded_string(row.get("page_id") or row.get("parent_page_id") or
                                    row.get("page_key") or f"{row['doc_id']}#page={page}", "page_id", 2200)
    row["citation"] = bounded_string(row.get("citation") or row.get("source"), "citation", 4000)
    row["source_sha256"] = hash_value(row.get("source_sha256") or row.get("source_hash"), "source_sha256")
    text_hash = sha256(row["text"].encode("utf-8")).hexdigest()
    if row.get("text_sha256", text_hash) != text_hash:
        raise ValueError("text_sha256 does not match text")
    row["text_sha256"] = text_hash
    if "char_start" in row or "char_end" in row:
        start, end = row.get("char_start"), row.get("char_end")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("Invalid evidence character offsets")
    for name in ("jurisdiction", "publisher"):
        if name in row:
            bounded_string(row[name], name, 300)
    # Opaque DB key; legacy IDs (including paths) remain metadata, never file arguments.
    eid = sha256(row["record_id"].encode("utf-8")).hexdigest()
    if row.get("evidence_id", eid) != eid:
        raise ValueError("evidence_id does not match stable record_id")
    row.pop("record_sha256", None)
    row["evidence_id"] = eid
    row["record_sha256"] = digest(row)
    return row


@contextmanager
def store_lock(root):
    """One local owner at a time: stop the stdio server before importing updates."""
    with (root / ".insurerag.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Store busy; stop its other reader/importer first") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Store busy; stop its other reader/importer first") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class ClauseStore:
    """Chroma dense retrieval + existing BM25, fused with reciprocal ranks.

    Only explicit local embeddings are accepted. Chroma's embedding function is
    disabled on create AND reopen. The importer must supply trusted provenance.
    """

    def __init__(self, path, *, collection="insurance_clauses", model="local-hashing",
                 pooling="auto", query_instruction="", max_length=None, writable=False):
        if model != "local-hashing" and not Path(model).is_dir():
            raise ValueError("Embedding model must be local-hashing or an existing local directory")
        self.embedder = EmbeddingRetriever(model, pooling=pooling, query_instruction=query_instruction,
                                           max_length=max_length)
        self.fingerprint = self.embedder.index_fingerprint()
        self.dimension = self.fingerprint["embedding_dim"]
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Embedding dimension must be explicit")
        self.path = Path(path).resolve()
        self.writable = writable
        self.name = collection
        self._cache = None
        self._lock = None

    def __enter__(self):
        import chromadb
        from chromadb.config import Settings

        if self.writable:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not (self.path / "chroma.sqlite3").is_file():
            raise ValueError("Store does not exist; import clauses before starting MCP")
        self._lock = store_lock(self.path)
        self._lock.__enter__()
        try:
            self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
            contract = {"insurerag_schema": 1, "embedding_contract": canonical(self.fingerprint),
                        "dimension": self.dimension, "distance": "cosine"}
            if self.writable:
                self.collection = self.client.get_or_create_collection(
                    self.name, embedding_function=None, metadata=contract,
                    configuration={"hnsw": {"space": "cosine"}})
            else:
                self.collection = self.client.get_collection(self.name, embedding_function=None)
            if self.collection.metadata != contract:
                raise ValueError("Collection embedding/schema contract mismatch; use a separate collection")
            return self
        except BaseException:
            self._lock.__exit__(None, None, None)
            self._lock = None
            raise

    def __exit__(self, *args):
        if self._lock:
            self._lock.__exit__(*args)
            self._lock = None
        self._cache = None

    def _vectors(self, vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (count, self.dimension) or not np.isfinite(values).all():
            raise ValueError("Invalid embedding shape or non-finite values")
        if np.any(np.linalg.norm(values, axis=1) < 1e-8):
            raise ValueError("Zero embedding: text has no supported tokens")
        return values

    def upsert(self, records, *, vectors=None, fingerprint=None):
        if not self.writable:
            raise PermissionError("This store was opened read-only")
        rows = [normalize_record(row) for row in records]
        ids = [row["evidence_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate record_id in import batch")
        if vectors is not None:
            if fingerprint != self.fingerprint:
                raise ValueError("Legacy vector fingerprint does not match the configured embedder")
            vectors = self._vectors(vectors, len(rows))
        if not rows:
            return {"total": 0, "written": 0, "unchanged": 0}
        existing = {}
        for offset in range(0, len(ids), 200):
            data = self.collection.get(ids=ids[offset:offset + 200], include=["metadatas"])
            existing.update(zip(data["ids"], data["metadatas"]))
        for row in rows:
            previous = existing.get(row["evidence_id"])
            if previous and previous["doc_id"] != row["doc_id"]:
                raise ValueError("Existing record_id belongs to a different doc_id")
        changed = [i for i, row in enumerate(rows)
                   if existing.get(row["evidence_id"], {}).get("record_sha256") != row["record_sha256"]]
        # Validate/encode the entire changed batch before the first write. Interrupted
        # multi-batch writes can safely be rerun; there is no automatic deletion.
        if changed:
            embeddings = self._vectors(vectors[changed] if vectors is not None else
                                       self.embedder.embed_texts([rows[i]["text"] for i in changed]), len(changed))
            for offset in range(0, len(changed), 200):
                indices = changed[offset:offset + 200]
                selected = [rows[i] for i in indices]
                metadatas = [{**{key: row[key] for key in ("doc_id", "page_number", "source_sha256",
                              "record_sha256", "jurisdiction", "publisher") if key in row},
                              "record_json": canonical(row)} for row in selected]
                self.collection.upsert(ids=[row["evidence_id"] for row in selected],
                                       documents=[row["text"] for row in selected], metadatas=metadatas,
                                       embeddings=embeddings[offset:offset + len(indices)].tolist())
            self._cache = None
        return {"total": len(rows), "written": len(changed), "unchanged": len(rows) - len(changed)}

    def read_evidence(self, evidence_id, expected_sha256=None):
        hash_value(evidence_id, "evidence_id")
        if expected_sha256 is not None:
            hash_value(expected_sha256, "expected_sha256")
        result = self.collection.get(ids=[evidence_id], include=["metadatas"])
        if not result["ids"]:
            return {"status": "not_found", "evidence_id": evidence_id}
        row = json.loads(result["metadatas"][0]["record_json"])
        if expected_sha256 is not None and expected_sha256 != row["record_sha256"]:
            return {"status": "stale", "evidence_id": evidence_id, "record_sha256": row["record_sha256"]}
        return {"status": "found", "evidence": row}

    @staticmethod
    def _where(filters):
        if filters is None or filters == {}:
            return None
        if not isinstance(filters, dict) or not set(filters) <= {"doc_id", "page_number", "source_sha256", "jurisdiction", "publisher"}:
            raise ValueError("Unsupported filter; use allowlisted equality fields")
        conditions = []
        for key, value in filters.items():
            if key == "page_number":
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid page_number filter")
            elif key == "source_sha256":
                hash_value(value, key)
            else:
                bounded_string(value, key, 2000 if key == "doc_id" else 300)
            conditions.append({key: {"$eq": value}})
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def search_clauses(self, query, top_k=5, filters=None):
        bounded_string(query, "query", 2000)
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")
        where = self._where(filters)
        cache_key = canonical(filters)
        if self._cache is None or self._cache[0] != cache_key:
            data = self.collection.get(where=where, include=["documents", "metadatas"])
            pairs = sorted(zip(data["ids"], data["documents"], data["metadatas"]))
            sparse = SparseRetriever().build([pair[1] for pair in pairs])
            self._cache = (cache_key, pairs, sparse)
        _, pairs, sparse = self._cache
        if not pairs:
            return {"status": "ok", "results": [], "embedding": self.fingerprint}
        n = min(len(pairs), max(top_k * 4, 20))
        query_vector = self._vectors(self.embedder.embed_texts([self.embedder.query_instruction + query]), 1)
        dense = self.collection.query(query_embeddings=query_vector.tolist(), n_results=n,
                                      where=where, include=["distances"])
        ranks = {}
        for rank, eid in enumerate(dense["ids"][0], 1):
            ranks[eid] = {"dense_rank": rank, "rrf_score": 1 / (60 + rank)}
        for rank, idx in enumerate(SparseRetriever().search(query, sparse, top_k=n), 1):
            eid = pairs[idx][0]
            item = ranks.setdefault(eid, {"rrf_score": 0.0})
            item["bm25_rank"] = rank
            item["rrf_score"] += 1 / (60 + rank)
        metadata = {pair[0]: pair[2] for pair in pairs}
        ordered = sorted(ranks, key=lambda eid: (-ranks[eid]["rrf_score"], eid))[:top_k]
        results = [{"evidence": json.loads(metadata[eid]["record_json"]), **ranks[eid]} for eid in ordered]
        return {"status": "ok", "results": results, "embedding": self.fingerprint,
                "ranking": "equal-weight BM25+dense RRF(k=60); score is not confidence"}


class ChromaRetrieverAdapter:
    """Use with ReviewAgent and a corpus-specific runtime_id; approval stays mandatory."""
    def __init__(self, store):
        self.store = store

    def search(self, query, top_k):
        result = self.store.search_clauses(query, top_k)
        return [{"source": canonical({key: hit["evidence"][key] for key in
                 ("evidence_id", "record_sha256", "source_sha256", "citation", "page_id")}),
                 "text": hit["evidence"]["text"]} for hit in result["results"]]
