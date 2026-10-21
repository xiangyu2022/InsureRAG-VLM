"""Synthetic plumbing fixtures only; no insurance/real-model quality claims."""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.insurerag_vlm.chroma_knowledge import ClauseStore, ChromaRetrieverAdapter, normalize_record
from src.insurerag_vlm.retriever import EmbeddingRetriever, SparseRetriever


REPO = Path(__file__).resolve().parents[1]


def clauses():
    return [{"record_id": f"fictional-{i}#page=2::c001", "doc_id": f"fictional-{i}",
             "page_number": 2, "parent_page_id": f"fictional-{i}#page=2",
             "citation": f"FICTITIOUS fixture {i}, page 2", "source_sha256": sha256(f"synthetic source {i}".encode()).hexdigest(),
             "jurisdiction": region, "publisher": "FICTITIOUS", "text": text,
             "source_origin": "synthetic_protocol_test"}
            for i, region, text in [(1, "AA", "Dental crowns have a fictional waiting period of twelve months."),
                                    (2, "BB", "Travel baggage has a fictional deductible of fifty dollars.")]]


def run_import(path, records, *extra):
    source = path.parent / "records.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "src.insurerag_vlm.clause_store_cli", "--store", str(path),
                             "--records", str(source), *map(str, extra)], cwd=REPO, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_incremental_versions_filters_and_read_only(tmp_path, monkeypatch):
    path = tmp_path / "db"
    records = clauses()
    with ClauseStore(path, writable=True) as store:
        assert store.upsert(records)["written"] == 2
        assert store.collection.count() == 2
        def no_embedding(_):
            raise AssertionError("Unchanged records must not be re-embedded")
        original = store.embedder.embed_texts
        monkeypatch.setattr(store.embedder, "embed_texts", no_embedding)
        assert store.upsert(records) == {"total": 2, "written": 0, "unchanged": 2}
        monkeypatch.setattr(store.embedder, "embed_texts", original)
        first = store.search_clauses("dental crowns")["results"][0]
        evidence = first["evidence"]
        assert evidence["record_id"] == records[0]["record_id"]
        assert first["bm25_rank"] == 1 and first["dense_rank"] == 1
        assert evidence["page_id"] == records[0]["parent_page_id"]
        assert store.read_evidence(evidence["evidence_id"], evidence["record_sha256"])["status"] == "found"
        filtered = store.search_clauses("dental crowns", filters={"jurisdiction": "BB", "page_number": 2})
        assert [hit["evidence"]["doc_id"] for hit in filtered["results"]] == ["fictional-2"]
        assert store.search_clauses("crowns", filters={"doc_id": "absent"})["results"] == []
        assert store.search_clauses("crowns", filters={"source_sha256": records[0]["source_sha256"]})["results"][0]["evidence"]["doc_id"] == "fictional-1"
        records[0]["text"] = "Dental crowns have a fictional waiting period of six months."
        records[0]["source_sha256"] = sha256(b"revised synthetic source").hexdigest()
        assert store.upsert(records)["written"] == 1
        assert store.collection.count() == 2
        assert store.read_evidence(evidence["evidence_id"], evidence["record_sha256"])["status"] == "stale"
        assert "six months" in store.search_clauses("dental crowns")["results"][0]["evidence"]["text"]
    with ClauseStore(path) as store:
        assert store.dimension == 512
        assert store.fingerprint["backend"] == "local-hashing"
        with pytest.raises(PermissionError):
            store.upsert(records)
        assert store.read_evidence("0" * 64) == {"status": "not_found", "evidence_id": "0" * 64}
        adapted = ChromaRetrieverAdapter(store).search("dental", 1)[0]
        assert json.loads(adapted["source"])["evidence_id"] == evidence["evidence_id"]


def test_reject_invalid_import_before_writes_and_contract_mismatch(tmp_path):
    with ClauseStore(tmp_path / "db", writable=True) as store:
        rows = clauses()
        bad = dict(rows[1], source_sha256="unknown")
        with pytest.raises(ValueError, match="source_sha256"):
            store.upsert([rows[0], bad])
        assert store.collection.count() == 0
        with pytest.raises(ValueError, match="Duplicate"):
            store.upsert([rows[0], rows[0]])
        vectors = store.embedder.embed_texts([r["text"] for r in rows])
        with pytest.raises(ValueError, match="fingerprint"):
            store.upsert(rows, vectors=vectors, fingerprint={})
        with pytest.raises(ValueError, match="shape"):
            store.upsert(rows, vectors=vectors[:, :10], fingerprint=store.fingerprint)
        vectors[1, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            store.upsert(rows, vectors=vectors, fingerprint=store.fingerprint)
        assert store.collection.count() == 0
        store.collection.modify(metadata={"embedding_contract": "wrong"})
    with pytest.raises(ValueError, match="contract mismatch"):
        with ClauseStore(tmp_path / "db"):
            pass


@pytest.mark.parametrize("changes", [{"text_sha256": "0" * 64}, {"char_start": -1, "char_end": 5},
                                    {"page_number": True}, {"evidence_id": "0" * 64}])
def test_provenance_validation(changes):
    with pytest.raises(ValueError):
        normalize_record({**clauses()[0], **changes})


@pytest.mark.parametrize("arguments", [{"query": " "}, {"query": "a", "top_k": True},
                                      {"query": "a", "top_k": 21}, {"query": "a", "filters": {"$where": "anything"}},
                                      {"query": "a", "filters": {"doc_id": {"$ne": "x"}}}])
def test_query_validation(tmp_path, arguments):
    with ClauseStore(tmp_path / "db", writable=True) as store:
        with pytest.raises(ValueError):
            store.search_clauses(**arguments)


def test_no_remote_models_no_implicit_creation(tmp_path):
    with pytest.raises(ValueError, match="local"):
        ClauseStore(tmp_path / "db", model="BAAI/bge-small-en-v1.5")
    with pytest.raises(ValueError, match="does not exist"):
        with ClauseStore(tmp_path / "db"):
            pass
    assert not (tmp_path / "db").exists()


def test_query_instruction_and_identity_collision(tmp_path, monkeypatch):
    with ClauseStore(tmp_path / "db", writable=True, query_instruction="Represent this query: ") as store:
        store.upsert(clauses())
        encoded = []
        original = store.embedder.embed_texts
        def capture(texts):
            encoded.extend(texts)
            return original(texts)
        monkeypatch.setattr(store.embedder, "embed_texts", capture)
        store.search_clauses("dental")
        assert encoded == ["Represent this query: dental"]
        with pytest.raises(ValueError, match="different doc_id"):
            store.upsert([dict(clauses()[0], doc_id="wrong-document")])


def test_existing_hybrid_index_migration_with_provenance(tmp_path):
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "fictional.txt"
    source.write_text(clauses()[0]["text"], encoding="utf-8")
    index = tmp_path / "legacy"
    pipeline = DocumentRetrievalPipeline(ModelConfig(index_dir=index, corpus_source="documents",
                                                     retrieval_model="local-hashing", enable_image_signal=False))
    pipeline.build_index(corpus)
    rows = [json.loads(line) for line in (index / "hybrid_snippets.jsonl").read_text(encoding="utf-8").splitlines()]
    sidecar = tmp_path / "provenance.jsonl"
    sidecar.write_text("\n".join(json.dumps({"record_id": row["record_id"],
                       "source_sha256": sha256(source.read_bytes()).hexdigest()}) for row in rows), encoding="utf-8")
    result = run_import(tmp_path / "db", rows, "--vectors", index / "hybrid_snippets_dense.npy",
                        "--manifest", index / "hybrid_embedding_manifest.json", "--provenance", sidecar)
    assert result["written"] == len(rows)
    with ClauseStore(tmp_path / "db") as store:
        evidence = store.search_clauses("dental crowns")["results"][0]["evidence"]
        assert evidence["record_id"] == rows[0]["record_id"]
        assert evidence["source_sha256"] == sha256(source.read_bytes()).hexdigest()
        assert evidence["text"] == rows[0]["text"]


def test_legacy_vectors_and_sparse_compatibility(tmp_path, monkeypatch):
    rows = clauses()
    embedder = EmbeddingRetriever("local-hashing")
    texts = [r["text"] for r in rows]
    vectors = embedder.embed_texts(texts)
    vectors_path, manifest_path = tmp_path / "legacy.npy", tmp_path / "legacy.manifest.json"
    np.save(vectors_path, vectors)
    manifest_path.write_text(json.dumps({"embedding_fingerprint": embedder.index_fingerprint()}), encoding="utf-8")
    assert run_import(tmp_path / "db", rows, "--vectors", vectors_path, "--manifest", manifest_path)["written"] == 2
    with ClauseStore(tmp_path / "db", writable=True) as store:
        def no_embedding(_):
            raise AssertionError("Migration must reuse supplied vectors")
        monkeypatch.setattr(store.embedder, "embed_texts", no_embedding)
        store.upsert([dict(rows[0], publisher="Updated fixture")], vectors=vectors[:1], fingerprint=embedder.index_fingerprint())
    sparse = SparseRetriever()
    old = sparse.build_index(texts, tmp_path / "sparse.json")
    assert sparse.search("dental crowns", old) == sparse.search("dental crowns", sparse.build(texts))


def test_real_mcp_stdio_restart_and_validation(tmp_path):
    """Import subprocess exits, then two independent MCP server processes query it."""
    db = tmp_path / "db"
    assert run_import(db, clauses())["written"] == 2
    assert run_import(db, clauses())["unchanged"] == 2
    parameters = StdioServerParameters(command=sys.executable,
        args=["-m", "src.insurerag_vlm.clause_mcp", "--store", str(db)], cwd=str(REPO))

    async def visit(check_errors):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {tool.name for tool in tools} == {"search_clauses", "read_evidence"}
                assert all(tool.annotations.readOnlyHint and not tool.annotations.openWorldHint for tool in tools)
                result = await session.call_tool("search_clauses", {"query": "dental crowns", "filters": {"jurisdiction": "AA"}})
                assert not result.isError
                evidence = result.structuredContent["results"][0]["evidence"]
                read = await session.call_tool("read_evidence", {"evidence_id": evidence["evidence_id"],
                                                                  "expected_sha256": evidence["record_sha256"]})
                assert read.structuredContent["evidence"] == evidence
                if check_errors:
                    for tool, args in [
                        ("read_evidence", {"evidence_id": "../../secret.txt"}),
                        ("read_evidence", {"evidence_id": "file:///C:/secret.txt"}),
                        ("read_evidence", {"evidence_id": "0" * 64, "path": "secret.txt"}),
                        ("search_clauses", {"query": "a", "top_k": "2"}),
                        ("search_clauses", {"query": "a", "top_k": True}),
                        ("search_clauses", {"query": "a", "filters": {"path": "secret.txt"}}),
                        ("search_clauses", {"query": "a", "top_k": 21}),
                        ("upsert", {"text": "forbidden"}),
                    ]:
                        assert (await session.call_tool(tool, args)).isError
                    missing = await session.call_tool("read_evidence", {"evidence_id": "0" * 64})
                    assert not missing.isError and missing.structuredContent["status"] == "not_found"
                    stale = await session.call_tool("read_evidence", {"evidence_id": evidence["evidence_id"], "expected_sha256": "0" * 64})
                    assert stale.structuredContent["status"] == "stale"
                return evidence

    async def exercise():
        return await asyncio.wait_for(visit(True), timeout=60), await asyncio.wait_for(visit(False), timeout=60)

    before, after = asyncio.run(exercise())
    assert before == after
