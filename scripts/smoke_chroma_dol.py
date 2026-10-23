"""Read-only real-cache smoke, not benchmark scoring or generated-answer evaluation."""
import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.clause_snapshot import SNAPSHOT_SHA256


async def verify(args, sources):
    params = StdioServerParameters(command=sys.executable, cwd=str(ROOT), args=[
        "-m", "src.insurerag_vlm.clause_mcp", "--store", str(args.store.resolve()),
        "--collection", args.collection, "--model", "local-hashing"])
    cases = [("prescription drug health care spending reporting", {"doc_id": "dol_html_aca-part-56"}),
             ("contraceptive services coverage", None)]
    results = []
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            assert {tool.name for tool in (await session.list_tools()).tools} == {"search_clauses", "read_evidence"}
            for query, filters in cases:
                response = await session.call_tool("search_clauses", {"query": query, "top_k": 3, "filters": filters})
                assert not response.isError
                hits = response.structuredContent["results"]
                assert hits
                verified = []
                for hit in hits:
                    evidence = hit["evidence"]
                    read = await session.call_tool("read_evidence", {"evidence_id": evidence["evidence_id"],
                                                                     "expected_sha256": evidence["record_sha256"]})
                    assert not read.isError and read.structuredContent["status"] == "found"
                    assert read.structuredContent["evidence"] == evidence
                    source = sources[evidence["doc_id"]]
                    assert evidence["text"] == source["text"][evidence["char_start"]:evidence["char_end"]]
                    assert evidence["source_sha256"] == source["html_sha256"]
                    assert evidence["source_text_sha256"] == source["text_sha256"]
                    assert evidence["snapshot_sha256"] == SNAPSHOT_SHA256
                    if filters:
                        assert evidence["doc_id"] == filters["doc_id"]
                    verified.append({key: evidence[key] for key in ("evidence_id", "record_sha256", "doc_id", "char_start", "char_end")})
                results.append({"query": query, "filters": filters, "verified": verified})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/research_corpus/dol_additional_v1/records.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Private receipt path")
    args = parser.parse_args()
    before = args.snapshot.read_bytes()
    assert sha256(before).hexdigest() == SNAPSHOT_SHA256
    sources = {row["id"]: row for row in map(json.loads, before.decode().splitlines())}
    async def exercise():
        first = await asyncio.wait_for(verify(args, sources), timeout=60)
        second = await asyncio.wait_for(verify(args, sources), timeout=60)
        assert first == second
        return first
    checked = asyncio.run(exercise())
    # Exercise the public project entry point after both MCP server processes exit.
    command = [sys.executable, "main.py", "knowledge-search", checked[0]["query"], "--store", str(args.store.resolve()),
               "--collection", args.collection, "--top-k", "3", "--filters", json.dumps(checked[0]["filters"])]
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    actual = [hit["evidence"]["evidence_id"] for hit in json.loads(result.stdout)["results"]]
    assert actual == [row["evidence_id"] for row in checked[0]["verified"]]
    assert args.snapshot.read_bytes() == before
    receipt = {"status": "passed", "collection": args.collection, "source_documents": len(sources),
               "embedding": "local-hashing", "dimension": 512, "server_processes": 2,
               "search_calls": 4, "read_calls": 12, "restart_evidence_stable": True,
               "source_offsets_verified": True, "project_cli_matches_mcp": True, "source_bytes_unchanged": True,
               "model_quality_evaluated": False, "checks": checked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "checks"}))


if __name__ == "__main__":
    main()
