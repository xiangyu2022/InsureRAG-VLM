"""Operator-only import; no ingestion function is exposed through MCP."""
import argparse
import json
from pathlib import Path

import numpy as np

from .chroma_knowledge import ClauseStore


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--store", type=Path, default=Path(".knowledge/chroma"))
    parser.add_argument("--collection", default="insurance_clauses")
    parser.add_argument("--model", default="local-hashing", help="local-hashing or existing local model directory; no downloads")
    parser.add_argument("--pooling", choices=("auto", "cls", "mean"), default="auto")
    parser.add_argument("--query-instruction", default="")
    parser.add_argument("--max-length", type=int)
    return parser


def store_options(args):
    return {key: getattr(args, key) for key in ("collection", "model", "pooling", "query_instruction", "max_length")}


def read_records(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Records must be a JSON array or JSONL")
    return data


def main():
    parser = common_parser("Import provenance-bound clauses into local Chroma")
    parser.add_argument("--records", type=Path, required=True, help="Trusted clause JSON/JSONL, including source SHA-256")
    parser.add_argument("--vectors", type=Path, help="Optional legacy .npy vectors, in exactly the same record order")
    parser.add_argument("--manifest", type=Path, help="Required with --vectors: existing embedding fingerprint manifest")
    parser.add_argument("--provenance", type=Path, help="Optional JSONL enrichment keyed by record_id; never overwrites existing fields")
    args = parser.parse_args()
    if bool(args.vectors) != bool(args.manifest):
        parser.error("--vectors and --manifest must be supplied together")
    records = read_records(args.records)
    if args.provenance:
        additions = read_records(args.provenance)
        by_id = {row["record_id"]: row for row in additions}
        if len(by_id) != len(additions) or not set(by_id) <= {r.get("record_id") for r in records}:
            raise ValueError("Duplicate or unknown provenance record_id")
        for row in records:
            for key, value in by_id.get(row.get("record_id"), {}).items():
                if key in row and row[key] != value:
                    raise ValueError("Provenance sidecar conflicts with existing record")
                row[key] = value
    vectors, fingerprint = None, None
    if args.vectors:
        vectors = np.load(args.vectors, allow_pickle=False)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        fingerprint = manifest.get("embedding_fingerprint", manifest)
    with ClauseStore(args.store, writable=True, **store_options(args)) as store:
        result = store.upsert(records, vectors=vectors, fingerprint=fingerprint)
        print(json.dumps({**result, "collection": args.collection, "dimension": store.dimension,
                          "embedding": store.fingerprint}, ensure_ascii=False))


if __name__ == "__main__":
    main()
