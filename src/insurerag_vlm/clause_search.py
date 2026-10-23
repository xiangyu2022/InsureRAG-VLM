"""Explicit project retrieval entry point backed by the same Chroma MCP store."""
import json

from .chroma_knowledge import ClauseStore
from .clause_store_cli import store_options


def search_from_args(args):
    filters = json.loads(args.filters) if args.filters else None
    with ClauseStore(args.store, **store_options(args)) as store:
        return store.search_clauses(args.question, args.top_k, filters)
