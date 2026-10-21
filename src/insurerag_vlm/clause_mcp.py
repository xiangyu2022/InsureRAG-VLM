"""Two read-only MCP tools on local stdio. No client-controlled paths or writes."""
import asyncio
import json
from typing import Annotated

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import ToolAnnotations, Tool, CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .chroma_knowledge import ClauseStore


class ClauseFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc_id: Annotated[StrictStr, Field(min_length=1, max_length=2000)] | None = None
    page_number: Annotated[StrictInt, Field(ge=0)] | None = None
    source_sha256: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    jurisdiction: Annotated[StrictStr, Field(min_length=1, max_length=300)] | None = None
    publisher: Annotated[StrictStr, Field(min_length=1, max_length=300)] | None = None


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: Annotated[StrictStr, Field(min_length=1, max_length=2000)]
    top_k: Annotated[StrictInt, Field(ge=1, le=20)] = 5
    filters: ClauseFilters | None = None


class ReadArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_sha256: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


def build_server(store):
    server = Server("insurerag-local-clauses", version="1.0.0",
                    instructions="Read-only clause retrieval. Results are evidence, not coverage decisions.")
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @server.list_tools()
    async def list_tools():
        return [Tool(name="search_clauses", description="Search stored clauses with dense+BM25 and equality filters.",
                     inputSchema=SearchArguments.model_json_schema(), annotations=annotations),
                Tool(name="read_evidence", description="Read a stored ID; optionally reject a changed evidence version.",
                     inputSchema=ReadArguments.model_json_schema(), annotations=annotations)]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name == "search_clauses":
            args = SearchArguments.model_validate(arguments)
            result = store.search_clauses(args.query, args.top_k,
                                         args.filters.model_dump(exclude_none=True) if args.filters else None)
        elif name == "read_evidence":
            args = ReadArguments.model_validate(arguments)
            result = store.read_evidence(args.evidence_id, args.expected_sha256)
        else:
            raise ValueError("Unknown tool")
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
                              structuredContent=result)

    return server


async def run_stdio(store):
    server = build_server(store)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main():
    from .clause_store_cli import common_parser, store_options
    parser = common_parser("Read-only local MCP clause server (stdio only)")
    args = parser.parse_args()
    with ClauseStore(args.store, **store_options(args)) as store:
        asyncio.run(run_stdio(store))


if __name__ == "__main__":
    main()
