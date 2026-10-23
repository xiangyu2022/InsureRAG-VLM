"""Synchronous Agent adapter over a real, locally owned MCP stdio session."""
import asyncio
from contextlib import asynccontextmanager, ExitStack
import json
from pathlib import Path
import sys

from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .chroma_knowledge import canonical, normalize_record


class MCPFailure(RuntimeError):
    pass


class MCPClauseRetriever:
    """Fixed local server command, exactly two allowed tools, no model-selected paths.

    One session owns the store lock for the entire Agent command, including review
    validation and release. A new process revalidates evidence before approval.
    """
    def __init__(self, store, collection, *, timeout=10, filters=None, record_schema=1):
        self.store = str(Path(store).resolve())
        self.collection = collection
        self.timeout = timeout
        self.filters = filters
        self.record_schema = record_schema
        self.calls = []
        self._verified = {}

    @asynccontextmanager
    async def _session(self):
        params = StdioServerParameters(command=sys.executable, cwd=str(Path(__file__).resolve().parents[2]),
            args=['-m', 'src.insurerag_vlm.clause_mcp', '--store', self.store,
                  '--collection', self.collection, '--model', 'local-hashing', '--record-schema', str(self.record_schema)])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await asyncio.wait_for(session.initialize(), self.timeout)
                tools = await asyncio.wait_for(session.list_tools(), self.timeout)
                if {tool.name for tool in tools.tools} != {'search_clauses', 'read_evidence'}:
                    raise MCPFailure('Unexpected MCP tool set')
                self.session = session
                yield

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.portal = self.stack.enter_context(start_blocking_portal())
            self.stack.enter_context(self.portal.wrap_async_context_manager(self._session()))
            self._verified = {}
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self._verified = {}
        return self.stack.__exit__(*args)

    async def _call(self, name, arguments):
        if name not in {'search_clauses', 'read_evidence'}:
            raise MCPFailure('Tool is not allowlisted')
        result = await asyncio.wait_for(self.session.call_tool(name, arguments), self.timeout)
        self.calls.append({'tool': name, 'error': bool(result.isError)})
        if result.isError or not isinstance(result.structuredContent, dict):
            raise MCPFailure('MCP tool failed: ' + name)
        return result.structuredContent

    def _invoke(self, name, arguments):
        return self.portal.call(self._call, name, arguments)

    @staticmethod
    def _row(evidence):
        checked = normalize_record(evidence)
        if checked != evidence:
            raise ValueError('MCP evidence integrity mismatch')
        provenance = {key: value for key, value in evidence.items() if key != 'text'}
        if evidence.get('schema_version', 1) == 1:
            provenance['layout_status'] = {'status': 'unknown_legacy', 'extractor': 'legacy-compat-v1', 'needs_review': True}
        source = canonical({key: evidence[key] for key in ('evidence_id', 'record_sha256', 'citation')})
        return {'source': source, 'text': evidence['text'], 'reference': provenance}

    def search(self, query, top_k):
        result = self._invoke('search_clauses', {'query': query, 'top_k': top_k, 'filters': self.filters})
        if result.get('status') != 'ok' or result.get('embedding', {}).get('backend') != 'local-hashing' or result['embedding'].get('embedding_dim') != 512:
            raise ValueError('Unexpected search status or embedding contract')
        return [self._row(hit['evidence']) for hit in result['results']]

    def read_evidence(self, known):
        reference = known['reference']
        key = (reference['evidence_id'], reference['record_sha256'])
        if key not in self._verified:
            result = self._invoke('read_evidence', {'evidence_id': key[0], 'expected_sha256': key[1]})
            if result.get('status') != 'found':
                raise ValueError('MCP evidence unavailable: ' + str(result.get('status')))
            self._verified[key] = self._row(result['evidence'])
        actual = self._verified[key]
        compatible_legacy = ('record_id' not in reference and actual['reference'].get('schema_version', 1) == 1)
        metadata_matches = (all(actual['reference'].get(k) == v for k, v in reference.items()) if compatible_legacy
                            else actual['reference'] == reference)
        if not metadata_matches or actual['source'] != known['source']:
            raise ValueError('MCP evidence identity changed')
        if not actual['text'].startswith(known['text']):
            raise ValueError('MCP evidence text changed')
        return actual

    def validate_evidence(self, evidence):
        # Cached only during this live locked session; never trusted across restarts.
        for known in evidence.values():
            self.read_evidence(known)
