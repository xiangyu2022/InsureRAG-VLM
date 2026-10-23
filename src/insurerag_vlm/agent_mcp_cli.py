"""Local Chroma/MCP -> LangGraph -> mandatory operator approval. No publishing."""
import argparse
import json
from pathlib import Path

from .agent_draft import Limits
from .agent_graph import ReviewAgent, digest
from .agent_graph_demo import LazyLocalController
from .agent_mcp import MCPClauseRetriever


LIMITATIONS = ('Archived DOL guidance, not a personal policy determination; current applicability has not been '
               'revalidated. Retrieval uses 512-dimensional local-hashing, not BGE.')


class EvidenceQuoteController:
    """Deterministic plumbing driver: quotes evidence, does not claim to answer correctly."""
    context_tokens = 16384
    def __init__(self, refusal=False):
        self.refusal = refusal

    def next_action(self, system, payload, output_tokens, timeout):
        state = json.loads(payload)
        if self.refusal or not state['candidates']:
            value = {'action': 'ABSTAIN', 'parameters': {'reason': 'The deterministic driver cannot establish the requested conclusion from historical guidance.'},
                     'gaps': ['No validated model judgement or current policy applicability is available.']}
        elif not state['evidence']:
            value = {'action': 'READ', 'parameters': {'ids': [next(iter(state['candidates']))]},
                     'gaps': ['Read source evidence through MCP.']}
        else:
            eid, evidence = next(iter(state['evidence'].items()))
            quote = evidence['text'].strip()[:240]
            value = {'action': 'ANSWER', 'parameters': {'claims': [
                {'text': 'Unapproved historical-source excerpt for review only: ' + quote,
                 'citations': [{'id': eid, 'quote': quote}]}]}, 'gaps': []}
        return json.dumps(value), {}


def read_decision(path):
    if path.stat().st_size > 40000:
        raise ValueError('Review file exceeds 40000 bytes')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate review JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'resume'])
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--question')
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--db', type=Path, default=Path('.agent-state/mcp-review.sqlite'))
    parser.add_argument('--store', type=Path, default=Path('.knowledge/dol-aca-20261001/db'))
    parser.add_argument('--collection', default='dol_aca_20261001')
    parser.add_argument('--controller', choices=['mock-excerpt', 'mock-refusal', 'ollama'], default='mock-excerpt')
    parser.add_argument('--model', default='ollama:qwen3.5:4b')
    parser.add_argument('--expected-digest')
    args = parser.parse_args(argv)
    if (args.command == 'start') != bool(args.question):
        parser.error('Only start requires --question')
    if (args.command == 'resume') != bool(args.decision):
        parser.error('Only resume requires --decision')
    if args.controller == 'ollama' and not args.expected_digest:
        parser.error('Ollama requires the existing local model digest; no downloads')
    controller = (LazyLocalController(args.model, 'http://localhost:11434', args.expected_digest)
                  if args.controller == 'ollama' else EvidenceQuoteController(args.controller == 'mock-refusal'))
    runtime = digest({'integration': 'mcp-review-v1', 'store': str(args.store.resolve()), 'collection': args.collection,
                      'embedding': 'local-hashing-512', 'controller': args.controller,
                      'model': args.model if args.controller == 'ollama' else None,
                      'model_digest': args.expected_digest, 'limitations': LIMITATIONS})
    limits = Limits(top_k=2, evidence_chars=1600, context_tokens=16384, token_budget=100000)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    # Status needs only SQLite; rejection also works while MCP/model are unavailable.
    decision = read_decision(args.decision) if args.decision else None
    def run(retriever):
        agent = ReviewAgent(args.db, retriever, controller, runtime_id=runtime, limits=limits, review_abstentions=True)
        if args.command == 'start':
            return agent.start(args.thread_id, args.question)
        if args.command == 'status':
            return agent.status(args.thread_id)
        return agent.resume(args.thread_id, decision)
    if args.command == 'status' or (decision and decision.get('action') == 'reject'):
        result = run(None)
        calls = []
    else:
        with MCPClauseRetriever(args.store, args.collection) as retriever:
            result = run(retriever)
            calls = retriever.calls
    print(json.dumps({'controller': args.controller, 'model_quality_validated': False,
                      'limitations': LIMITATIONS, 'mcp_calls_this_command': calls, 'result': result},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
