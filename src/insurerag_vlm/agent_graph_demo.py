"""Offline LangGraph review demo using fictional fixtures, not model evaluation."""
import argparse
import json
from pathlib import Path

from .agent_demo import DOCUMENTS, QUESTIONS, SyntheticController
from .agent_draft import LocalModelController
from .agent_graph import ReviewAgent, digest


class FixtureRetriever:
    """Fixed synthetic evidence with stable identities across process restarts."""
    def search(self, query, top_k):
        return [{'source': 'synthetic://' + name, 'text': text}
                for name, text in DOCUMENTS.items()][:top_k]


class LazyLocalController:
    """A saved review must remain usable while the generation service is down."""
    context_tokens = 8192

    def __init__(self, model, endpoint, expected_digest):
        self.settings = (model, endpoint, expected_digest)
        self.controller = None

    def next_action(self, *args):
        if self.controller is None:
            self.controller = LocalModelController(*self.settings)
        return self.controller.next_action(*args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'resume'])
    parser.add_argument('--db', type=Path, default=Path('.agent-state/review.sqlite'))
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--scenario', choices=list(QUESTIONS), default='ordinary')
    parser.add_argument('--decision', type=Path, help='Explicit JSON review file; no default approval')
    parser.add_argument('--controller', choices=['scripted', 'ollama'], default='scripted')
    parser.add_argument('--model', default='ollama:qwen3.5:4b')
    parser.add_argument('--expected-digest')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    args = parser.parse_args(argv)
    if (args.command == 'resume') != bool(args.decision):
        parser.error('Only resume requires --decision')
    if args.controller == 'ollama' and not args.expected_digest:
        parser.error('Local generation requires an explicit --expected-digest; no downloads')
    controller = (SyntheticController(args.scenario) if args.controller == 'scripted'
                  else LazyLocalController(args.model, args.ollama_url, args.expected_digest))
    runtime = digest({'fixtures': DOCUMENTS, 'scenario': args.scenario,
                      'controller': args.controller, 'model': args.model if args.controller == 'ollama' else None,
                      'digest': args.expected_digest, 'endpoint': args.ollama_url})
    args.db.parent.mkdir(parents=True, exist_ok=True)
    agent = ReviewAgent(args.db, FixtureRetriever(), controller, runtime_id=runtime)
    if args.command == 'start':
        result = agent.start(args.thread_id, QUESTIONS[args.scenario])
    elif args.command == 'status':
        result = agent.status(args.thread_id)
    else:
        from .agent_draft import InvalidAction
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise InvalidAction('Duplicate review JSON key')
                result[key] = value
            return result
        if args.decision.stat().st_size > 40000:
            parser.error('Review file exceeds 40000 bytes')
        request = json.loads(args.decision.read_text(encoding='utf8'), object_pairs_hook=unique)
        result = agent.resume(args.thread_id, request)
    print(json.dumps({'synthetic_only': True, 'holdout_data_used': False,
                      'controller': args.controller, 'model_quality_validated': False,
                      'result': result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
