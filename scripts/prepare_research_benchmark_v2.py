#!/usr/bin/env python3
"""Freeze retrieval and actual prompts once, without any model-generation call."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2


class PromptCapture:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return ''

    def answer_trace(self, **kwargs):
        return {'generation_used': False, 'answer_backend': 'offline_prompt_capture'}


def prepare(args):
    args.benchmark = Path(args.benchmark)
    args.corpus = Path(args.corpus) if args.corpus else args.benchmark / 'corpus'
    for name in ('top_k', 'max_context_chars', 'num_ctx', 'num_predict', 'bootstrap'):
        if getattr(args, name) <= 0:
            raise ValueError(name + ' must be positive')
    if args.limit is not None and (args.limit <= 0 or args.split != 'dev'):
        raise ValueError('A positive limit is allowed only for a development smoke')
    lock, items, pages, snippets, audit = v2.verify_fixture(args.benchmark, args.corpus)
    if args.validate_only:
        print(json.dumps(audit, indent=2))
        return audit
    if args.output is None:
        raise ValueError('--output is required for preparation')
    if args.retrieval_model != 'local-hashing' and not Path(args.retrieval_model).is_dir():
        raise ValueError('Use explicit local-hashing or an existing local embedding checkpoint')
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    selected = [row for row in items if row['split'] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    modes = ['retrieved', 'oracle'] if args.mode == 'both' else [args.mode]
    capture = PromptCapture()
    cache, prepared, fingerprints = {}, [], {}
    started = time.perf_counter()
    for item in selected:
        for mode in modes:
            doc_id = item['document_scope'][0]
            row = {key: item[key] for key in ('id', 'split', 'question', 'answerable', 'document_scope',
                                              'document_family_id', 'category', 'fact_id')}
            row.update({'document_id': doc_id, 'mode': mode, 'case': item,
                        'annotation_sha256': v2.digest_object(item)})
            query_started = time.perf_counter()
            if mode == 'retrieved':
                if doc_id not in cache:
                    cache[doc_id] = v1.make_pipeline(args, [doc_id], pages, snippets, capture)
                pipeline, corpus, page_count = cache[doc_id]
                pipeline.config.graph_mode = args.graph_mode
                pipeline.config.max_page_chars = args.max_page_chars
                fingerprints[doc_id] = pipeline.retriever.index_fingerprint()
                capture.prompt = None
                query_started = time.perf_counter()
                result = pipeline.query_with_ranking(item['question'], corpus, top_k=args.top_k)
                context = result.get('retrieval_context', '')
                prompt = capture.prompt or v1.PROMPT_TEMPLATE.format(context=context, question=item['question'])
                ranking = result['source_ranking']
                row.update({'ranked_sources': [page['source'] for page in ranking],
                            'source_ranking': ranking, 'scoped_page_count': page_count,
                            'retrieval_status': 'ranked_context' if ranking else 'no_candidates_empty_context'})
            else:
                context, sources, context_type = v1.oracle_context(item, pages)
                prompt = v1.PROMPT_TEMPLATE.format(context=context, question=item['question'])
                row.update({'ranked_sources': [], 'oracle_sources': sources, 'oracle_context_type': context_type})
            row.update({'prompt': prompt, 'prompt_sha256': v1.sha(prompt),
                        'preparation_query_seconds': time.perf_counter() - query_started})
            sections = v1.supplied_source_sections(v1.context_from_prompt(prompt))
            row['gold_evidence_in_context'] = any(v1.whitespace(gold['evidence_span']) in v1.whitespace(section)
                for gold in item['gold'] for section in sections.get(gold['source'], [])) if item['answerable'] else None
            prepared.append(row)
    chosen_docs = {item['document_scope'][0] for item in selected}
    selected_pages = [page for page in pages if page['doc_id'] in chosen_docs]
    for name, rows in [('inputs.jsonl', prepared), ('pages.jsonl', selected_pages)]:
        (args.output / name).write_text(''.join(v2.canonical(row) + '\n' for row in rows), encoding='utf-8')
    protocol = {'protocol_version': v2.PROTOCOL_VERSION, 'score_version': v2.SCORE_VERSION,
        'prepared_utc': datetime.now(timezone.utc).isoformat(), 'preparation_generation_calls': 0,
        'benchmark_lock_sha256': v1.sha(args.benchmark / 'manifest.lock.json'), 'benchmark_lock': lock,
        'fixture_audit': audit, 'split': args.split, 'case_count': len(selected),
        'partial_dev_smoke': bool(args.limit), 'mode': args.mode,
        'request_schedule': [{'id': row['id'], 'mode': row['mode']} for row in prepared],
        'prepared_files': {name: v1.sha(args.output / name) for name in ('inputs.jsonl', 'pages.jsonl')},
        'prompt_contract': {'system': v1.SYSTEM, 'template': v1.PROMPT_TEMPLATE, 'schema': v1.SCHEMA},
        'prompt_contract_sha256': v2.digest_object({'system': v1.SYSTEM, 'template': v1.PROMPT_TEMPLATE, 'schema': v1.SCHEMA}),
        'code_sha256': v2.source_hashes(ROOT), 'retrieval_fingerprints': fingerprints,
        'retrieval_config': {key: getattr(args, key) for key in ('retrieval_model', 'retrieval_mode', 'graph_mode', 'top_k', 'max_page_chars', 'max_context_chars')},
        'generation_options': {'temperature': 0, 'seed': 42, 'num_ctx': args.num_ctx, 'num_predict': args.num_predict,
                               'top_k': 40, 'top_p': 1.0, 'repeat_penalty': 1.0, 'presence_penalty': 0.0},
        'thinking': False, 'statistics': {'bootstrap_draws': args.bootstrap, 'bootstrap_seed': 42, 'cluster_unit': 'document_id'},
        'preparation_wall_seconds': time.perf_counter() - started,
        'evaluation_path': 'Frozen actual hybrid retrieval/context followed by direct JSON model contract; no application repair.',
        'limitations': ['AI-authored cases and exact-key/quote checks are not expert semantic accuracy.',
                        'Document/family split separation does not establish absence from previous SFT or public pretraining.',
                        'Oracle evidence is an optional diagnostic and is not the primary retrieved result.']}
    v1.write_json(protocol, args.output / 'protocol.json')
    print(json.dumps({'status': 'prepared', 'cases': len(selected), 'requests': len(prepared),
                      'protocol_sha256': v1.sha(args.output / 'protocol.json'), 'generation_calls': 0}, indent=2))
    return protocol


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, default=ROOT / 'data/benchmarks/research_v2')
    p.add_argument('--corpus', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--mode', choices=['retrieved', 'oracle', 'both'], default='retrieved')
    p.add_argument('--retrieval-model', default='local-hashing')
    p.add_argument('--graph-mode', choices=['off', 'explicit', 'all'], default='explicit')
    p.add_argument('--retrieval-mode', choices=['hybrid_text', 'hybrid_multimodal', 'dense_only'], default='hybrid_text')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--max-context-chars', type=int, default=8000)
    p.add_argument('--max-page-chars', type=int, default=2400)
    p.add_argument('--num-ctx', type=int, default=8192)
    p.add_argument('--num-predict', type=int, default=384)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--limit', type=int)
    p.add_argument('--validate-only', action='store_true')
    return p


if __name__ == '__main__':
    prepare(parser().parse_args())
