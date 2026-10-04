"""Controlled synthetic cross-page retrieval study. No LLM or legal accuracy claim."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from scripts.prepare_research_benchmark_v2 import PromptCapture
from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2

DEFAULT = ROOT/'data/benchmarks/graph_paths_v1'
ARMS = {'bm25': ('sparse_only', 'off'), 'dense': ('dense_only', 'off'),
        'hybrid': ('hybrid_text', 'off'), 'hybrid_explicit_graph': ('hybrid_text', 'explicit'),
        'hybrid_candidate_graph': ('hybrid_text', 'all')}
FAMILIES = [
    ('water_backup', 'basement pump failure', 'A maintenance inspection must have been completed before the event.', ['maintenance inspection', 'before']),
    ('garage_theft', 'bicycle theft from a locked garage', 'A police report must be filed within 48 hours of discovery.', ['police report', '48']),
    ('equipment', 'portable generator breakdown', 'The equipment must have a recorded service visit in the preceding year.', ['service visit', 'preceding year']),
    ('travel', 'trip interruption after illness', 'A treating clinician must provide a signed statement confirming inability to travel.', ['clinician', 'signed statement']),
    ('cyber', 'personal identity restoration', 'The claimant must complete identity verification before reimbursement.', ['identity verification', 'before']),
    ('pet', 'emergency treatment for a companion animal', 'An itemized invoice from a licensed veterinarian must be submitted.', ['itemized invoice', 'veterinarian']),
]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')


def write_rows(path, rows):
    path.write_text(''.join(v2.canonical(r)+'\n' for r in rows), encoding='utf8')


def build(folder):
    folder.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261001)
    pages, cases = [], []
    for n in range(30):
        family, topic, condition, keys = FAMILIES[n % len(FAMILIES)]
        packet = f'RESEARCH-{n+1:03d}'
        a, b, c = f'COV{n+1:03d}', f'COND{n+1:03d}', f'LIM{n+1:03d}'
        value = rng.randrange(17, 98) * 100
        rival = value + rng.randrange(10, 20)*100
        texts = [
            f'Section {a}: Coverage for {topic}. Eligibility conditions are incorporated by reference. See Section {b}.',
            f'Section {b}: {condition} For the financial cap, refer to Section {c}.',
            f'Section {c}: The maximum benefit is ${value:,} per incident. This amount applies only when the referenced eligibility condition is met.',
            f'Section OFFER{n}: An unissued marketing proposal for {topic} advertises a ${rival:,} payout cap without an eligibility condition. This proposal is not part of the issued coverage clause.',
            f'Section EXAMPLE{n}: An educational example of {topic} uses a $500 deductible and a $20,000 payout cap. Those illustrative amounts establish no issued benefit.',
            f'Section OTHER{n}: An unrelated property extension has a $12,000 maximum benefit. It is not incorporated into the coverage clause for {topic}.',
            f'Section CLAIM{n}: Submit a claim with receipts and a description of the incident. This administrative page establishes neither a payout cap nor coverage eligibility.',
            f'Section PRICE{n}: No annual premium, policy effective date, or claim settlement has been provided in this invented packet.',
            f'Section TERMS{n}: A benefit cap limits reimbursement. Eligibility describes conditions that must be satisfied. Examples and proposals do not amend issued terms.',
            f'Section EXCLUSION{n}: General exclusions for unrelated risks are omitted from this diagnostic. This page cannot establish any additional exception for {topic}.',
        ]
        split = 'dev' if n < 6 else 'test'
        for number, text in enumerate(texts, 1):
            pages.append({'record_id': f'rag_page::{packet}::p{number:04d}', 'record_type': 'page',
                          'doc_id': packet, 'packet_id': packet, 'policy_number': packet, 'page': number,
                          'source_file': packet+'.txt', 'citation': f'{packet}.txt#page={number}',
                          'source_origin': 'synthetic_graph_diagnostic', 'content_type': 'synthetic_policy_packet', 'text': text})
        definitions = [
            ('multi_hop', f'For {topic}, what maximum benefit and eligibility condition apply under the terms referenced by the coverage clause?', [1, 2, 3], f'${value:,}. {condition}', [[str(value)]]+[[k] for k in keys]),
            ('one_hop', f'Which eligibility condition is referenced by the coverage clause for {topic}?', [1, 2], condition, [[k] for k in keys]),
            ('direct', f'In Section {c}, what is the maximum benefit per incident?', [3], f'${value:,}.', [[str(value)]]),
            ('unsupported', f'What exact annual premium is charged for {topic} in this packet?', [], '', []),
        ]
        for category, question, evidence_pages, answer, answer_keys in definitions:
            cases.append({'id': packet+'_'+category, 'split': split, 'document_scope': [packet],
                          'document_family_id': family, 'category': category, 'question': question,
                          'answerable': bool(evidence_pages), 'reference_answer': answer, 'answer_keys': answer_keys,
                          'gold': [{'source': f'{packet}.txt#page={p}', 'page': p, 'evidence_span': texts[p-1]} for p in evidence_pages]})
    write_rows(folder/'pages.jsonl', pages)
    write_rows(folder/'cases.jsonl', cases)
    write(folder/'manifest.lock.json', {'version': 'graph_paths_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': 20261001, 'frozen_before_retrieval': True, 'packets': 30, 'page_records': len(pages),
        'cases': len(cases), 'dev_cases': 24, 'test_cases': 96, 'author': 'AI-authored synthetic diagnostic',
        'files': {n: v1.sha(folder/n) for n in ['pages.jsonl', 'cases.jsonl']},
        'limitations': ['Six known structural templates shared across dev/test; changed values are not independent scenario diversity.',
                       'Text records are not PDF pages. No customer data or legal authority.',
                       'The explicit-reference rule is intentionally exercised; this is mechanism evidence, not prevalence or real-policy performance.',
                       'Unsupported cases test absence handling only if a generation experiment is separately run; retrieval alone cannot score abstention.']})


def bootstrap(rows, left, right, key, draws=2000):
    grouped = defaultdict(list)
    for r in rows:
        if r['arm'] in [left, right] and r['answerable']:
            grouped[r['packet']].append(r)
    ids = sorted(grouped)
    rng = random.Random(42)
    values = []
    for _ in range(draws):
        picked = [r for _ in ids for r in grouped[rng.choice(ids)]]
        mean = lambda arm: sum(r[key] for r in picked if r['arm'] == arm)/sum(r['arm'] == arm for r in picked)
        values.append(mean(right)-mean(left))
    values.sort()
    return {'low': values[int(.025*draws)], 'high': values[int(.975*draws)], 'draws': draws,
            'unit': 'packet', 'limitation': 'Conditional on six shared synthetic templates; not generalization uncertainty.'}


def run(args):
    folder, output = args.fixture, args.output
    lock = json.loads((folder/'manifest.lock.json').read_text())
    for name, digest in lock['files'].items():
        if v1.sha(folder/name) != digest:
            raise ValueError('Frozen fixture changed: '+name)
    pages = v1.read_jsonl(folder/'pages.jsonl')
    cases = [c for c in v1.read_jsonl(folder/'cases.jsonl') if c['split'] == args.split]
    output.mkdir(parents=True, exist_ok=False)
    code = {**v2.source_hashes(ROOT), 'scripts/graph_retrieval_study.py': v1.sha(Path(__file__))}
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'fixture_sha256': v1.sha(folder/'manifest.lock.json'),
                'split': args.split, 'arms': ARMS, 'top_k': 3, 'context_chars': 3600, 'max_page_chars': 2400,
                'code_sha256': code, 'generation_calls': 0, 'retrieval_model': args.retrieval_model,
                'graph_seed_pages': 4, 'graph_max_hops': 2, 'graph_max_expansions': 8}
    write(output/'protocol.json', protocol)
    results, shared_retriever, edge_inventory = [], None, []
    for packet in sorted({c['document_scope'][0] for c in cases}):
        selected = [p for p in pages if p['doc_id'] == packet]
        corpus = output/'artifacts'/packet/'corpus'
        corpus.mkdir(parents=True)
        write_rows(corpus/'rag_pages.jsonl', selected)
        write_rows(corpus/'rag_snippets.jsonl', [{**p, 'record_id': p['record_id']+'::snippet', 'parent_page_id': p['record_id']} for p in selected])
        config = ModelConfig(retrieval_model=args.retrieval_model, use_hf_api=False, retrieval_mode='hybrid_text',
            vlm_model='local-extractive', corpus_source='curated', curated_dataset_dir=corpus,
            index_dir=output/'artifacts'/packet/'index', enable_image_signal=False,
            max_answer_pages=3, max_retrievals=3, max_context_chars=3600, max_page_chars=2400)
        pipeline = DocumentRetrievalPipeline(config)
        pipeline.vlm_client = PromptCapture()
        if shared_retriever is not None:
            pipeline.retriever = shared_retriever
        pipeline.build_index(corpus)
        shared_retriever = pipeline.retriever
        edges = pipeline._ensure_indices(corpus)['graph_edges']
        edge_inventory.extend(edges)
        for case in [c for c in cases if c['document_scope'] == [packet]]:
            for arm, (retrieval_mode, graph_mode) in ARMS.items():
                pipeline.config.retrieval_mode, pipeline.config.graph_mode = retrieval_mode, graph_mode
                start = time.perf_counter()
                response = pipeline.query_with_ranking(case['question'], corpus, top_k=3)
                seconds = time.perf_counter()-start
                ranking = response['source_ranking']
                sources = [p['source'] for p in ranking]
                gold_sources = {g['source'] for g in case['gold']}
                sections = v1.supplied_source_sections(response['retrieval_context'])
                evidence = [any(v1.whitespace(g['evidence_span']) in v1.whitespace(s) for s in sections.get(g['source'], [])) for g in case['gold']]
                results.append({'id': case['id'], 'packet': packet, 'family': case['document_family_id'],
                    'category': case['category'], 'answerable': case['answerable'], 'arm': arm,
                    'page_recall': len(gold_sources.intersection(sources))/len(gold_sources) if gold_sources else None,
                    'all_gold_pages': gold_sources.issubset(sources) if gold_sources else None,
                    'complete_evidence_in_context': all(evidence) if evidence else None,
                    'context': response['retrieval_context'], 'ranking': ranking, 'seconds': seconds})
        print(json.dumps({'packet': packet, 'completed_queries': len(results)}), flush=True)
    write_rows(output/'predictions.jsonl', results)
    write_rows(output/'graph_edges.jsonl', edge_inventory)
    metrics = {}
    for arm in ARMS:
        group = [r for r in results if r['arm'] == arm and r['answerable']]
        metrics[arm] = {'supported_cases': len(group), **{k: sum(r[k] for r in group)/len(group) for k in ['page_recall', 'all_gold_pages', 'complete_evidence_in_context']},
                       'by_category': {cat: {'n': sum(r['category'] == cat for r in group),
                           'complete_evidence': sum(r['complete_evidence_in_context'] for r in group if r['category'] == cat)/sum(r['category'] == cat for r in group)}
                           for cat in sorted({r['category'] for r in group})}}
    delta = metrics['hybrid_explicit_graph']['complete_evidence_in_context'] - metrics['hybrid']['complete_evidence_in_context']
    summary = {'split': args.split, 'cases': len(cases), 'queries': len(results), 'generation_calls': 0,
               'edge_types': dict(Counter(e['relation_status'] for e in edge_inventory)), 'arms': metrics,
               'paired_complete_evidence_delta': {'difference': delta, 'interval': bootstrap(results, 'hybrid', 'hybrid_explicit_graph', 'complete_evidence_in_context')},
               'embedding_fingerprint': shared_retriever.index_fingerprint(),
               'limitations': lock['limitations']+['Full-span evidence completeness is stricter than semantic sufficiency.',
                    'No generation was run here; unsupported cases have no correctness metric.',
                    'Query times include different cache/order effects and are not a serving-latency benchmark.']}
    write(output/'summary.json', summary)
    if code != {**v2.source_hashes(ROOT), 'scripts/graph_retrieval_study.py': v1.sha(Path(__file__))}:
        raise RuntimeError('Code changed during study')
    print(json.dumps({'complete': True, 'summary': metrics, 'delta': delta}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'run'])
    p.add_argument('--fixture', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--retrieval-model', default='local-hashing')
    args = p.parse_args()
    if args.action == 'build':
        build(args.fixture)
    elif args.output is None:
        p.error('--output required for run')
    else:
        run(args)
