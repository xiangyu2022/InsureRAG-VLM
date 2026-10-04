"""Global retrieval over all expanded public sources; no gold document filtering."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2
from scripts.prepare_research_benchmark_v2 import PromptCapture
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


def run(args):
    fixture = ROOT/'data/benchmarks/research_v2'
    _, cases, _, _, _ = v2.verify_fixture(fixture, fixture/'corpus')
    cases = [c for c in cases if c['split'] == args.split]
    corpus = args.corpus
    manifest = json.loads((corpus/'manifest.json').read_text(encoding='utf8'))
    for name, digest in manifest['files'].items():
        if v1.sha(corpus/name) != digest:
            raise ValueError('Expanded corpus checksum mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    config = ModelConfig(retrieval_model=args.retrieval_model, use_hf_api=False, vlm_model='local-extractive',
        corpus_source='curated', curated_dataset_dir=corpus, index_dir=args.output/'index',
        retrieval_mode='hybrid_text', enable_image_signal=False, max_retrievals=3, max_answer_pages=3,
        max_page_chars=2400, max_context_chars=8000)
    pipeline = DocumentRetrievalPipeline(config)
    pipeline.vlm_client = PromptCapture()
    pipeline.build_index(corpus)
    edges = pipeline._ensure_indices(corpus)['graph_edges']
    source_hashes = {**v2.source_hashes(ROOT), 'scripts/eval_expanded_retrieval.py': v1.sha(Path(__file__))}
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'generation_calls': 0,
        'scope': f'All {manifest["counts"]["documents"]} source documents searched together; no per-question gold document filter',
        'split': args.split, 'cases': len(cases), 'code_sha256': source_hashes,
        'benchmark_lock_sha256': v1.sha(fixture/'manifest.lock.json'),
        'corpus_manifest_sha256': v1.sha(corpus/'manifest.json'), 'corpus_counts': manifest['counts'],
        'embedding': pipeline.retriever.index_fingerprint(), 'top_k': 3, 'max_page_chars': 2400, 'max_context_chars': 8000,
        'arms': ['off', 'explicit', 'all'], 'edge_status_counts': dict(Counter(e['relation_status'] for e in edges)),
        'edge_relation_counts': dict(Counter(e['relation'] for e in edges)),
        'limitations': ['Questions identify their archived source in natural language; global retrieval is not blind source discovery.',
            'Consumer guides rarely encode actual issued-policy links; absence of graph gain is a valid result.',
            'Page hit on a long HTML record does not show that the necessary evidence reached the prompt.']}
    v1.write_json(protocol, args.output/'protocol.json')
    results = []
    for i, case in enumerate(cases, 1):
        for mode in ['off', 'explicit', 'all']:
            pipeline.config.graph_mode = mode
            start = time.perf_counter()
            response = pipeline.query_with_ranking(case['question'], corpus, top_k=3)
            ranking = response['source_ranking']
            sources = [r['source'] for r in ranking]
            gold = {r['source'] for r in case['gold']}
            sections = v1.supplied_source_sections(response['retrieval_context'])
            complete = all(any(v1.whitespace(g['evidence_span']) in v1.whitespace(t) for t in sections.get(g['source'], [])) for g in case['gold'])
            result = {'id': case['id'], 'document_id': case['document_scope'][0], 'category': case['category'],
                      'answerable': case['answerable'], 'graph_mode': mode, 'question': case['question'],
                      'gold_sources': sorted(gold), 'ranked_sources': sources,
                      'page_hit': bool(gold.intersection(sources)) if gold else None,
                      'mrr': next((1/(j+1) for j, source in enumerate(sources) if source in gold), 0) if gold else None,
                      'complete_evidence_in_context': complete if gold else None,
                      'context': response['retrieval_context'], 'seconds': time.perf_counter()-start}
            results.append(result)
            with (args.output/'predictions.jsonl').open('a', encoding='utf8') as out:
                out.write(v2.canonical(result)+'\n')
        if i % 20 == 0:
            print(json.dumps({'completed_cases': i, 'scheduled': len(cases)}), flush=True)
    summary = {}
    for mode in ['off', 'explicit', 'all']:
        group = [r for r in results if r['answerable'] and r['graph_mode'] == mode]
        summary[mode] = {'supported_cases': len(group), **{key: sum(r[key] for r in group)/len(group)
                          for key in ['page_hit', 'mrr', 'complete_evidence_in_context']}}
    v1.write_json({'arms': summary, 'generation_calls': 0, 'queries': len(results),
                  'protocol': protocol, 'predictions_sha256': v1.sha(args.output/'predictions.jsonl')}, args.output/'summary.json')
    if source_hashes != {**v2.source_hashes(ROOT), 'scripts/eval_expanded_retrieval.py': v1.sha(Path(__file__))}:
        raise RuntimeError('Evaluation source changed')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=ROOT/'data/research_corpus/expanded_public_v1')
    p.add_argument('--retrieval-model', default='local-hashing')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    run(p.parse_args())
