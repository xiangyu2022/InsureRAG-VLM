"""Label-free candidate scoring. Selection/scoring metrics are a separate command."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha, verify_fixture
from scripts.eval_insuranceqa_scale import SparseBM25, ranked
from src.insurerag_vlm.reranker import LocalCrossEncoder


def run(args):
    verify_fixture(args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    answers = sorted(read_jsonl(args.fixture/'answers.jsonl'), key=lambda a: int(a['id']))
    cases = read_jsonl(args.fixture/f'{args.split}.jsonl')
    # Discard all annotations before candidate construction or neural scoring.
    cases = [{'id': c['id'], 'question': c['question']} for c in cases]
    dense = np.load(args.baseline/'answer_embeddings.npy')
    queries = np.load(args.baseline/'query_embeddings.npy')
    offset = 0 if args.split == 'valid' else len(read_jsonl(args.fixture/'valid.jsonl'))
    if dense.shape[0] != len(answers) or len(queries[offset:offset+len(cases)]) != len(cases):
        raise ValueError('Cached embedding row counts do not match frozen fixture')
    old_protocol = json.loads((args.baseline/'protocol.json').read_text())
    if old_protocol['fixture_lock_sha256'] != sha(args.fixture/'manifest.lock.json'):
        raise ValueError('Cached embeddings belong to another fixture')
    import torch
    torch.set_num_threads(4)
    threadpool_limits(4)
    reranker = LocalCrossEncoder(args.model, args.device, args.batch_size, 512)
    files = [Path(__file__), ROOT/'src/insurerag_vlm/reranker.py',
             ROOT/'scripts/eval_insuranceqa_scale.py', ROOT/'scripts/prepare_insuranceqa.py']
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'split': args.split,
        'questions': len(cases), 'answer_count': len(answers), 'candidate_rule': 'Union of BGE top 100 and positive-score BM25 top 100',
        'candidate_limit': 200, 'label_injection': False, 'gold_used_for_scoring': False,
        'fixture_lock_sha256': sha(args.fixture/'manifest.lock.json'), 'code_sha256': code,
        'baseline_protocol_sha256': sha(args.baseline/'protocol.json'),
        'embedding_cache_sha256': {name: sha(args.baseline/name) for name in ['answer_embeddings.npy', 'query_embeddings.npy']},
        'embedding': old_protocol['embedding'], 'reranker': reranker.fingerprint(),
        'selection_lock_sha256': sha(args.selection) if args.selection else None,
        'generation_calls': 0, 'fine_tuning_examples': 0}
    if args.split == 'test' and not args.selection:
        raise ValueError('Freeze a validation selection before scoring new test candidates')
    write_json(protocol, args.output/'protocol.json')
    sparse = SparseBM25([a['text'] for a in answers])
    start = time.perf_counter(); pair_seconds = 0.; count = 0
    with (args.output/'scores.jsonl').open('w', encoding='utf8') as handle:
        for start_idx in range(0, len(cases), 8):
            group = cases[start_idx:start_idx+8]
            # Preserve the frozen baseline's matrix-vector arithmetic exactly.
            # Batched GEMM can exchange nearly tied answer IDs in float32.
            group_dense = [dense @ queries[offset+start_idx+j] for j in range(len(group))]
            rows = []; pairs = []
            for c, d in zip(group, group_dense):
                b = sparse.scores(c['question'])
                bo, do = ranked(b, positive_only=True), ranked(d)
                union = sorted(set(bo)|set(do))
                row = {'id': c['id'], 'candidate_ids': [answers[j]['id'] for j in union],
                    'dense': d[union].tolist(), 'sparse': b[union].tolist(),
                    'bge_ids': [answers[j]['id'] for j in do], 'bm25_ids': [answers[j]['id'] for j in bo]}
                rows.append(row)
                pairs.extend((c['question'], answers[j]['text']) for j in union)
            t = time.perf_counter(); logits = reranker.score_pairs(pairs); pair_seconds += time.perf_counter()-t
            pointer = 0
            for row in rows:
                n = len(row['candidate_ids']); row['cross'] = logits[pointer:pointer+n].tolist(); pointer += n
                handle.write(json.dumps(row)+'\n'); count += 1
            handle.flush()
            if count % 80 == 0 or count == len(cases):
                print(json.dumps({'scored_questions': count, 'total': len(cases),
                    'pairs': reranker.pairs_scored, 'elapsed_seconds': round(time.perf_counter()-start, 1)}), flush=True)
    if code != {p.relative_to(ROOT).as_posix(): sha(p) for p in files}:
        raise ValueError('Scoring implementation changed during run')
    verify_fixture(args.fixture)
    write_json({'status': 'completed', 'questions': count, 'pairs': reranker.pairs_scored,
        'seconds': time.perf_counter()-start, 'cross_encoder_seconds': pair_seconds,
        'scores_sha256': sha(args.output/'scores.jsonl'), 'protocol_sha256': sha(args.output/'protocol.json')},
        args.output/'completion.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, default=ROOT/'data/benchmarks/insuranceqa_v2')
    p.add_argument('--baseline', type=Path, default=ROOT/'reports/insuranceqa_v2/retrieval_frozen')
    p.add_argument('--split', choices=['valid', 'test'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--selection', type=Path)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
