"""Query the locked asymmetric retriever with the fixed insurance reranker."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha
from scripts.eval_insuranceqa_scale import SparseBM25, ranked
from scripts.eval_insuranceqa_reranker import rank_row
from scripts.eval_query_adaptation import check_contract, modelpath, FIXED_RERANKER, CFG
from src.insurerag_vlm.query_adaptation import QueryEncoder, blend_queries
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def verify_inference_files(model, expected):
    actual = {p.resolve().relative_to(ROOT.parent).as_posix(): sha(p) for p in model.iterdir()
              if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors'}}
    prefix = model.resolve().relative_to(ROOT.parent).as_posix()+'/'
    recorded = {n: h for n, h in expected.items() if n.startswith(prefix)}
    if not recorded or actual != recorded: raise ValueError('Inference checkpoint files changed')


def main(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    if not args.question.strip() or not 1 <= args.top_k <= 100: raise ValueError('Nonempty query and top-k 1..100 required')
    check_contract()
    run = ROOT / 'reports/query_adaptation_v1'
    lockpath = run / 'selection.lock.json'
    lock = json.loads(lockpath.read_text(encoding='utf8'))
    files = json.loads((run / 'checkpoint_files.lock.json').read_text(encoding='utf8'))['files_sha256']
    config = lock['config']
    basepath = ROOT / '../models/bge-small-en-v1.5'
    selected = modelpath(config)
    for path in {basepath, selected, FIXED_RERANKER}: verify_inference_files(path, files)
    assert sha(selected / 'model.safetensors') == lock['query_weights_sha256']
    assert sha(FIXED_RERANKER / 'model.safetensors') == lock['fixed_reranker_weights_sha256']
    corpuspath = {
        'insuranceqa': ROOT / 'data/benchmarks/insuranceqa_v2/answers.jsonl',
        'condition': ROOT / 'data/benchmarks/condition_v1/answers.jsonl',
        'fiqa': ROOT / 'data/benchmarks/fiqa_v1/answers.jsonl',
    }[args.corpus]
    answers = read_jsonl(corpuspath)
    allpath = ROOT / 'data/training/query_adaptation_v1/answers.jsonl'
    cache = run / 'index_cache'
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf8'))
    assert sha(allpath) == manifest['answer_file_sha256']
    assert sha(cache / 'answer_embeddings.npy') == manifest['answer_embeddings_sha256']
    allanswers = read_jsonl(allpath)
    lookup = {a['id']: i for i, a in enumerate(allanswers)}
    indices = [lookup[a['id']] for a in answers]
    assert all(a['text'] == allanswers[j]['text'] for a, j in zip(answers, indices))
    dense = np.load(cache / 'answer_embeddings.npy')[indices]
    encoder = QueryEncoder(basepath, args.device)
    query = encoder.encode([args.question])
    del encoder
    if args.device == 'cuda': torch.cuda.empty_cache()
    if config['name'] != 'original':
        encoder = QueryEncoder(selected, args.device)
        learned = encoder.encode([args.question])
        query = blend_queries(query, learned, config['alpha'])
        del encoder
        if args.device == 'cuda': torch.cuda.empty_cache()
    d = dense @ query[0]
    b = SparseBM25([a['text'] for a in answers]).scores(args.question)
    do, bo = ranked(d), ranked(b, positive_only=True)
    union = sorted(set(do) | set(bo))
    reranker = DomainCrossEncoder(FIXED_RERANKER, args.device)
    c = reranker.score(args.question, [answers[i]['text'] for i in union])
    row = {'candidate_ids': [answers[i]['id'] for i in union], 'dense': d[union].tolist(),
           'sparse': b[union].tolist(), 'cross': c.tolist(), 'bge_ids': [answers[i]['id'] for i in do]}
    order = rank_row(row, CFG)
    byid = {a['id']: a for a in answers}
    result = {'question': args.question, 'corpus': args.corpus, 'corpus_size': len(answers),
              'candidate_count': len(union), 'query_configuration': config, 'ranking_configuration': CFG,
              'query_encoder_passes': 1 if config['name'] == 'original' else 2,
              'document_encoder': 'unchanged BGE-small-en-v1.5', 'reranker_changed_this_iteration': False,
              'selection_lock_sha256': sha(lockpath), 'selected_query_weights_sha256': lock['query_weights_sha256'],
              'generation_calls': 0, 'research_retrieval_only': True, 'answers_expert_adjudicated': False,
              'seconds_including_loading': time.perf_counter()-start,
              'results': [{'rank': i+1, 'answer_id': a, 'text': byid[a]['text'],
                           **{k: byid[a][k] for k in ['domain', 'source_group', 'source_url', 'upstream_document_id'] if k in byid[a]}}
                          for i, a in enumerate(order[:args.top_k])]}
    output = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        if args.output.exists(): raise ValueError('Output already exists')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf8')
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--question', required=True)
    p.add_argument('--corpus', choices=['insuranceqa', 'condition', 'fiqa'], default='insuranceqa')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--output', type=Path)
    main(p.parse_args())
