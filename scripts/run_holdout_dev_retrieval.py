"""Offline DEV retrieval only; incomplete corpus/review gates fail before models load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from importlib.metadata import version

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.holdout_corpus import build_corpus
from src.insurerag_vlm.holdout_dev_runner import readiness, run_requests, fingerprint
from src.insurerag_vlm.holdout_retrieval import ARMS, rank_arm, rank_six_arms
from src.insurerag_vlm.holdout_standalone import run_requests as run_standalone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['documents', 'manifest', 'requests', 'clearance']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--models', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--execution', choices=['standalone','shared'], default='standalone')
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    docs = [json.loads(line) for line in a.documents.read_text(encoding='utf8').splitlines()]
    manifest, requests, clearance = read(a.manifest), read(a.requests), read(a.clearance)
    check = readiness(docs, manifest, requests, clearance)
    print(json.dumps(check))
    if not check['ready']:
        return 2
    if a.check_only:
        return 0
    if a.models is None or a.output is None:
        p.error('--models and --output are required for inference')
    pins = {
        'bge-small-en-v1.5': '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
        'insurerag-query-adaptation-v1-seed-42/epoch-2': 'e067a9a14428aabb32dbd2f0e84a7af31bcb8290ffc6fe422faed906d4d5b857',
        'insurerag-evidence-reranker-v1-seed-123/epoch-2': '697d3a725ad9d684c1b978365bda314114e8dbec2a27c2841a07f028ed2e81b8'}
    model_files = {}
    for name, wanted in pins.items():
        directory = a.models/name
        if sha(directory/'model.safetensors') != wanted:
            raise ValueError('Inherited model weight fingerprint mismatch')
        model_files[name] = {f.name: sha(f) for f in sorted(directory.iterdir())
                             if f.is_file() and f.suffix in ('.json', '.txt', '.safetensors')}
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from scripts.eval_insuranceqa_scale import SparseBM25
    from src.insurerag_vlm.query_adaptation import QueryEncoder, QUERY_INSTRUCTION, blend_queries
    from src.insurerag_vlm.domain_reranker import DomainCrossEncoder
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    cold_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(a.models/'bge-small-en-v1.5', local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    chunks = sorted(build_corpus(docs, manifest, count), key=lambda c: c['id'])
    ids = [c['id'] for c in chunks]
    # Reviewed source headers are part of the encoded/reranked context budget.
    texts = [c.get('retrieval_text', c['text']) for c in chunks]
    by_id = dict(zip(ids, texts))
    code_files = ['scripts/run_holdout_dev_retrieval.py', 'scripts/eval_insuranceqa_scale.py',
                  'src/insurerag_vlm/holdout_corpus.py', 'src/insurerag_vlm/holdout_dev_runner.py',
                  'src/insurerag_vlm/holdout_retrieval.py', 'src/insurerag_vlm/query_adaptation.py',
                  'src/insurerag_vlm/reranker.py', 'src/insurerag_vlm/domain_reranker.py',
                  'src/insurerag_vlm/holdout_standalone.py','src/insurerag_vlm/holdout_dev_scoring.py',
                  'src/insurerag_vlm/holdout_experiment.py','scripts/score_holdout_dev.py']
    bindings = dict(source_file_sha256=sha(a.documents), source_manifest_sha256=sha(a.manifest),
                    clearance_sha256=sha(a.clearance), models=model_files,
                    code={name:sha(ROOT/name) for name in code_files},
                    protocol_sha256=sha(ROOT/'reports/holdout1000_v1/evaluation_protocol_v1.json'),
                    execution=a.execution, standalone_warmup='one fixed synthetic query per missing arm',
                    device=a.device, torch_version=torch.__version__, numpy_version=np.__version__,
                    transformers_version=version('transformers'), chunks_sha256=fingerprint(chunks),
                    query_blend=.5, lexical_weight=.2, cross_weight=.5, document_batch_size=32,
                    document_tokens=512, per_chunk_chars=2400, candidate_budget_per_arm=200)
    # The saved source-only corpus is also the scorer's immutable chunk-ID universe.
    run_dir = a.output/('standalone' if a.execution == 'standalone' else 'requests')
    runner = run_standalone if a.execution == 'standalone' else run_requests
    saved_manifest = run_dir/'run_manifest.json'
    if saved_manifest.exists() and read(saved_manifest).get('bindings') != bindings:
        raise ValueError('Run bindings changed; preserve the prior output directory')
    chunk_file = a.output/'corpus_chunks.jsonl'
    if chunk_file.exists():
        saved_chunks = [json.loads(line) for line in chunk_file.read_text(encoding='utf8').splitlines()]
        if saved_chunks != chunks:
            raise ValueError('Saved source corpus differs')
    else:
        a.output.mkdir(parents=True, exist_ok=True)
        with chunk_file.open('x',encoding='utf8') as handle:
            for row in chunks:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    expected_records = ([run_dir/arm/(fingerprint(r)+'.json') for r in requests for arm in ARMS]
                        if a.execution == 'standalone' else [run_dir/(fingerprint(r)+'.json') for r in requests])
    if saved_manifest.exists() and all(path.exists() for path in expected_records):
        def forbidden(*_):
            raise AssertionError('A complete resume must not invoke any model')
        print(json.dumps(runner(requests,ids,forbidden,run_dir,bindings)))
        return 0
    a.output.mkdir(parents=True, exist_ok=True)
    base = QueryEncoder(a.models/'bge-small-en-v1.5', a.device)
    cache_key = fingerprint(dict(chunks=bindings['chunks_sha256'], model=model_files['bge-small-en-v1.5'],
                                 encoder_code=bindings['code']['src/insurerag_vlm/query_adaptation.py'],
                                 runner_code=bindings['code']['scripts/run_holdout_dev_retrieval.py'], device=a.device))
    vector_file, meta_file = a.output/'document_vectors.npy', a.output/'document_vectors.json'
    if vector_file.exists() or meta_file.exists():
        if not vector_file.exists() or not meta_file.exists():
            raise ValueError('Incomplete vector cache; preserve it for inspection')
        meta = read(meta_file)
        if meta.get('key') != cache_key or meta.get('file_sha256') != sha(vector_file):
            raise ValueError('Document vector cache fingerprint mismatch')
        vectors = np.load(vector_file, allow_pickle=False)
        reused_vectors = True
    else:
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                features = base.tokenizer(texts[start:start+32], padding=True, truncation=False,
                                          return_tensors='pt').to(a.device)
                if features['input_ids'].shape[1] > 512:
                    raise ValueError('Document encoding would truncate')
                hidden = base.model(**features).last_hidden_state[:,0].float()
                batches.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        vectors = np.vstack(batches)
        np.save(vector_file, vectors, allow_pickle=False)
        meta_file.write_text(json.dumps(dict(key=cache_key, file_sha256=sha(vector_file)),indent=2)+'\n',encoding='utf8')
        reused_vectors = False
    if (vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)):
        raise ValueError('Invalid document vector matrix')
    learned = QueryEncoder(a.models/'insurerag-query-adaptation-v1-seed-42/epoch-2',a.device)
    cross = DomainCrossEncoder(a.models/'insurerag-evidence-reranker-v1-seed-123/epoch-2',a.device)
    bm25 = SparseBM25(texts)
    setup = dict(cold_setup_seconds=time.perf_counter()-cold_start, document_vectors_reused=reused_vectors,
                 source_documents=len(docs), chunks=len(chunks), max_document_tokens=max(c['tokens'] for c in chunks))
    def infer(request):
        q = request['question']
        trace = dict(query_tokens_original=len(base.tokenizer.encode(QUERY_INSTRUCTION+q,truncation=False)),
                     query_tokens_adapted=len(learned.tokenizer.encode(QUERY_INSTRUCTION+q,truncation=False)))
        if max(trace.values()) > 512:
            raise ValueError('Query exceeds encoder budget; no silent query truncation')
        start = time.perf_counter(); original = base.encode([q]); trace['original_query_encode_seconds']=time.perf_counter()-start
        start = time.perf_counter(); adapted = blend_queries(original,learned.encode([q]),.5); trace['adapted_query_encode_seconds']=time.perf_counter()-start
        start = time.perf_counter(); sparse = bm25.scores(q); trace['sparse_search_seconds']=time.perf_counter()-start
        start = time.perf_counter(); dense = vectors@original[0]; dense_adapted=vectors@adapted[0]; trace['both_dense_search_seconds']=time.perf_counter()-start
        def score(selected):
            selected_texts = [by_id[i] for i in selected]
            full = cross.tokenizer([q]*len(selected),selected_texts,truncation=False)['input_ids']
            packed = cross.tokenizer([q]*len(selected),selected_texts,truncation=True,max_length=cross.max_length)['input_ids']
            trace['reranker_pairs']=[dict(chunk_id=i,full_tokens=len(f),encoded_tokens=len(t),truncated=len(f)>len(t)) for i,f,t in zip(selected,full,packed)]
            begin=time.perf_counter();values=cross.score(q,selected_texts);trace['shared_cross_seconds']=time.perf_counter()-begin
            return values
        start=time.perf_counter();result=rank_six_arms(ids,dense,dense_adapted,sparse,score);trace['six_arm_fusion_cross_seconds']=time.perf_counter()-start
        trace['candidate_scores']=result['candidate_scores']
        return dict(orders=result['orders'],trace=trace)
    def infer_arm(arm, request):
        q = request['question']; trace = {}; dense = sparse = None
        if arm != 'bm25':
            trace['query_tokens_original'] = len(base.tokenizer.encode(QUERY_INSTRUCTION+q,truncation=False))
            if trace['query_tokens_original'] > 512:
                raise ValueError('Query exceeds encoder budget; no silent truncation')
            vector = base.encode([q])
            if arm.endswith('_adapted'):
                trace['query_tokens_adapted'] = len(learned.tokenizer.encode(QUERY_INSTRUCTION+q,truncation=False))
                if trace['query_tokens_adapted'] > 512:
                    raise ValueError('Adapted query exceeds encoder budget')
                vector = blend_queries(vector,learned.encode([q]),.5)
            dense = vectors@vector[0]
        if arm != 'dense_original':
            sparse = bm25.scores(q)
        def score(selected):
            selected_texts = [by_id[i] for i in selected]
            full = cross.tokenizer([q]*len(selected),selected_texts,truncation=False)['input_ids']
            packed = cross.tokenizer([q]*len(selected),selected_texts,truncation=True,max_length=cross.max_length)['input_ids']
            trace['reranker_pairs'] = [dict(chunk_id=i,full_tokens=len(f),encoded_tokens=len(t),truncated=len(f)>len(t))
                                      for i,f,t in zip(selected,full,packed)]
            return cross.score(q,selected_texts)
        result = rank_arm(arm,ids,dense=dense,sparse=sparse,cross_score=score)
        trace['candidate_ids'] = result.get('candidate_ids',[])
        trace['cross_pairs'] = result['cross_pairs']
        return dict(order=result['order'],trace=trace)
    if a.execution == 'standalone':
        synchronize = (lambda:torch.cuda.synchronize(a.device)) if a.device == 'cuda' else (lambda:None)
        warmups = []
        for arm in ARMS:
            if all((run_dir/arm/(fingerprint(r)+'.json')).exists() for r in requests):continue
            start = time.perf_counter()
            try:
                infer_arm(arm,dict(id='synthetic_warmup',question='Synthetic retrieval warm-up.'))
                synchronize()
                warmups.append(dict(arm=arm,status='success',seconds=time.perf_counter()-start))
            except Exception as exc:
                warmups.append(dict(arm=arm,status='failed',seconds=time.perf_counter()-start,
                                    error=dict(type=type(exc).__name__,message=str(exc))))
        setup['synthetic_warmups'] = warmups
        summary = run_standalone(requests,ids,infer_arm,run_dir,bindings,synchronize=synchronize)
    else:
        summary = run_requests(requests,ids,infer,run_dir,bindings)
    setup['summary']=summary
    (a.output/'setup.json').write_text(json.dumps(setup,indent=2)+'\n',encoding='utf8')
    print(json.dumps(setup))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
