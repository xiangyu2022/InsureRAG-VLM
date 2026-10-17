"""Offline real-model wiring smoke with synthetic sources; never a DEV/TEST score."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.holdout_corpus import build_corpus, digest, source_snapshot_digest
from src.insurerag_vlm.holdout_retrieval import ARMS, rank_six_arms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior smoke artifacts; choose a new output directory')
    expected = {
        'bge-small-en-v1.5': '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
        'insurerag-query-adaptation-v1-seed-42/epoch-2': 'e067a9a14428aabb32dbd2f0e84a7af31bcb8290ffc6fe422faed906d4d5b857',
        'insurerag-evidence-reranker-v1-seed-123/epoch-2': '697d3a725ad9d684c1b978365bda314114e8dbec2a27c2841a07f028ed2e81b8'}
    fingerprints = {}
    for name, wanted in expected.items():
        path = args.models / name
        if sha(path/'model.safetensors') != wanted:
            raise ValueError('Pinned inherited model weights changed')
        fingerprints[name] = {p.name: sha(p) for p in sorted(path.iterdir())
                              if p.is_file() and p.suffix in ('.json', '.txt', '.safetensors')}
    import numpy as np
    import torch
    from scripts.eval_insuranceqa_scale import SparseBM25
    from src.insurerag_vlm.query_adaptation import QueryEncoder, QUERY_INSTRUCTION, blend_queries
    from src.insurerag_vlm.domain_reranker import DomainCrossEncoder
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested local CUDA runtime unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    source_texts = [
        'Synthetic catalog: the violet option belongs to the amber contract. This invented example is not insurance guidance.',
        'Synthetic catalog: the silver option belongs to the cobalt contract. This invented example is not insurance guidance.',
        'Synthetic catalog: the copper option belongs to the jade contract. This invented example is not insurance guidance.',
        'Synthetic catalog: the gold option belongs to the coral contract. This invented example is not insurance guidance.']
    docs = [dict(id='synthetic_'+str(i), text=t, source_sha256=digest(t),
                 normalized_text_sha256=digest(t), publisher='synthetic_fixture',
                 jurisdiction='synthetic', source_url='synthetic:'+str(i))
            for i, t in enumerate(source_texts)]
    manifest = dict(version=1, selection_basis='source_only',
        source_snapshot_sha256=source_snapshot_digest(docs), entries=[
        dict(document_id=d['id'], source_sha256=d['source_sha256'],
             text_sha256=d['normalized_text_sha256'], selection_basis='source_only',
             ranges=[[0,len(d['text'])]], atomic_ranges=[], atomic_ranges_reviewed=True,
             review=dict(decision='include',record_sha256=sha(Path(__file__)),
                         scope_rationale='Entire original synthetic fixture text, no external sources or tables.'))
        for d in docs])
    question = 'Which invented contract has the violet option in this synthetic catalog?'
    base = QueryEncoder(args.models/'bge-small-en-v1.5', args.device)
    count = lambda text: len(base.tokenizer.encode(text, add_special_tokens=True, truncation=False))
    chunks = build_corpus(docs, manifest, count)
    chunks.sort(key=lambda c:c['id'])
    assert all(count(c['text']) <= 512 for c in chunks)
    assert count(QUERY_INSTRUCTION+question) <= 512
    with torch.inference_mode():
        features=base.tokenizer([c['text'] for c in chunks], padding=True,
                               truncation=False, return_tensors='pt').to(args.device)
        document_vectors=torch.nn.functional.normalize(
            base.model(**features).last_hidden_state[:,0].float(),dim=-1).cpu().numpy()
    original=base.encode([question])
    del features,base
    gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
    learned=QueryEncoder(args.models/'insurerag-query-adaptation-v1-seed-42/epoch-2',args.device)
    adapted=blend_queries(original,learned.encode([question]),.5)
    del learned
    gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
    assert np.isfinite(document_vectors).all()
    sparse=SparseBM25([c['text'] for c in chunks]).scores(question)
    cross=DomainCrossEncoder(args.models/'insurerag-evidence-reranker-v1-seed-123/epoch-2',args.device)
    by_id={c['id']:c for c in chunks}
    pair_lengths=[]
    def score(selected):
        texts=[by_id[i]['text'] for i in selected]
        tokens=cross.tokenizer([question]*len(texts),texts,truncation=False)['input_ids']
        pair_lengths.extend(map(len,tokens))
        if any(len(t)>cross.max_length for t in tokens):
            raise ValueError('Synthetic reranker smoke unexpectedly requires truncation')
        return cross.score(question,texts)
    result=rank_six_arms([c['id'] for c in chunks], document_vectors@original[0],
                         document_vectors@adapted[0], sparse, score)
    assert set(result['orders'])==set(ARMS)
    assert all(len(v)==4 and len(set(v))==4 and set(v)<=set(by_id)
               for v in result['orders'].values())
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='SYNTHETIC_REAL_MODEL_WIRING_PASSED_NOT_DEV_RESULTS',
        device=args.device, synthetic_documents=len(docs), chunks=len(chunks), queries=1,
        arms=list(ARMS), inherited_models_loaded=3, local_model_processes=1,
        encoder_forward_batches=3, cross_forward_batches=1, unique_cross_pairs=cross.pairs_scored,
        maximum_document_tokens=max(c['tokens'] for c in chunks),
        maximum_pair_tokens=max(pair_lengths), truncated_documents=0,truncated_pairs=0,
        qa_annotations_loaded=0, real_dev_questions=0, real_test_questions=0,
        generation_calls=0, network_requests=0, training_runs=0, paid_api_calls=0,
        runtime_seconds=time.perf_counter()-start,model_files_sha256=fingerprints,
        code_sha256={p:sha(ROOT/p) for p in ['scripts/smoke_holdout_retrieval.py',
                     'src/insurerag_vlm/holdout_retrieval.py','src/insurerag_vlm/holdout_corpus.py',
                     'src/insurerag_vlm/query_adaptation.py','src/insurerag_vlm/reranker.py',
                     'src/insurerag_vlm/domain_reranker.py','scripts/eval_insuranceqa_scale.py']},
        limitation='Tests runtime wiring only. No retrieval-quality conclusion or DEV selection. Full reviewed source corpus and support mapping remain required.')
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary.json',summary),('rankings.json',result),
                       ('synthetic_sources.json',docs),('synthetic_manifest.json',manifest)]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['model_files_sha256','code_sha256']}))


if __name__=='__main__':
    main()
