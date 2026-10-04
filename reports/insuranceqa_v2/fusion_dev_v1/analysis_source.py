"""Explore fusion on validation ONLY. Original test predictions used only for old-arm diagnosis."""
from pathlib import Path
import sys,json,time,hashlib
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent/'InsureRAG-VLM';sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha,verify_fixture
from scripts.eval_insuranceqa_scale import SparseBM25,ranked,metrics,aggregate
FIX=ROOT/'data/benchmarks/insuranceqa_v2';OLD=ROOT/'reports/insuranceqa_v2/retrieval_frozen'
OUT=ROOT/'reports/insuranceqa_v2/fusion_dev_v1';OUT.mkdir(exist_ok=False)
verify_fixture(FIX)
settings={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_split':'valid',
 'selection_rule':'Maximize valid Hit@10, then MRR@100, then Hit@1; original test is historical, not blind.',
 'candidate_depth_per_channel':100,'rrf_k':[5,20,60],
 'rrf_bm25_weight':[0,.05,.1,.2,.35,.5,1],
 'linear_bm25_weight':[0,.025,.05,.1,.15,.2,.3,.4,.5],
 'linear_normalization':'Per-query per-channel min-max over the union of top-100 candidates.',
 'fixture_sha256':sha(FIX/'manifest.lock.json'),'script_sha256':sha(Path(__file__))}
write_json(settings,OUT/'selection_protocol.json')
answers=sorted(read_jsonl(FIX/'answers.jsonl'),key=lambda a:int(a['id']));ids={a['id']:i for i,a in enumerate(answers)}
cases=read_jsonl(FIX/'valid.jsonl');sparse=SparseBM25([a['text'] for a in answers])
dense=np.load(OLD/'answer_embeddings.npy');q=np.load(OLD/'query_embeddings.npy')[:len(cases)]
threadpool_limits(4)
all_dense=q@dense.T
orders=[];candidates=[];gold=[];dense_scores=[];sparse_scores=[]
start=time.perf_counter()
for i,c in enumerate(cases):
    b=sparse.scores(c['question']);d=all_dense[i]
    bo=ranked(b,positive_only=True);do=ranked(d)
    union=sorted(set(bo)|set(do));candidates.append(np.array(union));orders.append((bo,do))
    gold.append({ids[x] for x in c['gold_answer_ids']});dense_scores.append(d[union]);sparse_scores.append(b[union])
    if (i+1)%500==0:print(json.dumps({'validation_scores':i+1,'seconds':time.perf_counter()-start}),flush=True)
np.savez_compressed(OUT/'valid_candidate_scores.npz',
 candidates=np.array(candidates,dtype=object),dense=np.array(dense_scores,dtype=object),
 sparse=np.array(sparse_scores,dtype=object),allow_pickle_note='Variable-length arrays; trusted local artifact only')
configs=[{'method':'weighted_rrf','k':k,'bm25_weight':w} for k in settings['rrf_k'] for w in settings['rrf_bm25_weight']]
configs += [{'method':'linear_minmax','bm25_weight':w} for w in settings['linear_bm25_weight']]
summary=[]
for cfg in configs:
    results=[]
    for i,(bo,do) in enumerate(orders):
        cand=candidates[i];w=cfg['bm25_weight']
        if cfg['method']=='weighted_rrf':
            br={j:r for r,j in enumerate(bo,1)};dr={j:r for r,j in enumerate(do,1)};k=cfg['k']
            s=np.array([(w/(k+br[j]) if j in br else 0)+(1/(k+dr[j]) if j in dr else 0) for j in cand])
        else:
            normal=lambda a:(a-a.min())/max(float(a.max()-a.min()),1e-12)
            s=(1-w)*normal(dense_scores[i])+w*normal(sparse_scores[i])
        order=cand[np.argsort(-s,kind='stable')[:100]].tolist()
        results.append(metrics(order,gold[i]))
    row={**cfg,**aggregate(results)};summary.append(row)
summary.sort(key=lambda r:(r['hit_at_10'],r['mrr_at_100'],r['hit_at_1']),reverse=True)
write_json({'settings':settings,'configurations':summary},OUT/'validation_sweep.json')

# Count the already-published BGE vs old RRF transition table; do not test new configs.
oldrows=read_jsonl(OLD/'predictions.jsonl');lookup={}
for r in oldrows:
    if r['scope']=='all_27413_answers':lookup.setdefault((r['split'],r['id']),{})[r['arm']]=r
diag={}
for split in ['valid','test']:
    counts=Counter();examples=[]
    original_cases={c['id']:c for c in read_jsonl(FIX/f'{split}.jsonl')}
    for (s,id),arms in lookup.items():
        if s!=split:continue
        b=bool(arms['bge']['hit_at_10']);r=bool(arms['rrf']['hit_at_10']);m=bool(arms['bm25']['hit_at_10'])
        counts[f'bge_{int(b)}_rrf_{int(r)}']+=1
        counts[f'bge_{int(b)}_bm25_{int(m)}']+=1
        if b and not r and len(examples)<8:
            c=original_cases[id];positive=c['gold_answer_ids'];pos=lambda a:{g:(a.index(g)+1 if g in a else None) for g in positive}
            examples.append({'id':id,'question':c['question'],'gold_ids':positive,
                'positions':{a:pos(arms[a]['top_answer_ids']) for a in ['bge','bm25','rrf']}})
    diag[split]={'transition_counts':dict(counts),'first_lost_examples':examples}
write_json(diag,OUT/'historical_diagnosis.json')
print(json.dumps({'validation_top':summary[:12],'historical_counts':{s:r['transition_counts'] for s,r in diag.items()}},indent=2),flush=True)
