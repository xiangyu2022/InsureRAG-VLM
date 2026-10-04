"""Frozen all-answer retrieval and untouched upstream-pool evaluation, no LLM."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys,time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha,verify_fixture
from src.insurerag_vlm.retriever import EmbeddingRetriever,tokenize


class SparseBM25:
    """Sparse matrix implementation of the repository's BM25 formula."""
    def __init__(self,texts,k1=1.5,b=.75):
        self.vectorizer=CountVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,dtype=np.float64)
        counts=self.vectorizer.fit_transform(texts).tocsr()
        lengths=np.asarray(counts.sum(axis=1)).ravel()
        df=np.asarray((counts>0).sum(axis=0)).ravel()
        norm=k1*(1-b+b*lengths/max(lengths.mean(),1e-10))
        rows=np.repeat(np.arange(counts.shape[0]),np.diff(counts.indptr))
        counts.data=counts.data*(k1+1)/(counts.data+norm[rows])
        self.matrix=counts.multiply(np.log(1+(len(texts)-df+.5)/(df+.5))).tocsr()
    def scores(self,query):
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()


def ranked(scores,allowed=None,limit=100,positive_only=False):
    indices=np.arange(len(scores)) if allowed is None else np.array(allowed,dtype=int)
    # Deterministic answer-ID order breaks ties. Zero BM25 scores are not hits.
    if positive_only:indices=indices[scores[indices]>0]
    return indices[np.argsort(-scores[indices],kind='stable')[:limit]].tolist()


def rrf(first,second,n,k=60):
    scores=np.zeros(n,dtype=np.float64)
    for order in [first,second]:
        for rank,index in enumerate(order,1):scores[index]+=1/(k+rank)
    return scores


def metrics(order,gold):
    positions=[i+1 for i,x in enumerate(order) if x in gold]
    return {**{f'hit_at_{k}':float(any(p<=k for p in positions)) for k in [1,5,10,100]},
            'mrr_at_100':1/min(positions) if positions else 0.,
            'label_recall_at_10':sum(x in gold for x in order[:10])/len(gold)}


def wilson(successes,n):
    if not n:return [None,None]
    p=successes/n;z=1.959963984540054;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    return [center-half,center+half]


def aggregate(rows):
    keys=['hit_at_1','hit_at_5','hit_at_10','hit_at_100','mrr_at_100','label_recall_at_10']
    return {'n':len(rows),**{key:float(np.mean([r[key] for r in rows])) if rows else None for key in keys},
            'hit_at_1_wilson_95':wilson(sum(r['hit_at_1'] for r in rows),len(rows)),
            'hit_at_10_wilson_95':wilson(sum(r['hit_at_10'] for r in rows),len(rows))}


def run(args):
    fixture=args.fixture
    lock=verify_fixture(fixture)
    answers=sorted(read_jsonl(fixture/'answers.jsonl'),key=lambda a:int(a['id']))
    id_to_index={a['id']:i for i,a in enumerate(answers)}
    cases=read_jsonl(fixture/'valid.jsonl')+read_jsonl(fixture/'test.jsonl')
    args.output.mkdir(parents=True,exist_ok=False)
    code_files=[Path(__file__),ROOT/'scripts/prepare_insuranceqa.py',ROOT/'src/insurerag_vlm/retriever.py']
    code_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in code_files}
    retriever=EmbeddingRetriever(args.model,use_hf_api=False,pooling='cls',max_length=512,
        query_instruction='Represent this sentence for searching relevant passages: ')
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_lock_sha256':sha(fixture/'manifest.lock.json'),
        'code_sha256':code_hashes,'answer_count':len(answers),'valid_questions':2000,'test_questions':2000,
        'arms':['bm25','bge','rrf'],'pool_modes':['all_27413_answers','original_solr_1000_pool'],
        'bm25':{'k1':1.5,'b':.75},'rrf':{'k':60,'depth':100},'fit_questions':0,'label_injection':False,
        'embedding':retriever.index_fingerprint(),'device':args.device,'max_tokens_per_answer':512,
        'query_instruction':retriever.query_instruction,'generation_calls':0,
        'limitations':['Author splits are question-level, not document-family holdout.',
          'Historical FAQ annotations; unlabeled plausible answers can be scored incorrect.',
          'BGE pretraining exposure unknown. No fine-tuning or hyperparameter selection in this run.',
          'Wilson intervals assume independent questions; near duplicates remain.',
          'FAQ retrieval measures neither policy coverage interpretation nor graph reasoning.']}
    write_json(protocol,args.output/'protocol.json')
    import torch
    torch.set_num_threads(4)
    tokenizer,model=retriever._ensure_local_transformer()
    model.to(args.device)
    emb=[];truncated=0;start=time.perf_counter()
    for offset in range(0,len(answers),256):
        texts=[a['text'] for a in answers[offset:offset+256]]
        lengths=tokenizer(texts,truncation=False,add_special_tokens=True,verbose=False)['input_ids']
        truncated+=sum(len(x)>512 for x in lengths)
        emb.append(retriever.embed_texts(texts))
        if offset%2048==0:print(json.dumps({'answer_embeddings':min(offset+256,len(answers)),'total':len(answers)}),flush=True)
    dense=np.vstack(emb)
    np.save(args.output/'answer_embeddings.npy',dense)
    qvectors=[]
    for offset in range(0,len(cases),256):
        qvectors.append(retriever.embed_texts([retriever.query_instruction+c['question'] for c in cases[offset:offset+256]]))
    qvectors=np.vstack(qvectors)
    np.save(args.output/'query_embeddings.npy',qvectors)
    sparse=SparseBM25([a['text'] for a in answers])
    results=[]
    with (args.output/'predictions.jsonl').open('w',encoding='utf8') as handle:
        for i,case in enumerate(cases):
            bm25=sparse.scores(case['question']);bge=dense @ qvectors[i]
            gold={id_to_index[x] for x in case['gold_answer_ids']}
            for scope,allowed in [('all_27413_answers',None),('original_solr_1000_pool',[id_to_index[x] for x in case['official_pool_ids']])]:
                orders={'bm25':ranked(bm25,allowed,positive_only=True),'bge':ranked(bge,allowed)}
                orders['rrf']=ranked(rrf(orders['bm25'],orders['bge'],len(answers)),allowed,positive_only=True)
                for arm,order in orders.items():
                    row={'id':case['id'],'split':case['split'],'domain':case['domain'],'arm':arm,'scope':scope,
                         'exact_overlap':case['exact_question_overlap_with_train_or_valid'],
                         'candidate_count':len(allowed) if allowed is not None else len(answers),
                         'gold_in_pool':bool(gold & set(allowed)) if allowed is not None else True,
                         'top_answer_ids':[answers[j]['id'] for j in order],**metrics(order,gold)}
                    results.append(row);handle.write(json.dumps(row)+'\n')
            if (i+1)%200==0:print(json.dumps({'evaluated_questions':i+1,'total':len(cases)}),flush=True)
    summaries={}
    for split in ['valid','test']:
        summaries[split]={}
        for scope in protocol['pool_modes']:
            summaries[split][scope]={}
            for arm in protocol['arms']:
                rows=[r for r in results if (r['split'],r['scope'],r['arm'])==(split,scope,arm)]
                summaries[split][scope][arm]={**aggregate(rows),
                    'no_exact_overlap':aggregate([r for r in rows if not r['exact_overlap']]),
                    'by_domain':{d:aggregate([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})},
                    'gold_present_in_pool':sum(r['gold_in_pool'] for r in rows)}
    if code_hashes!={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in code_files}:raise ValueError('Evaluation code changed during run')
    verify_fixture(fixture)
    report={'status':'completed','questions':len(cases),'scored_rankings':len(results),'generation_calls':0,
        'seconds':time.perf_counter()-start,'answers_truncated_at_512_tokens':truncated,'summaries':summaries,
        'predictions_sha256':sha(args.output/'predictions.jsonl'),'protocol_sha256':sha(args.output/'protocol.json')}
    write_json(report,args.output/'summary.json')
    print(json.dumps({scope:{arm:{k:v for k,v in metrics.items() if k in ['n','hit_at_1','hit_at_10','mrr_at_100']} for arm,metrics in rows.items()} for scope,rows in summaries['test'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,default=ROOT/'data/benchmarks/insuranceqa_v2')
    p.add_argument('--model',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
