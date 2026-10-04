"""Mine current-retriever training negatives and frozen reranker teacher scores."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_query_adaptation_data import write_jsonl
from scripts.prepare_reranker_training import normalize
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from src.insurerag_vlm.query_adaptation import QueryEncoder,blend_queries
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder
from src.insurerag_vlm.reranker import blend_scores


def main():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();data=ROOT/'data/training/evidence_reranker_v1';run=ROOT/'reports/evidence_reranker_v1'
    cache=run/'index_cache';cache.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'data_manifest.lock.json').read_text(encoding='utf8'))
    for n,h in manifest['files'].items():assert sha(data/n)==h
    groups=read_jsonl(data/'query_groups.jsonl');answers=read_jsonl(data/'answers.jsonl')
    lookup={a['id']:a for a in answers};normal={a['id']:normalize(a['text']) for a in answers}
    prior=ROOT/'reports/query_adaptation_v1';selection=json.loads((prior/'selection.lock.json').read_text(encoding='utf8'))
    querypath=ROOT/'../models/insurerag-query-adaptation-v1-seed-42/epoch-2'
    assert sha(querypath/'model.safetensors')==selection['query_weights_sha256']
    basepath=ROOT/'../models/bge-small-en-v1.5'
    assert sha(basepath/'model.safetensors')==selection['document_weights_sha256']
    previous=ROOT/'data/training/query_adaptation_v1'
    oldanswers=read_jsonl(previous/'answers.jsonl');assert answers[:len(oldanswers)]==oldanswers
    oldcache=prior/'index_cache';oldmanifest=json.loads((oldcache/'manifest.json').read_text(encoding='utf8'))
    for n in ['answer_embeddings','training_query_embeddings']:assert sha(oldcache/(n+'.npy'))==oldmanifest[n+'_sha256']
    extra=answers[len(oldanswers):]
    encoder=EmbeddingRetriever(str(basepath),use_hf_api=False,pooling='cls',max_length=512)
    _,m=encoder._ensure_local_transformer();m.to('cuda')
    dense=np.vstack([np.load(oldcache/'answer_embeddings.npy'),encoder.embed_texts([a['text'] for a in extra])])
    np.save(cache/'answer_embeddings.npy',dense)
    del encoder,m;torch.cuda.empty_cache()
    oldgroups=read_jsonl(previous/'train_groups.jsonl');oldmap={g['id']:i for i,g in enumerate(oldgroups)}
    oldqueries=np.load(oldcache/'training_query_embeddings.npy');baseq=np.empty((len(groups),384),dtype=np.float32)
    newix=[]
    for i,g in enumerate(groups):
        if g['id'] in oldmap:baseq[i]=oldqueries[oldmap[g['id']]]
        else:newix.append(i)
    encoder=QueryEncoder(basepath,'cuda');baseq[newix]=encoder.encode([groups[i]['question'] for i in newix],64)
    del encoder;torch.cuda.empty_cache()
    encoder=QueryEncoder(querypath,'cuda');adapted=encoder.encode([g['question'] for g in groups],64)
    query=blend_queries(baseq,adapted,.5);np.save(cache/'training_queries.npy',query)
    del encoder;torch.cuda.empty_cache()
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'answer_count':len(answers),'query_count':len(groups),
                'answer_file_sha256':sha(data/'answers.jsonl'),'query_groups_sha256':sha(data/'query_groups.jsonl'),
                'answer_embeddings_sha256':sha(cache/'answer_embeddings.npy'),'training_queries_sha256':sha(cache/'training_queries.npy'),
                'query_selection_sha256':sha(prior/'selection.lock.json'),'original_document_weights_sha256':selection['document_weights_sha256'],
                'adapted_query_weights_sha256':selection['query_weights_sha256'],'query_alpha':.5,'test_queries_encoded':False},cache/'manifest.json')
    print(json.dumps({'encoded_documents':len(answers),'encoded_training_queries':len(groups),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    forbidden=set(json.loads((data/'isolation.json').read_text(encoding='utf8'))['forbidden_answer_ids'])
    position={a['id']:i for i,a in enumerate(answers)}
    corpora={
        'insuranceqa':read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl'),
        'government':read_jsonl(ROOT/'data/benchmarks/condition_v1/answers.jsonl'),
        'general':read_jsonl(ROOT/'data/benchmarks/condition_v1/answers.jsonl'),
        'finance':read_jsonl(ROOT/'data/benchmarks/fiqa_v1/answers.jsonl'),
    }
    reports=defaultdict(list)
    for a in extra:reports[a['source_group']].append(a)
    oldnegative={g['id']:g['negative_ids'] for g in read_jsonl(ROOT/'data/training/condition_listwise_v1/train_groups.jsonl')}
    rng=np.random.default_rng(20261003);pools={};mining_exclusions=[]
    domain_indices=defaultdict(list)
    for i,g in enumerate(groups):domain_indices[g['source_domain']].append(i)
    for domain,indices in domain_indices.items():
        subsets={domain:indices} if domain!='financial_report' else {s:[i for i in indices if groups[i]['source_group']==s] for s in sorted({groups[i]['source_group'] for i in indices})}
        for scope,ix in subsets.items():
            corpus=corpora[domain] if domain!='financial_report' else reports[scope]
            ids=[a['id'] for a in corpus];dp=dense[[position[a] for a in ids]]
            sparse=SparseBM25([a['text'] for a in corpus]);csc=sparse.matrix.tocsc()
            allow=np.array([j for j,a in enumerate(ids) if a not in forbidden])
            allowed=set(ids[j] for j in allow)
            for off in range(0,len(ix),64):
                bi=ix[off:off+64];products=dp@query[bi].T
                for k,i in enumerate(bi):
                    g=groups[i];gold=set(g['positive_ids']);postexts={normal[a] for a in gold}
                    accept=lambda a:a not in gold and a in allowed and normal[a] not in postexts
                    q=sparse.vectorizer.transform([g['question']]).tocsr();b=np.asarray(csc[:,q.indices]@q.data).ravel();d=products[:,k]
                    do=[ids[j] for j in ranked(d,allow,limit=30) if accept(ids[j])][:16]
                    bo=[ids[j] for j in ranked(b,allow,limit=30,positive_only=True) if accept(ids[j])][:16]
                    candidates=[];seen=set()
                    for a in do+bo+oldnegative.get(g['id'],[]):
                        if accept(a) and normal[a] not in seen:candidates.append(a);seen.add(normal[a])
                    randomid=None
                    # Stop once the needed distinct distractors are found; do not materialize
                    # tens of thousands of unnecessary candidates for every training query.
                    for j in rng.permutation(allow):
                        a=ids[j]
                        if not accept(a) or normal[a] in seen:continue
                        if len(candidates)<10:candidates.append(a);seen.add(normal[a])
                        else:randomid=a;break
                    if len(candidates)<5:
                        mining_exclusions.append({'id':g['id'],'reason':'fewer_than_five_distinct_unlabelled_negatives'});continue
                    pools[g['id']]={'candidate_ids':candidates,'lexical':bo,'random':randomid}
        print(json.dumps({'retrieved_domain':domain,'groups_so_far':len(pools),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    modelpath=ROOT/'../models/insurerag-condition-listwise-v1-seed-123/epoch-1'
    assert sha(modelpath/'model.safetensors')==selection['fixed_reranker_weights_sha256']
    teacher=DomainCrossEncoder(modelpath,'cuda',64,512);accepted=[]
    for off in range(0,len(groups),32):
        batch=[g for g in groups[off:off+32] if g['id'] in pools];pairs=[];wanted=[]
        for g in batch:
            p=pools[g['id']];ids=list(dict.fromkeys(g['positive_ids']+p['candidate_ids']+([p['random']] if p['random'] else [])))
            wanted.append(ids);pairs.extend((g['question'],lookup[a]['text']) for a in ids)
        values=teacher.score_pairs(pairs);cursor=0
        for g,ids in zip(batch,wanted):
            scores=dict(zip(ids,map(float,values[cursor:cursor+len(ids)])));cursor+=len(ids);p=pools[g['id']]
            order=sorted(p['candidate_ids'],key=lambda a:(-scores[a],a));hard=order[:6]
            lex=[a for a in p['lexical'] if a not in hard][:4]
            lex += [a for a in order if a not in hard+lex][:4-len(lex)]
            negatives=hard+lex
            if p['random'] and p['random'] not in negatives:negatives.append(p['random'])
            assert len(negatives)>=5 and not set(g['positive_ids']+negatives)&forbidden
            accepted.append({**g,'negative_ids':negatives,'negative_sources':{**{a:'current_hard' for a in hard},**{a:'lexical_hard' for a in lex},**({p['random']:'random'} if p['random'] and p['random'] not in hard+lex else {})},
                             'anchor_scores':{a:scores[a] for a in g['positive_ids']+negatives}})
        if (off+len(batch))%1024==0 or off+32>=len(groups):print(json.dumps({'scored_training_groups':off+len(batch),'pairs_scored':teacher.pairs_scored,'seconds':round(time.perf_counter()-start,1)}),flush=True)
    write_jsonl(accepted,data/'train_groups.jsonl');write_json({'exclusions':mining_exclusions,'remaining_forbidden_pairs':0},data/'mining_audit.json')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'training_questions':len(accepted),
                'source_domains':dict(Counter(g['source_domain'] for g in accepted)),
                'positive_pairs':sum(len(g['positive_ids']) for g in accepted),'negative_pairs':sum(len(g['negative_ids']) for g in accepted),
                'anchor_model':teacher.fingerprint(),'anchor_pairs_scored':teacher.pairs_scored,
                'candidate_retriever':'fixed prior 50/50 query adaptation + original BGE docs + BM25; training queries only',
                'data_manifest_sha256':sha(data/'data_manifest.lock.json'),'index_manifest_sha256':sha(cache/'manifest.json'),
                'seconds':time.perf_counter()-start,'code_sha256':sha(Path(__file__)),
                'files':{n:sha(data/n) for n in ['answers.jsonl','query_groups.jsonl','train_groups.jsonl','isolation.json','mining_audit.json']}},data/'manifest.lock.json')
    print(json.dumps({'completed':True,'groups':len(accepted),'seconds':round(time.perf_counter()-start,1)}),flush=True)


if __name__=='__main__':main()
