"""Mine student-confusing training candidates, retaining source isolation and teacher scores."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import write_jsonl
from scripts.prepare_reranker_training import normalize
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.reranker import LocalCrossEncoder
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

def run():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);start=time.perf_counter()
    fixture=ROOT/'data/benchmarks/condition_v1';old=ROOT/'data/training/retention_v2'
    out=ROOT/'data/training/condition_v1';index=ROOT/'reports/condition_v1/index_cache'
    for directory in [fixture,old]:
        lock=json.loads((directory/'manifest.lock.json').read_text(encoding='utf8'))
        for name,h in lock['files'].items():assert sha(directory/name)==h
    audit=json.loads((ROOT/'reports/condition_v1/fixture_verification.json').read_text(encoding='utf8'))
    assert not audit['mismatches'] and audit['fixture_sha256']==sha(fixture/'manifest.lock.json')
    out.mkdir(parents=True,exist_ok=False);index.mkdir(parents=True,exist_ok=False)
    answers=read_jsonl(fixture/'answers.jsonl');lookup={a['id']:a['text'] for a in answers};normal={k:normalize(v) for k,v in lookup.items()}
    original=read_jsonl(old/'train_groups.jsonl');added=read_jsonl(fixture/'train_additions.jsonl')
    held=[c for split in ['valid','test'] for c in read_jsonl(ROOT/f'data/benchmarks/multidomain_v1/{split}.jsonl')]+read_jsonl(fixture/'test.jsonl')
    heldsources={c['source_group'] for c in held}
    forbidden={a['id'] for a in answers if a.get('source_group') in heldsources}
    texts={normal[a] for a in forbidden}|{normalize(a['text']) for a in read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl')}
    forbidden.update(a for a in lookup if normal[a] in texts)
    mask=np.array([a['id'] in forbidden for a in answers]);sourceids=defaultdict(list)
    for a in answers:
        if a['id'] not in forbidden and a.get('domain')=='government':sourceids[a.get('source_group')].append(a['id'])
    groups=original+[{'id':c['id'],'question':c['question'],'positive_ids':c['gold_answer_ids'],'negative_ids':[],
                    'source_domain':c['domain'],'domain':c['domain'],'source_group':c['source_group'],'teacher_scores':{}} for c in added]
    assert len({g['id'] for g in groups})==len(groups)
    assert all(not set(g['positive_ids'])&forbidden for g in groups)
    encoder=EmbeddingRetriever(str(ROOT/'../models/bge-small-en-v1.5'),use_hf_api=False,pooling='cls',max_length=512,query_instruction='Represent this sentence for searching relevant passages: ')
    _,embedding=encoder._ensure_local_transformer();embedding.to('cuda')
    cache=ROOT/'reports/multidomain_v1/index_cache';cachelock=json.loads((cache/'manifest.json').read_text(encoding='utf8'))
    assert sha(old/'answers.jsonl')==cachelock['answer_file_sha256'] and sha(cache/'answer_embeddings.npy')==cachelock['answer_embeddings_sha256']
    baseanswers=read_jsonl(old/'answers.jsonl');assert answers[:len(baseanswers)]==baseanswers
    base=np.load(cache/'answer_embeddings.npy');newanswers=answers[len(baseanswers):]
    dense=np.vstack([base,encoder.embed_texts([a['text'] for a in newanswers])]);np.save(index/'answer_embeddings.npy',dense)
    write_json({'answer_count':len(answers),'answer_file_sha256':sha(fixture/'answers.jsonl'),'answer_embeddings_sha256':sha(index/'answer_embeddings.npy'),
        'embedding':encoder.index_fingerprint(),'parent_index_sha256':sha(cache/'manifest.json')},index/'manifest.json')
    queries=[]
    for i in range(0,len(groups),256):
        queries.append(encoder.embed_texts([encoder.query_instruction+g['question'] for g in groups[i:i+256]]))
        if i%4096==0:print(json.dumps({'query_embeddings':min(i+256,len(groups))}),flush=True)
    qv=np.vstack(queries);np.save(index/'training_query_embeddings.npy',qv)
    embedding.to('cpu');del encoder,embedding;torch.cuda.empty_cache()
    sparse=SparseBM25([a['text'] for a in answers]);csc=sparse.matrix.tocsc();rng=np.random.default_rng(20261001);poolrows=[]
    def lexical(question):
        q=sparse.vectorizer.transform([question]).tocsr()
        return np.asarray(csc[:,q.indices]@q.data).ravel()
    for g in groups[::max(1,len(groups)//20)]:
        assert np.allclose(lexical(g['question']),sparse.scores(g['question']),rtol=1e-12,atol=1e-12)
    for i,g in enumerate(groups):
        if i%64==0:batchdense=dense@qv[i:i+64].T
        d=batchdense[:,i%64].copy();b=lexical(g['question']);d[mask]=-np.inf;b[mask]=0
        pos=set(g['positive_ids']);postexts={normal[a] for a in pos};seen=set();pool=[];lex=[]
        def accept(a):return a not in pos and a not in forbidden and normal[a] not in postexts
        do=[answers[j]['id'] for j in ranked(d) if accept(answers[j]['id'])][:12]
        bo=[answers[j]['id'] for j in ranked(b,positive_only=True) if accept(answers[j]['id'])][:12]
        neighbors=sourceids.get(g.get('source_group'),[]) if g['source_domain']=='government' else []
        for a in do+bo+g['negative_ids']+neighbors:
            if accept(a) and normal[a] not in seen:pool.append(a);seen.add(normal[a])
        for a in bo:
            if a in pool and a not in lex:lex.append(a)
        randomid=next(answers[j]['id'] for j in rng.permutation(len(answers)) if accept(answers[j]['id']) and normal[answers[j]['id']] not in seen)
        poolrows.append({'pool':pool,'lexical':lex,'random':randomid})
        if (i+1)%2000==0:print(json.dumps({'retrieved_training_pools':i+1}),flush=True)
    student=DomainCrossEncoder(ROOT/'../models/insurerag-retention-v2/epoch-2','cuda',64,512)
    for offset in range(0,len(groups),16):
        batch=groups[offset:offset+16];pools=poolrows[offset:offset+16]
        pairs=[(g['question'],lookup[a]) for g,p in zip(batch,pools) for a in p['pool']]
        scores=student.score_pairs(pairs);k=0
        for g,p in zip(batch,pools):
            values=scores[k:k+len(p['pool'])];k+=len(p['pool']);order=np.argsort(-values,kind='stable')
            hard=[p['pool'][j] for j in order[:6]];lex=[a for a in p['lexical'] if a not in hard][:4]
            if len(lex)<4:lex += [p['pool'][j] for j in order if p['pool'][j] not in hard+lex][:4-len(lex)]
            assert len(hard)==6 and len(lex)==4
            g['negative_ids']=hard+lex+[p['random']]
            g['negative_sources']={**{a:'student_hard' for a in hard},**{a:'lexical_hard' for a in lex},p['random']:'random'}
            g['student_mining_scores']={a:float(s) for a,s in zip(p['pool'],values) if a in hard+lex}
        if (offset+len(batch))%1008==0 or offset+len(batch)==len(groups):print(json.dumps({'student_mined_groups':offset+len(batch),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    fingerprint=student.fingerprint();studentpairs=student.pairs_scored;del student;torch.cuda.empty_cache()
    write_jsonl(groups,out/'train_groups_mined.jsonl')
    teacher=LocalCrossEncoder(ROOT/'../models/ms-marco-MiniLM-L6-v2','cuda',64,512)
    missing=[(g,a) for g in groups for a in g['positive_ids']+g['negative_ids'] if a not in g['teacher_scores']]
    for i in range(0,len(missing),256):
        batch=missing[i:i+256];scores=teacher.score_pairs([(g['question'],lookup[a]) for g,a in batch])
        for (g,a),s in zip(batch,scores):g['teacher_scores'][a]=float(s)
        if i%16384==0:print(json.dumps({'new_teacher_pairs':min(i+256,len(missing)),'total':len(missing)}),flush=True)
    ambiguous=Counter()
    for g in groups:
        assert not set(g['positive_ids']+g['negative_ids'])&forbidden
        g['teacher_scores']={a:g['teacher_scores'][a] for a in g['positive_ids']+g['negative_ids']}
        best=max(g['teacher_scores'][a] for a in g['positive_ids'])
        ambiguous[g['source_domain']]+=sum(g['teacher_scores'][a]>=best-1 for a in g['negative_ids'])
    (out/'answers.jsonl').write_bytes((fixture/'answers.jsonl').read_bytes());write_jsonl(groups,out/'train_groups.jsonl')
    write_json({'forbidden_sources':sorted(heldsources),'forbidden_answer_ids':len(forbidden),'remaining_forbidden_pairs':0,
        'legacy_insuranceqa_shared_labels':'Original FAQ split has historical shared labels; retained and separately disclosed.',
        'teacher_ambiguous_negative_pairs':dict(ambiguous),'ambiguity_is_not_positive_relabeling':True},out/'source_isolation.json')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'training_questions':len(groups),'source_domains':dict(Counter(g['source_domain'] for g in groups)),
        'available_positive_pairs':sum(len(g['positive_ids']) for g in groups),'mined_negative_pairs':sum(len(g['negative_ids']) for g in groups),
        'teacher':teacher.fingerprint(),'student_miner':fingerprint,'student_pairs_scored':studentpairs,'additional_teacher_pairs':teacher.pairs_scored,
        'parent_training_manifest_sha256':sha(old/'manifest.lock.json'),'fixture_sha256':sha(fixture/'manifest.lock.json'),
        'seconds':time.perf_counter()-start,'code_sha256':sha(Path(__file__)),
        'retrieval_implementation':'Batched dense products; CSC query-column BM25 with 1e-12 parity checks on 21 training queries. Training-only candidate mining.',
        'files':{n:sha(out/n) for n in ['answers.jsonl','train_groups.jsonl','source_isolation.json']}},out/'manifest.lock.json')
    print(json.dumps({'completed':True,'groups':len(groups),'seconds':round(time.perf_counter()-start,1)}),flush=True)

if __name__=='__main__':run()
