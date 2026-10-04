"""Score frozen candidates or build the source-held-out condition benchmark candidates."""
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4)
    cases=read_jsonl(args.questions);answers=read_jsonl(args.answers);lookup={a['id']:a['text'] for a in answers}
    run=ROOT/'reports/condition_v1';plan=json.loads((run/'selection_protocol.json').read_text(encoding='utf8'))
    assert sha(Path(__file__))==plan['scoring_code_sha256']
    selection=json.loads((run/'selection.lock.json').read_text(encoding='utf8')) if args.split=='test' else None
    if selection:
        assert selection['selected_on']=='valid'
        if args.mode=='score':assert sha(args.model/'model.safetensors') in {selection[k] for k in ['selected_weights_sha256','previous_weights_sha256','public_weights_sha256']}
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'split':args.split,'questions':len(cases),'question_file_sha256':sha(args.questions),
        'answer_file_sha256':sha(args.answers),'answer_count':len(answers),'gold_injection':False,'code_sha256':sha(Path(__file__)),
        'selection_lock_sha256':sha(run/'selection.lock.json') if selection else None,'selection_protocol_sha256':sha(run/'selection_protocol.json')}
    if args.mode=='candidates':
        from scripts.eval_insuranceqa_scale import SparseBM25,ranked
        from src.insurerag_vlm.retriever import EmbeddingRetriever
        index=run/'index_cache';indexlock=json.loads((index/'manifest.json').read_text(encoding='utf8'))
        assert sha(args.answers)==indexlock['answer_file_sha256'] and sha(index/'answer_embeddings.npy')==indexlock['answer_embeddings_sha256']
        encoder=EmbeddingRetriever(str(args.model),use_hf_api=False,pooling='cls',max_length=512,query_instruction=indexlock['embedding']['query_instruction'])
        strip=lambda d:{k:v for k,v in d.items() if k!='download_provenance.json'}
        assert strip(encoder.index_fingerprint()['checkpoint_files_sha256'])==strip(indexlock['embedding']['checkpoint_files_sha256'])
        _,model=encoder._ensure_local_transformer();model.to('cuda')
        vectors=np.vstack([encoder.embed_texts([encoder.query_instruction+c['question'] for c in cases[i:i+128]]) for i in range(0,len(cases),128)])
        np.save(args.output/'query_embeddings.npy',vectors);dense=np.load(index/'answer_embeddings.npy');sparse=SparseBM25([a['text'] for a in answers])
        protocol['embedding']=encoder.index_fingerprint();protocol['index_sha256']=sha(index/'manifest.json');write_json(protocol,args.output/'protocol.json')
        with (args.output/'scores.jsonl').open('w',encoding='utf8') as handle:
            for c,v in zip(cases,vectors):
                d=dense@v;b=sparse.scores(c['question']);do=ranked(d);bo=ranked(b,positive_only=True);union=sorted(set(do)|set(bo))
                handle.write(json.dumps({'id':c['id'],'candidate_ids':[answers[j]['id'] for j in union],
                    'bge_ids':[answers[j]['id'] for j in do],'bm25_ids':[answers[j]['id'] for j in bo],'dense':d[union].tolist(),'sparse':b[union].tolist()})+'\n')
    else:
        rows=read_jsonl(args.candidates/'scores.jsonl')
        completion=args.candidates/'completion.json'
        if completion.exists():expected=json.loads(completion.read_text(encoding='utf8'))['scores_sha256']
        else:expected=plan['historical_score_hashes'][str((args.candidates/'scores.jsonl').relative_to(ROOT)).replace('\\','/')]
        assert sha(args.candidates/'scores.jsonl')==expected
        assert [r['id'] for r in rows]==[c['id'] for c in cases]
        model=DomainCrossEncoder(args.model,'cuda',64,512);protocol['model']=model.fingerprint();protocol['candidate_source_sha256']=expected
        write_json(protocol,args.output/'protocol.json')
        with (args.output/'scores.jsonl').open('w',encoding='utf8') as handle:
            for offset in range(0,len(cases),8):
                batch=rows[offset:offset+8];cs=cases[offset:offset+8];pairs=[(c['question'],lookup[a]) for r,c in zip(batch,cs) for a in r['candidate_ids']]
                scores=model.score_pairs(pairs);i=0
                for row in batch:
                    n=len(row['candidate_ids']);handle.write(json.dumps({**row,'cross':scores[i:i+n].tolist()})+'\n');i+=n
                handle.flush()
                if (offset+len(batch))%200==0 or offset+len(batch)==len(cases):print(json.dumps({'scored':offset+len(batch),'total':len(cases),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    assert sha(Path(__file__))==protocol['code_sha256']
    write_json({'status':'completed','seconds':time.perf_counter()-start,'questions':len(cases),
        'scores_sha256':sha(args.output/'scores.jsonl'),'protocol_sha256':sha(args.output/'protocol.json')},args.output/'completion.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['candidates','score'])
    for name in ['questions','answers','model','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidates',type=Path);p.add_argument('--split',choices=['valid','test'],required=True);run(p.parse_args())
