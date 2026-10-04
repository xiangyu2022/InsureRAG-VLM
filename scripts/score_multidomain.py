"""Build label-free mixed-corpus candidates, or score a local model on frozen candidates."""
import argparse
from datetime import datetime,timezone
import json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4)
    fixture=ROOT/'data/benchmarks/multidomain_v1';data=ROOT/'data/training/retention_v1';index=ROOT/'reports/multidomain_v1/index_cache'
    lock=json.loads((fixture/'manifest.lock.json').read_text(encoding='utf8'))
    if sha(fixture/f'{args.split}.jsonl')!=lock['files'][f'{args.split}.jsonl']:raise ValueError('Question fixture changed')
    indexlock=json.loads((index/'manifest.json').read_text(encoding='utf8'))
    if sha(data/'answers.jsonl')!=indexlock['answer_file_sha256']:raise ValueError('Answer corpus changed')
    cases=read_jsonl(fixture/f'{args.split}.jsonl');answers=read_jsonl(data/'answers.jsonl');lookup={a['id']:a['text'] for a in answers}
    selection=json.loads(args.selection.read_text(encoding='utf8')) if args.selection else None
    if args.split=='test' and (not selection or selection['selected_on']!='valid'):raise ValueError('Fresh test requires frozen selection')
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    codefiles=[Path(__file__),ROOT/'src/insurerag_vlm/domain_reranker.py',ROOT/'src/insurerag_vlm/reranker.py',ROOT/'scripts/eval_insuranceqa_scale.py']
    code={p.relative_to(ROOT).as_posix():sha(p) for p in codefiles}
    if args.mode=='candidates':
        if sha(index/'answer_embeddings.npy')!=indexlock['answer_embeddings_sha256']:raise ValueError('Index changed')
        encoder=EmbeddingRetriever(str(args.model),use_hf_api=False,pooling='cls',max_length=512,query_instruction=indexlock['embedding']['query_instruction'])
        strip=lambda d:{k:v for k,v in d.items() if k!='download_provenance.json'}
        if strip(encoder.index_fingerprint()['checkpoint_files_sha256'])!=strip(indexlock['embedding']['checkpoint_files_sha256']):raise ValueError('Wrong embedding model')
        protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,'split':args.split,'questions':len(cases),
            'fixture_lock_sha256':sha(fixture/'manifest.lock.json'),'index_manifest_sha256':sha(index/'manifest.json'),
            'answer_count':len(answers),'selection_lock_sha256':sha(args.selection) if args.selection else None,
            'code_sha256':code,'embedding':encoder.index_fingerprint(),'gold_injection':False}
        write_json(protocol,args.output/'protocol.json')
        _,model=encoder._ensure_local_transformer();model.to('cuda')
        queryvectors=np.vstack([encoder.embed_texts([encoder.query_instruction+c['question'] for c in cases[i:i+128]]) for i in range(0,len(cases),128)])
        dense=np.load(index/'answer_embeddings.npy');sparse=SparseBM25([a['text'] for a in answers]);np.save(args.output/'query_embeddings.npy',queryvectors)
        with (args.output/'scores.jsonl').open('w',encoding='utf8') as handle:
            for i,case in enumerate(cases):
                d=dense@queryvectors[i];b=sparse.scores(case['question']);do=ranked(d);bo=ranked(b,positive_only=True);union=sorted(set(do)|set(bo))
                row={'id':case['id'],'candidate_ids':[answers[j]['id'] for j in union],'bge_ids':[answers[j]['id'] for j in do],
                     'bm25_ids':[answers[j]['id'] for j in bo],'dense':d[union].tolist(),'sparse':b[union].tolist()}
                handle.write(json.dumps(row)+'\n')
    else:
        source=json.loads((args.candidates/'completion.json').read_text(encoding='utf8'))
        if sha(args.candidates/'scores.jsonl')!=source['scores_sha256']:raise ValueError('Candidate artifact changed')
        candidates=read_jsonl(args.candidates/'scores.jsonl')
        if [r['id'] for r in candidates]!=[c['id'] for c in cases]:raise ValueError('Wrong candidate split/order')
        model=DomainCrossEncoder(args.model,'cuda',64,512)
        protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,'split':args.split,'questions':len(cases),
            'fixture_lock_sha256':sha(fixture/'manifest.lock.json'),'candidate_source_sha256':sha(args.candidates/'scores.jsonl'),
            'selection_lock_sha256':sha(args.selection) if args.selection else None,'model':model.fingerprint(),'code_sha256':code,'gold_injection':False}
        if args.split=='test':
            allowed={selection['selected_weights_sha256'],selection['public_weights_sha256'],selection['previous_weights_sha256']}
            if sha(args.model/'model.safetensors') not in allowed:raise ValueError('Model not authorized by frozen test plan')
        write_json(protocol,args.output/'protocol.json')
        questions={c['id']:c['question'] for c in cases}
        with (args.output/'scores.jsonl').open('w',encoding='utf8') as handle:
            for offset in range(0,len(candidates),8):
                batch=candidates[offset:offset+8];pairs=[(questions[r['id']],lookup[a]) for r in batch for a in r['candidate_ids']]
                scores=model.score_pairs(pairs);position=0
                for row in batch:
                    n=len(row['candidate_ids']);handle.write(json.dumps({**row,'cross':scores[position:position+n].tolist()})+'\n');position+=n
                handle.flush()
                if (offset+len(batch))%200==0 or offset+len(batch)==len(candidates):
                    print(json.dumps({'model':args.model.name,'split':args.split,'scored':offset+len(batch),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    if code!={p.relative_to(ROOT).as_posix():sha(p) for p in codefiles}:raise ValueError('Scoring code changed')
    write_json({'status':'completed','seconds':time.perf_counter()-start,'scores_sha256':sha(args.output/'scores.jsonl'),
        'protocol_sha256':sha(args.output/'protocol.json'),'questions':len(cases)},args.output/'completion.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['candidates','score']);p.add_argument('--split',choices=['valid','test'],required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',type=Path);p.add_argument('--selection',type=Path)
    run(p.parse_args())
