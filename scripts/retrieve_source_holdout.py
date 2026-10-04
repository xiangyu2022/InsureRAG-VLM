"""Label-free retrieval over pinned historical answers plus new publisher FAQ text."""
import argparse,gc,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
LOCAL=ROOT/'reports/source_holdout_v1/local'
MODELS=ROOT.parent/'models'
HISTORICAL=ROOT.parent/'insurerag-main-improvement'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def verify_test_lock(arm):
    path=ROOT/'reports/source_holdout_v1/selection.lock.json'
    if not path.exists():raise ValueError('Test remains sealed until strategy lock')
    lock=read(path)
    if arm not in lock['frozen_test_comparators']:raise ValueError('Arm not registered before test opening')
    for relative,digest in lock['inference_files_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Frozen inference code changed: '+relative)
    if sha(LOCAL/'sealed/test.json')!=lock['sealed_test_sha256']:raise ValueError('Sealed test changed')

def retrieve(requests, output, arm, device):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from scripts.eval_insuranceqa_scale import SparseBM25,ranked
    from scripts.eval_insuranceqa_reranker import rank_row
    from src.insurerag_vlm.query_adaptation import QueryEncoder,blend_queries
    from src.insurerag_vlm.domain_reranker import DomainCrossEncoder
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    new=read(LOCAL/'sealed/answers.json')
    oldfile=HISTORICAL/'data/training/evidence_reranker_v1/answers.jsonl'
    cache=HISTORICAL/'reports/evidence_reranker_v1/index_cache'
    manifest=read(cache/'manifest.json')
    for p,key in [(oldfile,'answer_file_sha256'),(cache/'answer_embeddings.npy','answer_embeddings_sha256')]:
        if sha(p)!=manifest[key]:raise ValueError('Historical corpus/index checksum mismatch')
    with oldfile.open(encoding='utf8') as f: old=[json.loads(line) for line in f if line.strip()]
    if len(old)!=192231:raise ValueError('Historical denominator changed')
    answers=old+new
    if len({r['id'] for r in answers})!=len(answers):raise ValueError('Duplicate evidence ID')
    base=MODELS/'bge-small-en-v1.5'
    query=MODELS/'insurerag-query-adaptation-v1-seed-42/epoch-2'
    crosspath=MODELS/'insurerag-evidence-reranker-v1-seed-123/epoch-2'
    expected={base:'3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
              query:'e067a9a14428aabb32dbd2f0e84a7af31bcb8290ffc6fe422faed906d4d5b857',
              crosspath:'697d3a725ad9d684c1b978365bda314114e8dbec2a27c2841a07f028ed2e81b8'}
    for p,digest in expected.items():
        if sha(p/'model.safetensors')!=digest:raise ValueError('Model weights mismatch')
    with threadpool_limits(4):
        encoder=QueryEncoder(base,device)
        dense_path=LOCAL/'new_answer_embeddings.npy'
        dense_manifest=LOCAL/'new_answer_embeddings.json'
        if dense_path.exists():
            dm=read(dense_manifest)
            if dm['answers_sha256']!=sha(LOCAL/'sealed/answers.json') or dm['embeddings_sha256']!=sha(dense_path):raise ValueError('New index mismatch')
            newdense=np.load(dense_path)
        else:
            vectors=[]
            with torch.inference_mode():
                for i in range(0,len(new),16):
                    batch=encoder.tokenizer([r['text'] for r in new[i:i+16]],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                    vectors.append(torch.nn.functional.normalize(encoder.model(**batch).last_hidden_state[:,0].float(),dim=-1).cpu().numpy())
            newdense=np.vstack(vectors);np.save(dense_path,newdense)
            write(dense_manifest,{'answers_sha256':sha(LOCAL/'sealed/answers.json'),'embeddings_sha256':sha(dense_path),
                                 'model_weights_sha256':expected[base],'document_instruction':'','max_tokens':512,'pooling':'CLS','dtype':'float32'})
        t=time.perf_counter();original=encoder.encode([r['question'] for r in requests]);original_seconds=time.perf_counter()-t
        del encoder;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        if arm=='original_query':
            queries=original;query_seconds=original_seconds
        else:
            encoder=QueryEncoder(query,device);t=time.perf_counter()
            adapted=encoder.encode([r['question'] for r in requests]);query_seconds=original_seconds+time.perf_counter()-t
            queries=blend_queries(original,adapted,.5)
            del encoder;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        dense=np.vstack([np.load(cache/'answer_embeddings.npy',mmap_mode='r'),newdense])
        print(json.dumps({'phase':'sparse_index','answers':len(answers)}),flush=True)
        sparse=SparseBM25([r['text'] for r in answers])
        model=DomainCrossEncoder(crosspath,device)
        packer=DocumentRetrievalPipeline(ModelConfig(vlm_model='local-extractive',retrieval_model='local-hashing',max_context_chars=8000,max_page_chars=2400,max_answer_pages=5))
        index_seconds=time.perf_counter()-start
        config={'pool':'union200','lexical_weight':.2,'cross_weight':{'cross_only':1.0,'cross_quarter':.25}.get(arm,.5)}
        byid={r['id']:r for r in answers}
        protocol={'started_utc':datetime.now(timezone.utc).isoformat(),'arm':arm,'device':device,'historical_answers':len(old),'new_answer_chunks':len(new),
                  'historical_answers_sha256':sha(oldfile),'historical_embeddings_sha256':manifest['answer_embeddings_sha256'],
                  'new_answers_sha256':sha(LOCAL/'sealed/answers.json'),'new_embeddings_sha256':sha(dense_path),
                  'weights_sha256':{p.name+'_'+p.parent.name:v for p,v in expected.items()},'rank_config':config,
                  'query_alpha':0 if arm=='original_query' else .5,'retrieval_output_depth':10,'packing_top_k':5,'context_chars':8000,
                  'page_chars':2400,'query_encoding_batch_seconds':query_seconds,'setup_seconds':index_seconds,
                  'script_sha256':sha(Path(__file__)),'label_fields_received':False,'inference_fields':['id','question'],
                  'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
                  'reranker_fingerprint':model.fingerprint()}
        write(output/'protocol.json',protocol)
        with (output/'retrieval.jsonl').open('w',encoding='utf8') as handle:
            for request,qv in zip(requests,queries):
                t=time.perf_counter();d=dense@qv;b=sparse.scores(request['question'])
                do=ranked(d);bo=ranked(b,positive_only=True);union=sorted(set(do)|set(bo))
                c=model.score(request['question'],[answers[i]['text'] for i in union])
                row={'candidate_ids':[answers[i]['id'] for i in union],'bge_ids':[answers[i]['id'] for i in do],
                     'dense':d[union].tolist(),'sparse':b[union].tolist(),'cross':c.tolist()}
                order=rank_row(row,config)
                selected=[{'rank':i+1,'answer_id':aid,**{k:v for k,v in byid[aid].items() if k!='id'}} for i,aid in enumerate(order[:10])]
                pages=[{'source':r['answer_id'],'text_snippet':r['text'],'document_type':'public_qa',
                        'primary_clause_type':'general','section_anchor':' | '.join(str(r.get(k,'')) for k in ['source_group','source_url'])}
                       for r in selected[:5]]
                context=packer.pack_long_context(pages,5)
                record={**request,'arm':arm,'order':order[:10],'results':selected,'context':context,
                        'reranked_candidates':len(union),'candidate_scores':row,
                        'retrieval_seconds':time.perf_counter()-t,'amortized_query_encode_seconds':query_seconds/len(requests)}
                handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush()
                print(json.dumps({'phase':'retrieved','completed':len(order),'id':request['id'],'seconds':record['retrieval_seconds']}),flush=True)
        write(output/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'questions':len(requests),
                                       'retrieval_sha256':sha(output/'retrieval.jsonl'),'protocol_sha256':sha(output/'protocol.json')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--arm',choices=['baseline','original_query','cross_only','cross_quarter'],default='baseline')
    p.add_argument('--run-suffix',default='',help='Distinct artifact directory for timing/parity reruns; never overwrite prior evidence')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    if args.split=='test':verify_test_lock(args.arm)
    manifest=read(LOCAL/'sealed/manifest.json');path=LOCAL/'sealed'/(args.split+'.json')
    if sha(path)!=manifest['files_sha256'][path.name]:raise ValueError('Sealed split changed')
    cases=read(path);requests=[{'id':r['id'],'question':r['question']} for r in cases]
    if args.run_suffix and not args.run_suffix.replace('_','').isalnum():raise ValueError('Run suffix must contain only letters, numbers or underscores')
    out=LOCAL/(args.split+'_'+args.arm+('_'+args.run_suffix if args.run_suffix else ''));out.mkdir(exist_ok=False)
    retrieve(requests,out,args.arm,args.device)
if __name__=='__main__':main()
