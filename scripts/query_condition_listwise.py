"""Query the latest validation-selected model against a hash-verified local corpus."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from scripts.eval_insuranceqa_reranker import rank_row
from scripts.query_domain_reranker import verify_files
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);start=time.perf_counter()
    if not args.question.strip() or not 1<=args.top_k<=100:raise ValueError('Nonempty question and top-k 1..100 required')
    run=ROOT/'reports/condition_listwise_v1';lockpath=run/'selection.lock.json';lock=json.loads(lockpath.read_text(encoding='utf8'))
    assert lock['selected_on']=='valid'
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen ranking implementation changed')
    name=lock['selected_model'];modelpath=args.reranker_model or (ROOT/f'../models/insurerag-condition-listwise-v1-seed-{name.split("_")[-1]}/epoch-1' if lock['promoted_weights'] else ROOT/'../models/insurerag-retention-v2/epoch-2')
    if sha(modelpath/'model.safetensors')!=lock['selected_weights_sha256']:raise ValueError('Wrong selected model weights')
    if args.corpus=='condition':
        index=ROOT/'reports/condition_v1/index_cache';m=json.loads((index/'manifest.json').read_text(encoding='utf8'));answerpath=ROOT/'data/benchmarks/condition_v1/answers.jsonl'
        if sha(answerpath)!=m['answer_file_sha256']:raise ValueError('Answer corpus changed')
        answers=read_jsonl(answerpath);vectors=index/'answer_embeddings.npy';expected=m['answer_embeddings_sha256'];embedding=m['embedding']
    elif args.corpus=='insuranceqa':
        directory=ROOT/'data/benchmarks/insuranceqa_v2';m=json.loads((directory/'manifest.lock.json').read_text(encoding='utf8'))
        if sha(directory/'answers.jsonl')!=m['files']['answers.jsonl']:raise ValueError('FAQ corpus changed')
        answers=sorted(read_jsonl(directory/'answers.jsonl'),key=lambda a:int(a['id']));m=json.loads((ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json').read_text(encoding='utf8'))
        embedding=m['embedding'];vectors=ROOT/'reports/insuranceqa_v2/retrieval_frozen/answer_embeddings.npy';expected=m['embedding_cache_sha256']['answer_embeddings.npy']
    else:
        index=ROOT/'reports/hicric_public_v1/bge_index';m=json.loads((index/'protocol.json').read_text(encoding='utf8'));done=json.loads((index/'completion.json').read_text(encoding='utf8'))
        answerpath=ROOT/'data/research_corpus/hicric_public_v1/rag_snippets.jsonl'
        if done['status']!='completed' or sha(index/'protocol.json')!=done['protocol_sha256'] or sha(answerpath)!=m['records_sha256']:raise ValueError('Snippet corpus changed')
        answers=[{**a,'id':a['record_id']} for a in read_jsonl(answerpath)];embedding=m['embedding'];vectors=index/'answer_embeddings.npy';expected=done['answer_embeddings_sha256']
    if sha(vectors)!=expected:raise ValueError('Embedding vectors changed')
    dense=np.load(vectors);assert len(dense)==len(answers)
    encoder=EmbeddingRetriever(str(args.embedding_model),use_hf_api=False,pooling='cls',max_length=512,query_instruction=embedding['query_instruction'])
    verify_files(encoder.index_fingerprint()['checkpoint_files_sha256'],embedding['checkpoint_files_sha256'])
    _,emb=encoder._ensure_local_transformer();emb.to(args.device)
    q=encoder.embed_texts([encoder.query_instruction+args.question])[0];d=dense@q;b=SparseBM25([a['text'] for a in answers]).scores(args.question)
    do=ranked(d);bo=ranked(b,positive_only=True);union=sorted(set(do)|set(bo));model=DomainCrossEncoder(modelpath,args.device)
    protocol=run/f'valid_mixed_{name}/protocol.json' if lock['promoted_weights'] else ROOT/'reports/retention_v2/valid_new_epoch_2/protocol.json'
    verify_files(model.fingerprint()['files_sha256'],json.loads(protocol.read_text(encoding='utf8'))['model']['files_sha256'])
    cross=model.score(args.question,[answers[j]['text'] for j in union])
    row={'candidate_ids':[answers[j]['id'] for j in union],'dense':d[union].tolist(),'sparse':b[union].tolist(),'cross':cross.tolist(),'bge_ids':[answers[j]['id'] for j in do]}
    order=rank_row(row,lock['config']);lookup={a['id']:a for a in answers}
    result={'question':args.question,'corpus':args.corpus,'answer_candidates':len(answers),'candidates_scored':len(union),
        'config':lock['config'],'selected_model':name,'generation_calls':0,'research_retrieval_only':True,
        'selection_lock_sha256':sha(lockpath),'model_weights_sha256':lock['selected_weights_sha256'],'seconds_including_loading':time.perf_counter()-start,
        'results':[{'rank':i+1,'answer_id':a,'text':lookup[a]['text'],**{k:lookup[a][k] for k in ['source_url','source_group','doc_id','start_word','end_word','page_unit'] if k in lookup[a]}} for i,a in enumerate(order[:args.top_k])]}
    output=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(output,encoding='utf8')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--question',required=True)
    p.add_argument('--corpus',choices=['condition','insuranceqa','hicric'],default='condition');p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--embedding-model',type=Path,default=ROOT/'../models/bge-small-en-v1.5');p.add_argument('--reranker-model',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path);run(p.parse_args())
