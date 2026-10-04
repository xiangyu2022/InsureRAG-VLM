"""Search local research corpora with the frozen retention-selected reranker."""
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
from src.insurerag_vlm.retention import anchored_cross_scores


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);start=time.perf_counter()
    if not args.question.strip() or not 1<=args.top_k<=100:raise ValueError('Nonempty question and top-k 1..100 required')
    lockpath=ROOT/'reports/retention_v2/selection.lock.json';lock=json.loads(lockpath.read_text(encoding='utf8'))
    if lock['selected_on']!='valid':raise ValueError('Frozen validation selection required')
    for name,expected in lock['code_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen ranking code changed')
    modelpath=args.reranker_model or (ROOT/f"../models/insurerag-retention-v2/epoch-{lock['epoch']}" if lock['promoted'] else ROOT/'../models/insurerag-domain-reranker-v2/epoch-1')
    if sha(modelpath/'model.safetensors')!=lock['selected_weights_sha256']:raise ValueError('Wrong selected weights')
    if args.corpus=='multidomain':
        index=ROOT/'reports/multidomain_v1/index_cache';manifest=json.loads((index/'manifest.json').read_text(encoding='utf8'))
        answerpath=ROOT/'data/training/retention_v2/answers.jsonl'
        if sha(answerpath)!=manifest['answer_file_sha256']:raise ValueError('Corpus changed')
        answers=read_jsonl(answerpath);embedding_info=manifest['embedding'];vectors=index/'answer_embeddings.npy';expected=manifest['answer_embeddings_sha256']
    elif args.corpus=='insuranceqa':
        folder=ROOT/'data/benchmarks/insuranceqa_v2';manifest=json.loads((folder/'manifest.lock.json').read_text(encoding='utf8'))
        if sha(folder/'answers.jsonl')!=manifest['files']['answers.jsonl']:raise ValueError('Corpus changed')
        answers=sorted(read_jsonl(folder/'answers.jsonl'),key=lambda a:int(a['id']))
        proto=json.loads((ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json').read_text(encoding='utf8'))
        embedding_info=proto['embedding'];vectors=ROOT/'reports/insuranceqa_v2/retrieval_frozen/answer_embeddings.npy';expected=proto['embedding_cache_sha256']['answer_embeddings.npy']
    else:
        index=ROOT/'reports/hicric_public_v1/bge_index';proto=json.loads((index/'protocol.json').read_text(encoding='utf8'))
        done=json.loads((index/'completion.json').read_text(encoding='utf8'));answerpath=ROOT/'data/research_corpus/hicric_public_v1/rag_snippets.jsonl'
        if done['status']!='completed' or sha(index/'protocol.json')!=done['protocol_sha256'] or sha(answerpath)!=proto['records_sha256']:raise ValueError('Corpus/index changed')
        answers=[{**r,'id':r['record_id']} for r in read_jsonl(answerpath)];embedding_info=proto['embedding']
        vectors=index/'answer_embeddings.npy';expected=done['answer_embeddings_sha256']
    if sha(vectors)!=expected:raise ValueError('Index vectors changed')
    dense=np.load(vectors)
    if len(dense)!=len(answers):raise ValueError('Vector count mismatch')
    encoder=EmbeddingRetriever(str(args.embedding_model),use_hf_api=False,pooling='cls',max_length=512,query_instruction=embedding_info['query_instruction'])
    verify_files(encoder.index_fingerprint()['checkpoint_files_sha256'],embedding_info['checkpoint_files_sha256'])
    _,emb=encoder._ensure_local_transformer();emb.to(args.device)
    q=encoder.embed_texts([encoder.query_instruction+args.question])[0];d=dense@q;b=SparseBM25([a['text'] for a in answers]).scores(args.question)
    do=ranked(d);bo=ranked(b,positive_only=True);union=sorted(set(do)|set(bo));texts=[answers[j]['text'] for j in union]
    model=DomainCrossEncoder(modelpath,args.device)
    protocol=ROOT/f"reports/retention_v2/valid_new_epoch_{lock['epoch']}/protocol.json" if lock['promoted'] else ROOT/'reports/retention_v2/valid_new_previous/protocol.json'
    verify_files(model.fingerprint()['files_sha256'],json.loads(protocol.read_text(encoding='utf8'))['model']['files_sha256'])
    cross=model.score(args.question,texts);passes=1
    if lock['domain_weight']<1:
        if sha(args.public_model/'model.safetensors')!=lock['public_weights_sha256']:raise ValueError('Wrong public weights')
        teacher=DomainCrossEncoder(args.public_model,args.device)
        verify_files(teacher.fingerprint()['files_sha256'],json.loads((ROOT/'reports/retention_v2/valid_new_public/protocol.json').read_text(encoding='utf8'))['model']['files_sha256'])
        cross=anchored_cross_scores(cross,teacher.score(args.question,texts),lock['domain_weight']);passes=2
    row={'candidate_ids':[answers[j]['id'] for j in union],'dense':d[union].tolist(),'sparse':b[union].tolist(),'cross':cross.tolist(),
         'bge_ids':[answers[j]['id'] for j in do]}
    order=rank_row(row,lock['config']);lookup={a['id']:a for a in answers}
    result={'question':args.question,'corpus':args.corpus,'answer_candidates':len(answers),'candidates_scored_per_model':len(union),
            'reranker_passes':passes,'selected_epoch':lock['epoch'],'domain_weight':lock['domain_weight'],
            'generation_calls':0,'research_retrieval_only':True,'seconds_including_loading':time.perf_counter()-start,
            'selection_lock_sha256':sha(lockpath),'model_weights_sha256':lock['selected_weights_sha256'],
            'results':[{'rank':i+1,'answer_id':a,'text':lookup[a]['text'],**{k:lookup[a][k] for k in ['source_url','source_group','doc_id','start_word','end_word','page_unit'] if k in lookup[a]}} for i,a in enumerate(order[:args.top_k])]}
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf8')
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--question',required=True)
    p.add_argument('--corpus',choices=['insuranceqa','multidomain','hicric'],default='insuranceqa')
    p.add_argument('--embedding-model',type=Path,default=ROOT/'../models/bge-small-en-v1.5');p.add_argument('--reranker-model',type=Path)
    p.add_argument('--public-model',type=Path,default=ROOT/'../models/ms-marco-MiniLM-L6-v2')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--top-k',type=int,default=3);p.add_argument('--output',type=Path)
    run(p.parse_args())
