"""Run the validation-selected retrieval prototype, with explicit report scope."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha
from scripts.eval_evidence_reranker import check_contract,modelspec,RUN,QUERY
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.query_adaptation import QueryEncoder,blend_queries
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder
from scripts.verify_evidence_inputs import verify_fixed_query_inputs


def main(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();check_contract();fixed=verify_fixed_query_inputs();lock=json.loads((RUN/'selection.lock.json').read_text(encoding='utf8'))
    if not args.question.strip() or not 1<=args.top_k<=100:raise ValueError('Nonempty query and top-k 1..100 required')
    if args.corpus=='finqa' and not args.report:raise ValueError('FinQA requires explicit --report COMPANY/YEAR, e.g. ADI/2009')
    if args.corpus!='finqa' and args.report:raise ValueError('--report is supported only for the FinQA corpus')
    source={'insuranceqa':'data/benchmarks/insuranceqa_v2/answers.jsonl',
            'condition':'data/benchmarks/condition_v1/answers.jsonl','fiqa':'data/benchmarks/fiqa_v1/answers.jsonl',
            'finqa':'data/benchmarks/finqa_evidence_v1/answers.jsonl'}[args.corpus]
    answers=read_jsonl(ROOT/source)
    if args.report:answers=[a for a in answers if a['source_group']==args.report]
    if not answers:raise ValueError('Unknown or empty annual-report scope')
    cache=RUN/'index_cache';manifest=json.loads((cache/'manifest.json').read_text(encoding='utf8'))
    allpath=ROOT/'data/training/evidence_reranker_v1/answers.jsonl'
    assert sha(allpath)==manifest['answer_file_sha256'] and sha(cache/'answer_embeddings.npy')==manifest['answer_embeddings_sha256']
    allanswers=read_jsonl(allpath);lookup={a['id']:i for i,a in enumerate(allanswers)};ix=[lookup[a['id']] for a in answers]
    assert all(a['text']==allanswers[i]['text'] for a,i in zip(answers,ix))
    dense=np.load(cache/'answer_embeddings.npy')[ix]
    encoder=QueryEncoder(ROOT/'../models/bge-small-en-v1.5',args.device);base=encoder.encode([args.question]);del encoder
    if args.device=='cuda':torch.cuda.empty_cache()
    encoder=QueryEncoder(QUERY,args.device);adapted=encoder.encode([args.question]);del encoder
    if args.device=='cuda':torch.cuda.empty_cache()
    query=blend_queries(base,adapted,.5)[0];d=dense@query;b=SparseBM25([a['text'] for a in answers]).scores(args.question)
    do,bo=ranked(d),ranked(b,positive_only=True);union=sorted(set(do)|set(bo))
    modelpath=modelspec(lock['config']['model']);assert sha(modelpath/'model.safetensors')==lock['weights_sha256']
    model=DomainCrossEncoder(modelpath,args.device)
    # Validation-scoring fingerprint already freezes every inference file, not only weights.
    validation=json.loads((RUN/'valid_scores'/lock['config']['model']/'protocol.json').read_text(encoding='utf8'))
    assert model.fingerprint()['files_sha256']==validation['model_fingerprint']['files_sha256']
    cross=model.score(args.question,[answers[i]['text'] for i in union])
    row={'candidate_ids':[answers[i]['id'] for i in union],'bge_ids':[answers[i]['id'] for i in do],
         'dense':d[union].tolist(),'sparse':b[union].tolist(),'cross':cross.tolist()}
    cfg={'pool':'union200','lexical_weight':.2,'cross_weight':lock['config']['cross_weight']};order=rank_row(row,cfg)
    byid={a['id']:a for a in answers}
    result={'question':args.question,'corpus':args.corpus,'annual_report_scope':args.report,'corpus_candidates':len(answers),
            'reranked_candidates':len(union),'selected_config':lock['config'],'new_training_promoted':lock['new_training_promoted'],
            'selection_lock_sha256':sha(RUN/'selection.lock.json'),'selected_weights_sha256':lock['weights_sha256'],
            'fixed_encoders_match_inherited_inventories':bool(fixed),
            'query_encoder_passes':2,'generation_calls':0,'numerical_program_execution':False,'expert_answer_adjudication':False,
            'seconds_including_loading':time.perf_counter()-start,
            'results':[{'rank':i+1,'answer_id':a,'text':byid[a]['text'],
                        **{k:byid[a][k] for k in ['source_group','source_page','source_url','evidence_type','upstream_evidence_key'] if k in byid[a]}}
                       for i,a in enumerate(order[:args.top_k])]}
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        if args.output.exists():raise ValueError('Output already exists')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf8')
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--question',required=True)
    p.add_argument('--corpus',choices=['insuranceqa','condition','fiqa','finqa'],default='insuranceqa')
    p.add_argument('--report');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--output',type=Path);main(p.parse_args())
