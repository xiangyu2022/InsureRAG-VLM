"""Frozen-candidate evaluation with validation-tuned old-model controls."""
from collections import defaultdict
from datetime import datetime,timezone
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_query_adaptation_data import write_jsonl
from scripts.eval_query_adaptation import specs as prior_specs,partitions
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from scripts.eval_insuranceqa_reranker import rank_row
from scripts.analyze_condition_secondary_metrics import clusters
from src.insurerag_vlm.evidence_ranking import evidence_metrics,summarize_evidence
from src.insurerag_vlm.query_adaptation import QueryEncoder,blend_queries
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

RUN=ROOT/'reports/evidence_reranker_v1'
PRIOR=ROOT/'reports/query_adaptation_v1'
PREVIOUS=ROOT/'../models/insurerag-condition-listwise-v1-seed-123/epoch-1'
QUERY=ROOT/'../models/insurerag-query-adaptation-v1-seed-42/epoch-2'
PRIOR_ARM='seed_42_epoch_2_a50'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def modelspec(name):
    if name=='previous':return PREVIOUS
    _,seed,_,epoch=name.split('_')
    return ROOT/f'../models/insurerag-evidence-reranker-v1-seed-{seed}/epoch-{epoch}'


def check_contract():
    p=load(RUN/'selection_protocol.json')
    for n,h in p['frozen_code_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in p['fixture_files_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(PREVIOUS/'model.safetensors')==p['initial_reranker_weights_sha256']
    assert sha(QUERY/'model.safetensors')==p['fixed_query_weights_sha256']
    assert sha(PRIOR/'selection.lock.json')==p['prior_selection_sha256']
    return p


def specs(split):
    return prior_specs(split)+[('finqa',f'data/benchmarks/finqa_evidence_v1/{split}.jsonl',
                               'data/benchmarks/finqa_evidence_v1/answers.jsonl',None)]


def prepare_candidates(split):
    import torch
    plan=check_contract()
    if split=='test':assert (RUN/'selection.lock.json').exists()
    out=RUN/f'{split}_candidates';out.mkdir(exist_ok=False)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'split':split,
              'selection_protocol_sha256':sha(RUN/'selection_protocol.json'),
              'selection_lock_sha256':sha(RUN/'selection.lock.json') if split=='test' else None,
              'query_configuration':'fixed prior alpha .5','document_encoder':'unchanged original BGE',
              'candidate_policy':'dense100 union positive BM25 top100 within declared corpus/scope',
              'gold_injected':False,'reused_prior_reranker_scores_verified':True}
    write_json(protocol,out/'protocol.json')
    parent=load(PRIOR/split/'summary.json');source_pred=defaultdict(dict)
    assert sha(PRIOR/split/'predictions.jsonl')==parent['predictions_sha256']
    for r in read_jsonl(PRIOR/split/'predictions.jsonl'):
        if r['arm'] in ['bge','bm25','rrf',PRIOR_ARM]:source_pred[r['cohort']][r['id'],r['arm']]=r
    for co,qfile,_,_ in prior_specs(split):
        p=PRIOR/split/f'{co}_scores.jsonl';assert sha(p)==parent['score_files_sha256'][p.name]
        rows=[]
        with p.open(encoding='utf8') as f:
            for line in f:
                r=json.loads(line);arm=r['arms'][PRIOR_ARM];cross=dict(zip(r['candidate_ids'],r['cross']))
                rows.append({'id':r['id'],**arm,'previous_cross':[cross[a] for a in arm['candidate_ids']],
                             'public_bge_ids':source_pred[co][r['id'],'bge']['top_answer_ids'],
                             'bm25_ids':source_pred[co][r['id'],'bm25']['top_answer_ids']})
        assert [r['id'] for r in rows]==[q['id'] for q in read_jsonl(ROOT/qfile)]
        for r in rows:
            old_order=rank_row({**r,'cross':r['previous_cross']},{'pool':'union200','lexical_weight':.2,'cross_weight':.5})
            assert old_order==source_pred[co][r['id'],PRIOR_ARM]['top_answer_ids']
        write_jsonl(rows,out/f'{co}.jsonl')
    # All FinQA source documents are indexable; only train-split sources enter gradients.
    data=ROOT/'data/training/evidence_reranker_v1';cache=RUN/'index_cache';cm=load(cache/'manifest.json')
    assert sha(cache/'answer_embeddings.npy')==cm['answer_embeddings_sha256']
    assert sha(data/'answers.jsonl')==cm['answer_file_sha256']
    allanswers=read_jsonl(data/'answers.jsonl');position={a['id']:i for i,a in enumerate(allanswers)}
    dense=np.load(cache/'answer_embeddings.npy');fixture=ROOT/'data/benchmarks/finqa_evidence_v1'
    answers=read_jsonl(fixture/'answers.jsonl');cases=read_jsonl(fixture/f'{split}.jsonl')
    reports=defaultdict(list)
    for a in answers:reports[a['source_group']].append(a)
    encoder=QueryEncoder(ROOT/'../models/bge-small-en-v1.5','cuda');original=encoder.encode([q['question'] for q in cases],64)
    del encoder;torch.cuda.empty_cache()
    encoder=QueryEncoder(QUERY,'cuda');learned=encoder.encode([q['question'] for q in cases],64)
    query=blend_queries(original,learned,.5);del encoder;torch.cuda.empty_cache()
    np.save(out/'finqa_original_queries.npy',original);np.save(out/'finqa_adapted_queries.npy',learned)
    rows=[];sparse_cache={}
    for i,c in enumerate(cases):
        report=c['source_group'];docs=reports[report];ids=[a['id'] for a in docs];dp=dense[[position[a] for a in ids]]
        if report not in sparse_cache:sparse_cache[report]=SparseBM25([a['text'] for a in docs])
        b=sparse_cache[report].scores(c['question']);d=dp@query[i];d0=dp@original[i]
        do,bo=ranked(d),ranked(b,positive_only=True);union=sorted(set(do)|set(bo))
        rows.append({'id':c['id'],'candidate_ids':[ids[j] for j in union],
                     'bge_ids':[ids[j] for j in do],'public_bge_ids':[ids[j] for j in ranked(d0)],
                     'bm25_ids':[ids[j] for j in bo],'dense':d[union].tolist(),'sparse':b[union].tolist(),
                     'scope':report,'scope_candidate_count':len(docs)})
    write_jsonl(rows,out/'finqa.jsonl')
    write_json({'status':'completed','protocol_sha256':sha(out/'protocol.json'),
                'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='protocol.json'}},out/'manifest.lock.json')
    print(json.dumps({'prepared_candidates':split,'finqa_questions':len(cases)}),flush=True)


def score(split,modelname):
    import torch
    plan=check_contract()
    if split=='test':
        selected=load(RUN/'selection.lock.json')
        assert modelname in {'previous',selected['config']['model']}
        if modelname!='previous':assert sha(modelspec(modelname)/'model.safetensors')==selected['weights_sha256']
    cache=RUN/f'{split}_candidates'
    if not cache.exists():prepare_candidates(split)
    manifest=load(cache/'manifest.lock.json')
    for n,h in manifest['files_sha256'].items():assert sha(cache/n)==h,n
    out=RUN/f'{split}_scores'/modelname;out.mkdir(parents=True,exist_ok=False)
    model=DomainCrossEncoder(modelspec(modelname),'cuda',64,512)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'split':split,'model':modelname,
              'selection_protocol_sha256':sha(RUN/'selection_protocol.json'),
              'selection_lock_sha256':sha(RUN/'selection.lock.json') if split=='test' else None,
              'candidate_manifest_sha256':sha(cache/'manifest.lock.json'),'model_fingerprint':model.fingerprint(),
              'fixed_query_model_sha256':plan['fixed_query_weights_sha256'],'training_inputs_used':False}
    write_json(protocol,out/'protocol.json');start=time.perf_counter();reused=0
    for co,qfile,afile,_ in specs(split):
        cases=read_jsonl(ROOT/qfile);answers={a['id']:a['text'] for a in read_jsonl(ROOT/afile)}
        rows=read_jsonl(cache/f'{co}.jsonl');assert [c['id'] for c in cases]==[r['id'] for r in rows]
        with (out/f'{co}.jsonl').open('w',encoding='utf8') as f:
            for off in range(0,len(rows),16):
                batch=rows[off:off+16];qs=cases[off:off+16]
                if modelname=='previous' and co!='finqa':values=np.array([s for r in batch for s in r['previous_cross']]);reused+=len(values)
                else:values=model.score_pairs([(c['question'],answers[a]) for c,r in zip(qs,batch) for a in r['candidate_ids']])
                cursor=0
                for r in batch:
                    n=len(r['candidate_ids']);f.write(json.dumps({'id':r['id'],'candidate_ids':r['candidate_ids'],'cross':values[cursor:cursor+n].tolist()})+'\n');cursor+=n
                if (off+len(batch))%400==0 or off+len(batch)==len(rows):
                    f.flush();print(json.dumps({'scoring':modelname,'split':split,'cohort':co,'questions':off+len(batch),'total':len(rows),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    del model;torch.cuda.empty_cache()
    write_json({'status':'completed','seconds':time.perf_counter()-start,'protocol_sha256':sha(out/'protocol.json'),
                'new_pairs':sum(len(r['candidate_ids']) for co,*_ in specs(split) for r in read_jsonl(out/f'{co}.jsonl'))-reused,
                'reused_pairs':reused,'score_files_sha256':{p.name:sha(p) for p in out.glob('*.jsonl')}},out/'completion.json')


def configurations(plan):
    models=['previous']+[f'seed_{s}_epoch_{e}' for s in plan['seeds'] for e in plan['epochs']]
    return [{'name':f'{m}_cw{round(w*100)}','model':m,'cross_weight':w} for m in models for w in plan['cross_weights']]


def paired(new,old,cases,cohort):
    buckets=clusters(cases,shared_labels=cohort in {'insuranceqa','fiqa'})
    sizes=np.array([len(b) for b in buckets]);draws=np.random.default_rng(20261003).integers(0,len(buckets),(5000,len(buckets)))
    result={'n':len(cases),'clusters':len(buckets),'cluster_unit':'shared gold' if cohort in {'insuranceqa','fiqa'} else 'annual report' if cohort=='finqa' else 'source URL/title',
            'bootstrap_repetitions':5000,'multiple_comparisons_adjusted':False,'metrics':{}}
    for metric in ['hit_at_1','hit_at_10','mrr_at_100','ndcg_at_10','all_evidence_at_5','all_evidence_at_10','evidence_recall_at_5']:
        delta=np.array([a[metric]-b[metric] for a,b in zip(new,old)]);sums=np.array([delta[b].sum() for b in buckets]);boot=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
        result['metrics'][metric]={'difference':float(delta.mean()),'cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
                                   'wins':int((delta>0).sum()),'losses':int((delta<0).sum())}
    return result


def evaluate(split):
    plan=check_contract();allconfigs=configurations(plan);selection=None
    if split=='valid':configs=allconfigs
    else:
        selection=load(RUN/'selection.lock.json')
        configs=[{'name':'previous_default','model':'previous','cross_weight':.5},
                 {**selection['best_previous_config'],'name':'previous_validation_best'},
                 {'name':'previous_same_weight','model':'previous','cross_weight':selection['config']['cross_weight']},
                 {**selection['config'],'name':'selected'}]
    out=RUN/f'{split}_evaluation';out.mkdir(exist_ok=False)
    predictions=defaultdict(lambda:defaultdict(list));summaries={};comparisons={}
    for co,qfile,_,_ in specs(split):
        cases=read_jsonl(ROOT/qfile);rows=read_jsonl(RUN/f'{split}_candidates'/f'{co}.jsonl');model_scores={}
        for m in dict.fromkeys(c['model'] for c in configs):
            folder=RUN/f'{split}_scores'/m;completion=load(folder/'completion.json')
            assert sha(folder/f'{co}.jsonl')==completion['score_files_sha256'][f'{co}.jsonl']
            scored=read_jsonl(folder/f'{co}.jsonl');assert [r['id'] for r in scored]==[r['id'] for r in rows]
            assert all(a['candidate_ids']==b['candidate_ids'] for a,b in zip(scored,rows))
            model_scores[m]=scored
        for i,(case,row) in enumerate(zip(cases,rows)):
            for config in configs:
                raw={**row,'cross':model_scores[config['model']][i]['cross']}
                order=rank_row(raw,{'pool':'union200','lexical_weight':.2,'cross_weight':config['cross_weight']})
                predictions[co][config['name']].append({'id':case['id'],'top_answer_ids':order,**evidence_metrics(order,case['gold_answer_ids'],row['candidate_ids'])})
            for name,key in [('bge','public_bge_ids'),('adapted_dense','bge_ids'),('bm25','bm25_ids')]:
                order=row[key];predictions[co][name].append({'id':case['id'],'top_answer_ids':order,**evidence_metrics(order,case['gold_answer_ids'],order)})
            scores=defaultdict(float)
            for order in [row['bge_ids'],row['bm25_ids']]:
                for rank,a in enumerate(order,1):scores[a]+=1/(60+rank)
            positions={a:j for j,a in enumerate(row['candidate_ids'])}
            order=sorted(scores,key=lambda a:(-scores[a],positions[a]))[:100]
            predictions[co]['adapted_rrf'].append({'id':case['id'],'top_answer_ids':order,**evidence_metrics(order,case['gold_answer_ids'],row['candidate_ids'])})
        summaries[co]={};comparisons[co]={}
        for part,ix in partitions(co,cases).items():
            summaries[co][part]={name:summarize_evidence([r[i] for i in ix]) for name,r in predictions[co].items()}
            if split=='test':
                comparisons[co][part]={name:paired([predictions[co]['selected'][i] for i in ix],[predictions[co][name][i] for i in ix],[cases[i] for i in ix],co)
                                      for name in ['previous_default','previous_validation_best','previous_same_weight','bge']}
    write_jsonl([{'cohort':co,'arm':name,**r} for co,arms in predictions.items() for name,rows in arms.items() for r in rows],out/'predictions.jsonl')
    report={'status':'completed','split':split,'created_utc':datetime.now(timezone.utc).isoformat(),'summaries':summaries,
            'paired_selected_minus_controls':comparisons,'predictions_sha256':sha(out/'predictions.jsonl'),
            'selection_protocol_sha256':sha(RUN/'selection_protocol.json'),
            'selection_lock_sha256':sha(RUN/'selection.lock.json') if selection else None}
    write_json(report,out/'summary.json')
    if split=='valid':select(report,plan,allconfigs)
    print(json.dumps({'evaluated':split,'selected':selection['config'] if selection else None}),flush=True)


def select(report,plan,configs):
    source={'insuranceqa':('insuranceqa','all'),'government':('multidomain','government'),'general':('multidomain','general'),'finance':('fiqa','all'),'financial_report':('finqa','all')}
    rows=[]
    for c in configs:
        metrics={d:report['summaries'][co][part][c['name']] for d,(co,part) in source.items()}
        value=sum(w*metrics[d]['all_evidence_at_5' if d=='financial_report' else 'mrr_at_100'] for d,w in plan['selection']['domain_weights'].items())
        rows.append({'config':c,'domains':metrics,'utility':value})
    default=next(r for r in rows if r['config']['name']=='previous_cw50')
    for r in rows:
        r['eligible']=all(r['domains'][g['domain']][g['metric']]+1e-12>=default['domains'][g['domain']][g['metric']]-g['max_drop'] for g in plan['selection']['guards'])
    old=max([r for r in rows if r['config']['model']=='previous' and r['eligible']],key=lambda r:(r['utility'],r['domains']['insuranceqa']['hit_at_1']))
    eligible=[r for r in rows if r['config']['model']!='previous' and r['eligible'] and r['utility']>old['utility']+1e-12]
    chosen=max(eligible,key=lambda r:(r['utility'],r['domains']['insuranceqa']['hit_at_1'])) if eligible else old
    path=RUN/'selection.lock.json';assert not path.exists()
    write_json(rows,RUN/'validation_sweep.json')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'selected_on':'valid','new_training_promoted':bool(eligible),
                'config':chosen['config'],'best_previous_config':old['config'],'chosen_validation':chosen,'best_previous_validation':old,
                'weights_sha256':sha(modelspec(chosen['config']['model'])/'model.safetensors'),
                'selection_protocol_sha256':sha(RUN/'selection_protocol.json'),'validation_summary_sha256':sha(RUN/'valid_evaluation/summary.json'),
                'validation_sweep_sha256':sha(RUN/'validation_sweep.json'),'new_test_scored_before_selection':False},path)
    print(json.dumps({'selected':chosen['config'],'training_promoted':bool(eligible),'utility':chosen['utility'],'best_previous_utility':old['utility']}),flush=True)


if __name__=='__main__':
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['score','evaluate']);p.add_argument('--split',choices=['valid','test'],required=True);p.add_argument('--model')
    a=p.parse_args();score(a.split,a.model) if a.phase=='score' else evaluate(a.split)
