"""Validation-only retention selection and paired source-cluster test evaluation."""
import argparse
from datetime import datetime, timezone
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_insuranceqa_scale import aggregate,metrics
from scripts.eval_insuranceqa_reranker import rank_row,paired_summary
from scripts.eval_domain_training import verified_scores
from src.insurerag_vlm.retention import anchored_cross_scores

RUN=ROOT/'reports/retention_v2'
CFG={'pool':'hybrid100','lexical_weight':.2,'cross_weight':.5}
FILES=[Path(__file__),ROOT/'src/insurerag_vlm/retention.py',ROOT/'scripts/eval_insuranceqa_reranker.py',ROOT/'scripts/eval_insuranceqa_scale.py']
def hashes():return {p.relative_to(ROOT).as_posix():sha(p) for p in FILES}


def predict(rows,public,cases,beta,arm):
    if [r['id'] for r in rows]!=[c['id'] for c in cases] or [r['id'] for r in public]!=[c['id'] for c in cases]:
        raise ValueError('Question identity mismatch')
    result=[]
    for r,p,c in zip(rows,public,cases):
        for k in ['candidate_ids','dense','sparse','bge_ids','bm25_ids']:
            if r[k]!=p[k]:raise ValueError('Different candidate sets/scores')
        if arm=='bge':order=r['bge_ids']
        else:
            # beta=1 preserves the original raw-score ranking including exact ties.
            cross=r['cross'] if beta==1 else anchored_cross_scores(r['cross'],p['cross'],beta)
            order=rank_row({**r,'cross':cross},CFG)
        result.append({'id':c['id'],'arm':arm,'top_answer_ids':order,**metrics(order,set(c['gold_answer_ids']))})
    return result


def domains(pred,cases,legacy=False):
    if legacy:return {'insuranceqa':aggregate(pred)}
    return {d:aggregate([p for p,c in zip(pred,cases) if c['domain']==d]) for d in ['government','general']}


def source_paired(new,old,cases):
    """Percentile bootstrap resamples whole sources, preserving question weights."""
    groups={}
    for i,c in enumerate(cases):groups.setdefault(c['source_group'],[]).append(i)
    buckets=list(groups.values());delta=np.asarray([a['hit_at_10']-b['hit_at_10'] for a,b in zip(new,old)])
    sums=np.asarray([delta[ix].sum() for ix in buckets]);sizes=np.asarray([len(ix) for ix in buckets])
    draws=np.random.default_rng(20261001).integers(0,len(buckets),(5000,len(buckets)))
    boot=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return {'n':len(cases),'source_groups':len(buckets),'hit_at_10_difference':float(delta.mean()),
            'source_cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'bootstrap_repetitions':5000}


def run(args):
    planpath=RUN/'selection_protocol.json';lockpath=RUN/'selection.lock.json'
    if args.phase=='plan':
        if planpath.exists():raise ValueError('Plan already frozen')
        plan={'created_utc':datetime.now(timezone.utc).isoformat(),'selected_on':'valid','epochs':[1,2],
            'domain_weights':[.5,.75,1.],'fusion':CFG,'training_recipe':'rehearsal_from_v2, two epochs, learning rate 5e-6',
            'guards':{'insuranceqa_min_previous_minus':.005,'government_min_public_minus':.01,'general_min_public_minus':.01},
            'objective':'Equal-domain macro Hit@10 over insuranceqa/government/general; tie macro MRR, then Hit@1; stable order',
            'promotion':'Eligible new checkpoint must strictly exceed unanchored previous-v2 macro Hit@10. Otherwise retain previous v2; do not tune on test.',
            'controls':'Public teacher, previous v2, previous v2 with same three anchor weights, BGE only. All controls reported.',
            'new_validation_counts':{'insuranceqa_historical':2000,'government':127,'general':400},
            'test_plan':{'historical_insuranceqa':2000,'fresh_government':81,'fresh_general':600,
                         'historical_government_regression':416,'fresh_test_candidate_corpus':47969},
            'test_use':'Only frozen selected model plus public and previous controls; no post-test selection or example removal.',
            'training_manifest_sha256':sha(ROOT/'data/training/retention_v2/manifest.lock.json'),
            'fresh_fixture_sha256':sha(ROOT/'data/benchmarks/multidomain_v1/manifest.lock.json'),'code_sha256':hashes()}
        write_json(plan,planpath);print(json.dumps(plan,indent=2));return
    plan=json.loads(planpath.read_text(encoding='utf8'))
    if hashes()!=plan['code_sha256']:raise ValueError('Frozen evaluation implementation changed')
    if sha(ROOT/'data/benchmarks/multidomain_v1/manifest.lock.json')!=plan['fresh_fixture_sha256']:raise ValueError('Fixture changed')
    split='valid' if args.phase=='select' else 'test'
    legacy=read_jsonl(ROOT/f'data/benchmarks/insuranceqa_v2/{split}.jsonl')
    fresh=read_jsonl(ROOT/f'data/benchmarks/multidomain_v1/{split}.jsonl')
    lp,_=verified_scores(ROOT/('reports/insuranceqa_v2/rerank_valid_v2' if split=='valid' else 'reports/insuranceqa_v2/rerank_test_v1'),split)
    lv,_=verified_scores(ROOT/('reports/domain_training_v2/valid_epoch_1' if split=='valid' else 'reports/domain_training_v2/test_selected'),split)
    fp,_=verified_scores(RUN/f'{split}_new_public',split);fv,_=verified_scores(RUN/f'{split}_new_previous',split)
    if args.phase=='select':
        arms=[('public',lp,fp,1.),('previous',lv,fv,1.)]
        arms += [(f'previous_anchor_{beta}',lv,fv,beta) for beta in plan['domain_weights'] if beta!=1]
        sources={}
        for epoch in plan['epochs']:
            lpath=RUN/f'valid_legacy_epoch_{epoch}';fpath=RUN/f'valid_new_epoch_{epoch}'
            le,_=verified_scores(lpath,'valid');fe,_=verified_scores(fpath,'valid')
            sources[str(epoch)]={str(p.relative_to(ROOT)):sha(p/'scores.jsonl') for p in [lpath,fpath]}
            arms += [(f'epoch_{epoch}',le,fe,beta) for beta in plan['domain_weights']]
        sweep=[]
        for name,lrows,frows,beta in arms:
            dm={**domains(predict(lrows,lp,legacy,beta,name),legacy,True),**domains(predict(frows,fp,fresh,beta,name),fresh)}
            sweep.append({'model':name,'domain_weight':beta,'domains':dm,
                **{'macro_'+k:sum(v[k] for v in dm.values())/3 for k in ['hit_at_10','mrr_at_100','hit_at_1']}})
        public,previous=sweep[:2]
        for row in sweep:
            d=row['domains'];row['eligible']=(d['insuranceqa']['hit_at_10']>=previous['domains']['insuranceqa']['hit_at_10']-.005
                and all(d[n]['hit_at_10']>=public['domains'][n]['hit_at_10']-.01 for n in ['government','general']))
        eligible=[r for r in sweep if r['model'].startswith('epoch_') and r['eligible'] and r['macro_hit_at_10']>previous['macro_hit_at_10']]
        best=max(eligible,key=lambda r:(r['macro_hit_at_10'],r['macro_mrr_at_100'],r['macro_hit_at_1'])) if eligible else previous
        write_json(sweep,RUN/'validation_sweep.json')
        epoch=int(best['model'].split('_')[1]) if best['model'].startswith('epoch_') else None
        model=ROOT/f'../models/insurerag-retention-v2/epoch-{epoch}' if epoch else ROOT/'../models/insurerag-domain-reranker-v2/epoch-1'
        lock={'created_utc':datetime.now(timezone.utc).isoformat(),'selected_on':'valid','promoted':bool(epoch),'epoch':epoch,
            'model_path':str(model.resolve()),'domain_weight':best['domain_weight'],'config':CFG,'validation_metrics':best,
            'selected_weights_sha256':sha(model/'model.safetensors'),
            'public_weights_sha256':sha(ROOT/'../models/ms-marco-MiniLM-L6-v2/model.safetensors'),
            'previous_weights_sha256':sha(ROOT/'../models/insurerag-domain-reranker-v2/epoch-1/model.safetensors'),
            'selection_protocol_sha256':sha(planpath),'validation_sweep_sha256':sha(RUN/'validation_sweep.json'),
            'validation_artifacts':sources,'code_sha256':hashes(),'fresh_test_status':'Not scored at selection'}
        if lockpath.exists():raise ValueError('Lock already exists')
        write_json(lock,lockpath);print(json.dumps(lock,indent=2));return
    lock=json.loads(lockpath.read_text(encoding='utf8'));beta=lock['domain_weight']
    ln,lnp=verified_scores(RUN/'test_legacy_selected','test');fn,fnp=verified_scores(RUN/'test_new_selected','test')
    if any(p['selection_lock_sha256']!=sha(lockpath) for p in [lnp,fnp]):raise ValueError('Wrong test selection')
    preds={};summaries={};paired={}
    for cohort,cases,new,previous,public in [('historical_insuranceqa',legacy,ln,lv,lp),('fresh_multidomain',fresh,fn,fv,fp)]:
        arms={'bge':predict(public,public,cases,1.,'bge'),'public':predict(public,public,cases,1.,'public'),
              'previous':predict(previous,public,cases,1.,'previous'),'selected':predict(new,public,cases,beta,'selected'),
              'selected_unanchored':predict(new,public,cases,1.,'selected_unanchored'),
              'previous_same_anchor':predict(previous,public,cases,beta,'previous_same_anchor')}
        preds[cohort]=arms
        partitions={'all':list(range(len(cases)))} if cohort.startswith('historical') else {d:[i for i,c in enumerate(cases) if c['domain']==d] for d in ['government','general','all']}
        if not cohort.startswith('historical'):partitions['all']=list(range(len(cases)))
        summaries[cohort]={};paired[cohort]={}
        for part,ix in partitions.items():
            subset=[cases[i] for i in ix];ap={n:[p[i] for i in ix] for n,p in arms.items()}
            summaries[cohort][part]={n:aggregate(p) for n,p in ap.items()}
            comparison=paired_summary if cohort.startswith('historical') else source_paired
            paired[cohort][part]={n:comparison(ap['selected'],p,subset) for n,p in ap.items() if n!='selected'}
    out=RUN/'test_evaluation';out.mkdir(exist_ok=False)
    with (out/'predictions.jsonl').open('w',encoding='utf8') as handle:
        for cohort,arms in preds.items():
            for rows in arms.values():
                for row in rows:handle.write(json.dumps({'cohort':cohort,**row})+'\n')
    report={'summaries':summaries,'selected_minus_comparators':paired,'selection_lock_sha256':sha(lockpath),
            'predictions_sha256':sha(out/'predictions.jsonl'),'code_sha256':hashes(),
            'limitations':['InsuranceQA test is historical and repeatedly inspected.','Fresh test is source-held-out for this fine-tuning only; public pretraining exposure unknown.',
                          'Government has 81 cases across 14 sources; uncertainty must be reported.','Teacher blending costs two reranker passes if beta < 1.']}
    write_json(report,out/'summary.json');print(json.dumps({'summaries':summaries,'paired':paired},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['plan','select','test']);run(p.parse_args())
