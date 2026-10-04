"""Preregister two seeds, matched fusion controls, and one held-out government test."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_insuranceqa_scale import aggregate,metrics
from scripts.eval_insuranceqa_reranker import rank_row,paired_summary
from scripts.eval_retention_v2 import source_paired

RUN=ROOT/'reports/condition_v1';PREVIOUS=ROOT/'reports/retention_v2'
BASE={'pool':'hybrid100','lexical_weight':.2,'cross_weight':.5}
CODE=[Path(__file__),ROOT/'scripts/eval_insuranceqa_reranker.py',ROOT/'scripts/eval_insuranceqa_scale.py',ROOT/'src/insurerag_vlm/reranker.py',ROOT/'scripts/eval_retention_v2.py']
def hashes():return {p.relative_to(ROOT).as_posix():sha(p) for p in CODE}
def scored(folder):
    done=json.loads((folder/'completion.json').read_text(encoding='utf8'));assert done['status']=='completed' and done['scores_sha256']==sha(folder/'scores.jsonl')
    return read_jsonl(folder/'scores.jsonl')
def predict(rows,cases,cfg,arm):
    assert [r['id'] for r in rows]==[c['id'] for c in cases]
    out=[]
    for r,c in zip(rows,cases):
        order=r['bge_ids'] if arm=='bge' else rank_row(r,cfg)
        out.append({'id':c['id'],'arm':arm,'top_answer_ids':order,**metrics(order,set(c['gold_answer_ids']))})
    return out
def objective(domains,key):return sum(w*domains[d][key] for d,w in [('insuranceqa',.6),('government',.2),('general',.2)])
def run(phase):
    planpath=RUN/'selection_protocol.json';lockpath=RUN/'selection.lock.json'
    if phase=='plan':
        assert not planpath.exists()
        fixture=json.loads((ROOT/'data/benchmarks/condition_v1/manifest.lock.json').read_text(encoding='utf8'))
        plan={'created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[42,123],'epochs_per_seed':1,'selected_on':'valid',
            'initial_weights_sha256':sha(ROOT/'../models/insurerag-retention-v2/epoch-2/model.safetensors'),
            'configurations':[{'pool':p,'lexical_weight':.2,'cross_weight':w} for p in ['hybrid100','union200'] for w in [.35,.5,.65]],
            'objective':'0.6 insuranceqa + 0.2 government + 0.2 general Hit@10; tie same weighted MRR, then Hit@1; stable order',
            'guards_vs_previous_default':{'insuranceqa_max_drop':.002,'government_max_drop':.008,'general_max_drop':.01},
            'promotion':'Compare best eligible trained checkpoint/config with best eligible previous-model config. Promote only strict weighted Hit@10 improvement; otherwise keep best eligible previous config.',
            'training_recipe':'1 epoch, two independent seeds from same previous checkpoint; LR 3e-6; teacher-aware pairwise margin; FAQ 2x, government 8x, general 1x',
            'test_plan':{'historical_insuranceqa':2000,'historical_multidomain':681,'historical_government':416,'fresh_government':fixture['fresh_test_questions'],'fresh_government_sources':fixture['fresh_test_sources'],'fresh_answer_count':fixture['answer_candidates']},
            'test_use':'Only selected model, previous and public controls; both seeds reported on validation only; no post-test changes or relabeling.',
            'attribution':'Report previous and public at default and selected configuration to distinguish retraining from candidate/fusion changes.',
            'fixture_sha256':sha(ROOT/'data/benchmarks/condition_v1/manifest.lock.json'),'preflight_amendment':'Question-mark anywhere in extracted answer now rejected after source-format inspection; before any new student scoring/training/validation/test. Previous fixture and plan retained under preflight names.',
            'training_code_sha256':sha(ROOT/'scripts/train_condition_reranker.py'),
            'scoring_code_sha256':sha(ROOT/'scripts/score_condition_model.py'),'mining_code_sha256':sha(ROOT/'scripts/prepare_condition_training.py'),
            'historical_score_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [PREVIOUS/'historical_government/scores.jsonl']},'code_sha256':hashes()}
        write_json(plan,planpath);print(json.dumps(plan,indent=2));return
    plan=json.loads(planpath.read_text(encoding='utf8'));assert hashes()==plan['code_sha256']
    assert sha(ROOT/'data/benchmarks/condition_v1/manifest.lock.json')==plan['fixture_sha256']
    if phase=='select':
        assert not lockpath.exists()
        legacy=read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/valid.jsonl');mixed=read_jsonl(ROOT/'data/benchmarks/multidomain_v1/valid.jsonl')
        arms={'previous':(scored(PREVIOUS/'valid_legacy_epoch_2'),scored(PREVIOUS/'valid_new_epoch_2'))}
        artifacts={}
        for seed in plan['seeds']:
            paths=[RUN/f'valid_legacy_seed_{seed}',RUN/f'valid_mixed_seed_{seed}'];arms[f'seed_{seed}']=tuple(scored(p) for p in paths)
            artifacts[str(seed)]={p.relative_to(ROOT).as_posix():sha(p/'scores.jsonl') for p in paths}
        sweep=[]
        for name,(lr,mr) in arms.items():
            for cfg in plan['configurations']:
                lp=predict(lr,legacy,cfg,name);mp=predict(mr,mixed,cfg,name)
                domains={'insuranceqa':aggregate(lp),**{d:aggregate([p for p,c in zip(mp,mixed) if c['domain']==d]) for d in ['government','general']}}
                sweep.append({'model':name,'config':cfg,'domains':domains,**{'weighted_'+k:objective(domains,k) for k in ['hit_at_10','mrr_at_100','hit_at_1']}})
        baseline=next(r for r in sweep if r['model']=='previous' and r['config']==BASE)
        for r in sweep:r['eligible']=all(r['domains'][d]['hit_at_10']>=baseline['domains'][d]['hit_at_10']-drop for d,drop in [('insuranceqa',.002),('government',.008),('general',.01)])
        key=lambda r:(r['weighted_hit_at_10'],r['weighted_mrr_at_100'],r['weighted_hit_at_1'])
        previous=max([r for r in sweep if r['model']=='previous' and r['eligible']],key=key)
        trained=[r for r in sweep if r['model']!='previous' and r['eligible'] and r['weighted_hit_at_10']>previous['weighted_hit_at_10']]
        best=max(trained,key=key) if trained else previous;promoted=best['model']!='previous'
        model=ROOT/f'../models/insurerag-condition-v1-seed-{best["model"].split("_")[-1]}/epoch-1' if promoted else ROOT/'../models/insurerag-retention-v2/epoch-2'
        write_json(sweep,RUN/'validation_sweep.json')
        lock={'created_utc':datetime.now(timezone.utc).isoformat(),'selected_on':'valid','promoted_weights':promoted,'selected_model':best['model'],
            'model_path':str(model.resolve()),'config':best['config'],'validation_metrics':best,'best_previous_configuration':previous,
            'selected_weights_sha256':sha(model/'model.safetensors'),'previous_weights_sha256':plan['initial_weights_sha256'],
            'public_weights_sha256':sha(ROOT/'../models/ms-marco-MiniLM-L6-v2/model.safetensors'),
            'selection_protocol_sha256':sha(planpath),'validation_sweep_sha256':sha(RUN/'validation_sweep.json'),'validation_artifacts':artifacts,
            'training_manifest_sha256':sha(ROOT/'data/training/condition_v1/manifest.lock.json'),'fresh_test_status':'Not scored at selection','code_sha256':hashes()}
        write_json(lock,lockpath);print(json.dumps(lock,indent=2));return
    lock=json.loads(lockpath.read_text(encoding='utf8'));cfg=lock['config'];assert lock['selection_protocol_sha256']==sha(planpath)
    iq=read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/test.jsonl');mix=read_jsonl(ROOT/'data/benchmarks/multidomain_v1/test.jsonl')
    gov=read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/questions.jsonl');fresh=read_jsonl(ROOT/'data/benchmarks/condition_v1/test.jsonl')
    oldgov=read_jsonl(PREVIOUS/'historical_government/scores.jsonl');assert sha(PREVIOUS/'historical_government/scores.jsonl')==plan['historical_score_hashes']['reports/retention_v2/historical_government/scores.jsonl']
    govpublic=[{**r,'cross':r['cross_base']} for r in oldgov]
    cohorts=[('historical_insuranceqa',iq,scored(ROOT/'reports/insuranceqa_v2/rerank_test_v1'),scored(PREVIOUS/'test_legacy_selected'),scored(RUN/'test_legacy_selected')),
        ('historical_multidomain',mix,scored(PREVIOUS/'test_new_public'),scored(PREVIOUS/'test_new_selected'),scored(RUN/'test_mixed_selected')),
        ('historical_government',gov,govpublic,oldgov,scored(RUN/'test_oldgov_selected')),
        ('fresh_government',fresh,scored(RUN/'test_fresh_public'),scored(RUN/'test_fresh_previous'),scored(RUN/'test_fresh_selected'))]
    summaries={};paired={};predictions=[]
    for name,cases,public,previous,selected in cohorts:
        for new,old in zip(selected,previous):
            assert all(new[k]==old[k] for k in ['id','candidate_ids','dense','sparse','bge_ids','bm25_ids'])
        arms={'bge':predict(public,cases,BASE,'bge'),'public_default':predict(public,cases,BASE,'public_default'),
            'public_matched':predict(public,cases,cfg,'public_matched'),'previous_default':predict(previous,cases,BASE,'previous_default'),
            'previous_matched':predict(previous,cases,cfg,'previous_matched'),'selected':predict(selected,cases,cfg,'selected')}
        partitions={'all':list(range(len(cases)))} if name!='historical_multidomain' else {d:[i for i,c in enumerate(cases) if c['domain']==d] for d in ['government','general']}
        summaries[name]={};paired[name]={}
        for part,ix in partitions.items():
            subset=[cases[i] for i in ix];ap={arm:[ps[i] for i in ix] for arm,ps in arms.items()}
            summaries[name][part]={arm:aggregate(ps) for arm,ps in ap.items()}
            compare=paired_summary if name=='historical_insuranceqa' else source_paired
            paired[name][part]={arm:compare(ap['selected'],ps,subset) for arm,ps in ap.items() if arm!='selected'}
        predictions.extend({'cohort':name,**p} for ps in arms.values() for p in ps)
    out=RUN/'test_evaluation';out.mkdir(exist_ok=False)
    with (out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for r in predictions:f.write(json.dumps(r)+'\n')
    summary={'summaries':summaries,'paired_selected_minus_controls':paired,'selection_lock_sha256':sha(lockpath),
        'predictions_sha256':sha(out/'predictions.jsonl'),'code_sha256':hashes(),
        'limitations':['Historical tests were previously inspected.','Fresh government labels are publisher QA extracted mechanically, not expert-adjudicated.',
            'Fresh government has 29 source groups; cluster CI reported.','Retrieval Hit@10 is not generated-answer accuracy or legal correctness.','No new independent general-language test.']}
    write_json(summary,out/'summary.json');print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['plan','select','test']);run(p.parse_args().phase)
