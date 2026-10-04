"""Verify chronology, held-out isolation, score hashes and both actual training runs."""
import hashlib,json,math,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_reranker_training import normalize

def run():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);run=ROOT/'reports/condition_listwise_v1';data=ROOT/'data/training/condition_listwise_v1';fixture=ROOT/'data/benchmarks/condition_v1'
    plan=json.loads((run/'selection_protocol.json').read_text(encoding='utf8'));lock=json.loads((run/'selection.lock.json').read_text(encoding='utf8'))
    for directory in [data,fixture]:
        manifest=json.loads((directory/'manifest.lock.json').read_text(encoding='utf8'))
        for n,h in manifest['files'].items():assert sha(directory/n)==h,n
    fixturelock=json.loads((fixture/'manifest.lock.json').read_text(encoding='utf8'))
    dol=ROOT/'data/research_corpus/dol_additional_v1';source=json.loads((dol/'manifest.json').read_text(encoding='utf8'))
    assert sha(dol/'manifest.json')==fixturelock['dol_source_manifest_sha256'] and sha(dol/'records.jsonl')==source['records_sha256']
    assert sha(ROOT/'scripts/fetch_additional_dol.py')==source['code_sha256']
    for record in read_jsonl(dol/'records.jsonl'):
        assert hashlib.sha256(record['text'].encode('utf8')).hexdigest()==record['text_sha256']
        assert sha(ROOT/'../condition_source_cache'/(record['source_url'].rsplit('/',1)[-1]+'.html'))==record['html_sha256']
    hicric=ROOT/'data/research_corpus/hicric_public_v1';hm=json.loads((hicric/'manifest.json').read_text(encoding='utf8'))
    assert sha(hicric/'records.jsonl')==hm['files']['records.jsonl']
    assert lock['selection_protocol_sha256']==sha(run/'selection_protocol.json')
    assert lock['validation_sweep_sha256']==sha(run/'validation_sweep.json')
    assert lock['training_manifest_sha256']==sha(data/'manifest.lock.json')
    assert plan['fixture_sha256']==sha(fixture/'manifest.lock.json')
    for n,h in plan['code_sha256'].items():assert sha(ROOT/n)==h,n
    for key,name in [('training','train_condition_listwise.py'),('scoring','score_condition_listwise.py'),('mining','prepare_condition_listwise_training.py')]:assert plan[key+'_code_sha256']==sha(ROOT/'scripts'/name)
    groups=read_jsonl(data/'train_groups.jsonl');answers=read_jsonl(data/'answers.jsonl');lookup={a['id']:a for a in answers}
    parentgroups=read_jsonl(ROOT/'data/training/condition_v1/train_groups.jsonl')
    assert len(parentgroups)==len(groups)
    for a,b in zip(groups,parentgroups):
        assert all(a[k]==b[k] for k in ['id','question','positive_ids','negative_ids','negative_sources','teacher_scores','source_domain'])
    anchorlock=json.loads((data/'manifest.lock.json').read_text(encoding='utf8'))
    assert anchorlock['parent_condition_training_sha256']==sha(ROOT/'data/training/condition_v1/manifest.lock.json')
    assert anchorlock['anchor_model']['files_sha256']['model.safetensors']==plan['initial_weights_sha256']
    held=read_jsonl(fixture/'test.jsonl')+[c for split in ['valid','test'] for c in read_jsonl(ROOT/f'data/benchmarks/multidomain_v1/{split}.jsonl')]
    sources={c['source_group'] for c in held};forbidden_text={normalize(a['text']) for a in answers if a.get('source_group') in sources}
    forbidden_text.update(normalize(a['text']) for a in read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl'))
    forbidden={a['id'] for a in answers if normalize(a['text']) in forbidden_text}
    allpairs={a for g in groups for a in g['positive_ids']+g['negative_ids']};assert not allpairs&forbidden
    assert len({g['id'] for g in groups})==len(groups)==25987
    for g in groups:
        assert set(g['positive_ids']).isdisjoint(g['negative_ids']) and len(set(g['negative_ids']))==11
        assert set(g['teacher_scores'])==set(g['positive_ids']+g['negative_ids'])
        assert set(g['anchor_scores'])==set(g['positive_ids']+g['negative_ids'])
        assert Counter(g['negative_sources'].values())=={'student_hard':6,'lexical_hard':4,'random':1}
    assert (data/'answers.jsonl').read_bytes()==(fixture/'answers.jsonl').read_bytes()
    presentations=sum({'insuranceqa':2,'government':8,'general':1}[g['source_domain']] for g in groups);expectedsteps=math.ceil(presentations/16)
    previous=load_file(ROOT/'../models/insurerag-retention-v2/epoch-2/model.safetensors');training=[]
    for seed in plan['seeds']:
        folder=run/f'train_seed_{seed}';proto=json.loads((folder/'protocol.json').read_text(encoding='utf8'));done=json.loads((folder/'completion.json').read_text(encoding='utf8'))
        assert done['status']=='completed' and done['optimizer_steps']==expectedsteps and done['pair_exposures']==presentations*6
        assert proto['selection_protocol_sha256']==sha(run/'selection_protocol.json') and proto['training_manifest_sha256']==sha(data/'manifest.lock.json')
        assert proto['code_sha256']==plan['training_code_sha256'];assert sha(folder/'training_log.jsonl')==done['log_sha256']
        assert datetime.fromisoformat(proto['created_utc'])>=datetime.fromisoformat(plan['created_utc'])
        model=ROOT/f'../models/insurerag-condition-listwise-v1-seed-{seed}/epoch-1';assert sha(model/'model.safetensors')==done['checkpoint']['weights_sha256']
        values=load_file(model/'model.safetensors');assert values.keys()==previous.keys();changed=0;tensors=0
        for name,t in values.items():
            assert t.shape==previous[name].shape and torch.isfinite(t).all()
            n=int(torch.count_nonzero(t!=previous[name]));changed+=n;tensors+=int(n>0)
        assert changed>0
        training.append({'seed':seed,'steps':done['optimizer_steps'],'pair_exposures':done['pair_exposures'],'seconds':done['seconds'],
            'weights_sha256':done['checkpoint']['weights_sha256'],'changed_values':changed,'changed_tensors':tensors,'peak_cuda_allocated_bytes':done['peak_cuda_allocated_bytes']})
    assert training[0]['weights_sha256']!=training[1]['weights_sha256']
    parentrun=ROOT/'reports/condition_v1';parentlock=json.loads((parentrun/'selection.lock.json').read_text(encoding='utf8'))
    assert not parentlock['promoted_weights'] and plan['parent_selection_sha256']==sha(parentrun/'selection.lock.json')
    assert not (parentrun/'test_evaluation').exists(),'Fresh test must not have been opened by rejected recipe'
    rejected=[]
    for seed in plan['seeds']:
        folder=parentrun/f'train_seed_{seed}';done=json.loads((folder/'completion.json').read_text(encoding='utf8'))
        assert done['status']=='completed' and done['optimizer_steps']==expectedsteps and done['pair_exposures']==presentations*6
        assert sha(folder/'training_log.jsonl')==done['log_sha256']
        assert sha(ROOT/f'../models/insurerag-condition-v1-seed-{seed}/epoch-1/model.safetensors')==done['checkpoint']['weights_sha256']
        for cohort in ['legacy','mixed']:
            score=parentrun/f'valid_{cohort}_seed_{seed}';completion=json.loads((score/'completion.json').read_text(encoding='utf8'))
            assert sha(score/'scores.jsonl')==completion['scores_sha256'] and sha(score/'protocol.json')==completion['protocol_sha256']
        rejected.append({'seed':seed,'steps':done['optimizer_steps'],'pair_exposures':done['pair_exposures'],'seconds':done['seconds'],'weights_sha256':done['checkpoint']['weights_sha256']})
    checks=[]
    for folder in sorted(run.glob('valid_*'))+sorted(run.glob('test_*')):
        if not folder.is_dir() or not (folder/'completion.json').exists():continue
        done=json.loads((folder/'completion.json').read_text(encoding='utf8'));proto=json.loads((folder/'protocol.json').read_text(encoding='utf8'))
        assert done['status']=='completed' and sha(folder/'scores.jsonl')==done['scores_sha256'] and sha(folder/'protocol.json')==done['protocol_sha256']
        assert proto['code_sha256']==plan['scoring_code_sha256']
        if folder.name.startswith('test_'):
            assert proto['selection_lock_sha256']==sha(run/'selection.lock.json')
            assert datetime.fromisoformat(proto['created_utc'])>=datetime.fromisoformat(lock['created_utc'])
        if 'model' in proto:
            key='public_weights_sha256' if folder.name.endswith('_public') else 'previous_weights_sha256' if folder.name.endswith('_previous') else 'selected_weights_sha256'
            if folder.name.startswith('test_'):assert proto['model']['files_sha256']['model.safetensors']==lock[key]
        checks.append({'artifact':folder.name,'questions':done['questions'],'scores_sha256':done['scores_sha256']})
    summary=json.loads((run/'test_evaluation/summary.json').read_text(encoding='utf8'))
    assert summary['selection_lock_sha256']==sha(run/'selection.lock.json') and summary['predictions_sha256']==sha(run/'test_evaluation/predictions.jsonl')
    assert sha(Path(lock['model_path'])/'model.safetensors')==lock['selected_weights_sha256']
    result={'status':'verified','created_utc':datetime.now(timezone.utc).isoformat(),'unique_training_questions':len(groups),
        'domain_counts':dict(Counter(g['source_domain'] for g in groups)),'heldout_source_positive_or_negative_overlap':0,
        'government_source_span_audit_sha256':sha(run/'fixture_verification.json'),'fixture_manifest_sha256':sha(fixture/'manifest.lock.json'),
        'government_raw_source_hashes_verified':True,
        'training_manifest_sha256':sha(data/'manifest.lock.json'),'selected_model':lock['selected_model'],'promoted_weights':lock['promoted_weights'],
        'training_runs':training,'total_optimizer_steps':sum(t['steps'] for t in training),'total_pair_exposures':sum(t['pair_exposures'] for t in training),
        'rejected_parent_training_runs':rejected,'parent_selection_sha256':sha(parentrun/'selection.lock.json'),
        'all_recipes_optimizer_steps':sum(t['steps'] for t in training+rejected),'all_recipes_pair_exposures':sum(t['pair_exposures'] for t in training+rejected),
        'score_artifacts':checks,'selection_lock_sha256':sha(run/'selection.lock.json'),'test_evaluation_sha256':sha(run/'test_evaluation/summary.json'),
        'verification_code_sha256':sha(Path(__file__))}
    write_json(result,run/'verification.json');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
