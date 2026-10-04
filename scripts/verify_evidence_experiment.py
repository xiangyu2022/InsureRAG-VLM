"""Verify actual parameter changes, source/score hashes and validation-only choice."""
from datetime import datetime,timezone
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from scripts.eval_evidence_reranker import check_contract,modelspec,RUN,PREVIOUS,QUERY,configurations
from scripts.verify_evidence_inputs import verify_fixed_query_inputs

def load(p):return json.loads(p.read_text(encoding='utf8'))


def main(args):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);plan=check_contract();fixed=verify_fixed_query_inputs();initial=load_file(str(PREVIOUS/'model.safetensors'))
    data=ROOT/'data/training/evidence_reranker_v1';manifest=load(data/'manifest.lock.json')
    assert sha(data/'manifest.lock.json')==plan['training_manifest_sha256']
    for n,h in manifest['files'].items():assert sha(data/n)==h,n
    groups=read_jsonl(data/'train_groups.jsonl');forbidden=set(load(data/'isolation.json')['forbidden_answer_ids'])
    assert len(groups)==len({g['id'] for g in groups})==len({' '.join(g['question'].casefold().split()) for g in groups})
    assert not any((set(g['positive_ids'])|set(g['negative_ids']))&forbidden for g in groups)
    indices=sum(plan['training']['repeat_factors'][g['source_domain']] for g in groups)
    expected=math.ceil(math.ceil(indices/plan['training']['micro_groups'])/plan['training']['accumulation'])*plan['training']['epochs']
    runs=[];inference={}
    for seed in plan['seeds']:
        folder=RUN/f'train_seed_{seed}';protocol=load(folder/'protocol.json');completion=load(folder/'completion.json')
        assert completion['status']=='completed' and completion['optimizer_steps']==expected==protocol['optimizer_steps_planned']
        assert completion['query_presentations']==indices*plan['training']['epochs']
        assert completion['pair_presentations']==completion['query_presentations']*6
        assert completion['training_log_sha256']==sha(folder/'training_log.jsonl')
        history=read_jsonl(folder/'training_log.jsonl')
        assert all(a['step']<b['step'] for a,b in zip(history,history[1:]))
        assert all(math.isfinite(r[k]) for r in history for k in ['mean_rank_loss','mean_kl'])
        for k in ['optimizer_steps','query_presentations','pair_presentations']:
            assert completion[k]==history[-1]['step' if k=='optimizer_steps' else k]
        assert protocol['training_manifest_sha256']==sha(data/'manifest.lock.json')
        assert protocol['selection_protocol_sha256']==sha(RUN/'selection_protocol.json')
        assert datetime.fromisoformat(plan['created_utc'])<datetime.fromisoformat(protocol['created_utc'])
        epochs=[]
        for checkpoint in completion['epochs']:
            name=f'seed_{seed}_epoch_{checkpoint["epoch"]}';model=modelspec(name)
            assert sha(model/'model.safetensors')==checkpoint['weights_sha256']
            assert checkpoint['training_protocol_sha256']==sha(folder/'protocol.json')
            weights=load_file(str(model/'model.safetensors'));assert set(weights)==set(initial)
            changed=sum(int((weights[n]!=initial[n]).sum()) for n in initial)
            assert changed>1000000 and all(torch.isfinite(t).all() for t in weights.values())
            epochs.append({'epoch':checkpoint['epoch'],'changed_parameter_values':changed,'weights_sha256':checkpoint['weights_sha256']})
            inference[name]={p.name:sha(p) for p in model.iterdir() if p.is_file() and p.suffix in {'.json','.txt','.safetensors'}}
        runs.append({'seed':seed,'optimizer_steps':completion['optimizer_steps'],'query_presentations':completion['query_presentations'],
                     'pair_presentations':completion['pair_presentations'],'seconds':completion['seconds'],
                     'peak_cuda_allocated_bytes':completion['peak_cuda_allocated_bytes'],'epochs':epochs})
    inference['previous']={p.name:sha(p) for p in PREVIOUS.iterdir() if p.is_file() and p.suffix in {'.json','.txt','.safetensors'}}
    scored=[]
    for split in ['valid','test']:
        root=RUN/f'{split}_scores'
        if not root.exists():
            assert split=='test' and not args.require_test
            continue
        cm=load(RUN/f'{split}_candidates/manifest.lock.json')
        required={'previous'}|({f'seed_{s}_epoch_{e}' for s in plan['seeds'] for e in plan['epochs']}
                              if split=='valid' else {load(RUN/'selection.lock.json')['config']['model']})
        assert {p.name for p in root.iterdir() if p.is_dir()}==required
        for n,h in cm['files_sha256'].items():assert sha(RUN/f'{split}_candidates'/n)==h,n
        for folder in root.iterdir():
            if not folder.is_dir():continue
            p=load(folder/'protocol.json');c=load(folder/'completion.json')
            assert c['status']=='completed' and c['protocol_sha256']==sha(folder/'protocol.json')
            assert p['model_fingerprint']['files_sha256']==inference[folder.name]
            assert p['candidate_manifest_sha256']==sha(RUN/f'{split}_candidates/manifest.lock.json')
            assert p['selection_protocol_sha256']==sha(RUN/'selection_protocol.json')
            for n,h in c['score_files_sha256'].items():assert sha(folder/n)==h,n
            if split=='test':
                selection=load(RUN/'selection.lock.json')
                assert p['selection_lock_sha256']==sha(RUN/'selection.lock.json')
                assert datetime.fromisoformat(p['created_utc'])>datetime.fromisoformat(selection['created_utc'])
            scored.append({'split':split,'model':folder.name,'new_pairs':c['new_pairs'],'reused_pairs':c['reused_pairs'],
                           'completion_sha256':sha(folder/'completion.json')})
    selection=load(RUN/'selection.lock.json');assert selection['selected_on']=='valid'
    assert selection['config'] in configurations(plan)
    assert selection['weights_sha256']==sha(modelspec(selection['config']['model'])/'model.safetensors')
    assert selection['validation_summary_sha256']==sha(RUN/'valid_evaluation/summary.json')
    assert selection['validation_sweep_sha256']==sha(RUN/'validation_sweep.json')
    sweep=load(RUN/'validation_sweep.json')
    old=max([r for r in sweep if r['config']['model']=='previous' and r['eligible']],key=lambda r:(r['utility'],r['domains']['insuranceqa']['hit_at_1']))
    eligible=[r for r in sweep if r['config']['model']!='previous' and r['eligible'] and r['utility']>old['utility']+1e-12]
    chosen=max(eligible,key=lambda r:(r['utility'],r['domains']['insuranceqa']['hit_at_1'])) if eligible else old
    assert chosen['config']==selection['config'] and old['config']==selection['best_previous_config']
    evaluations={}
    for split in ['valid','test']:
        folder=RUN/f'{split}_evaluation'
        if not folder.exists():assert split=='test' and not args.require_test;continue
        summary=load(folder/'summary.json');assert summary['predictions_sha256']==sha(folder/'predictions.jsonl')
        assert summary['status']=='completed' and summary['selection_protocol_sha256']==sha(RUN/'selection_protocol.json')
        if split=='test':assert summary['selection_lock_sha256']==sha(RUN/'selection.lock.json')
        evaluations[split]={'summary_sha256':sha(folder/'summary.json'),'predictions_sha256':sha(folder/'predictions.jsonl')}
    audit=load(RUN/'data_verification.json')
    assert audit['training_manifest_sha256']==sha(data/'manifest.lock.json') and audit['positive_negative_or_forbidden_overlap']==0
    environment=load(RUN/'environment.json')
    assert environment['code_sha256']==sha(ROOT/'scripts/record_evidence_environment.py')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'verified','training_runs':runs,
            'total_optimizer_steps':sum(r['optimizer_steps'] for r in runs),'total_query_presentations':sum(r['query_presentations'] for r in runs),
            'total_pair_presentations':sum(r['pair_presentations'] for r in runs),'selection_sha256':sha(RUN/'selection.lock.json'),
            'selection_recomputed_from_frozen_validation':True,'scoring_artifacts':scored,'inference_files_sha256':inference,
            'query_encoder_weights_unchanged':sha(QUERY/'model.safetensors')==plan['fixed_query_weights_sha256'],
            'fixed_encoder_inference_files':fixed,'fixed_input_verifier_sha256':sha(ROOT/'scripts/verify_evidence_inputs.py'),
            'training_manifest_sha256':sha(data/'manifest.lock.json'),'data_audit_sha256':sha(RUN/'data_verification.json'),
            'unique_training_question_texts':len(groups),'training_question_duplicate_normalization':'casefold and whitespace',
            'environment_sha256':sha(RUN/'environment.json'),
            'evaluations':evaluations,'code_sha256':sha(Path(__file__))}
    write_json(result,RUN/('verification.json' if args.require_test else 'pre_test_verification.json'))
    print(json.dumps({'verified':True,'total_optimizer_steps':result['total_optimizer_steps'],'scored_models':len(scored)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-test',action='store_true');main(p.parse_args())
