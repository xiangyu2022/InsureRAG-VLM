"""Verify final artifacts, model identity, chronology and real parameter changes."""
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,write_json,read_jsonl

def run():
    torch.set_num_threads(4);run=ROOT/'reports/retention_v2';lockpath=run/'selection.lock.json'
    lock=json.loads(lockpath.read_text(encoding='utf8'));plan=json.loads((run/'selection_protocol.json').read_text(encoding='utf8'))
    assert sha(run/'selection_protocol.json')==lock['selection_protocol_sha256']
    assert sha(run/'validation_sweep.json')==lock['validation_sweep_sha256']
    for name,expected in lock['code_sha256'].items():assert sha(ROOT/name)==expected,name
    audit=json.loads((run/'data_audit.json').read_text(encoding='utf8'));assert audit['status']=='passed' and audit['new_heldout_source_training_pair_overlap']==0
    assert audit['training_manifest_sha256']==plan['training_manifest_sha256']==sha(ROOT/'data/training/retention_v2/manifest.lock.json')
    completion=json.loads((run/'train_run/completion.json').read_text(encoding='utf8'));protocol=json.loads((run/'train_run/protocol.json').read_text(encoding='utf8'))
    assert completion['status']=='completed' and completion['optimizer_steps']==2574 and completion['pair_exposures']==246960
    assert protocol['code_sha256']==sha(ROOT/'scripts/train_retention_reranker.py')
    assert protocol['training_manifest_sha256']==audit['training_manifest_sha256']
    checks=[]
    for folder in sorted(run.glob('*_new_*'))+sorted(run.glob('*_legacy_*')):
        done=json.loads((folder/'completion.json').read_text(encoding='utf8'));p=json.loads((folder/'protocol.json').read_text(encoding='utf8'))
        assert done['status']=='completed' and sha(folder/'scores.jsonl')==done['scores_sha256']
        assert sha(folder/'protocol.json')==done['protocol_sha256']
        modelhash=p['model']['files_sha256']['model.safetensors']
        if folder.name.startswith('test_'):
            assert p['selection_lock_sha256']==sha(lockpath)
            assert datetime.fromisoformat(p['created_utc'])>=datetime.fromisoformat(lock['created_utc'])
            key='public_weights_sha256' if folder.name.endswith('public') else 'previous_weights_sha256' if folder.name.endswith('previous') else 'selected_weights_sha256'
            assert modelhash==lock[key]
        checks.append({'artifact':folder.relative_to(ROOT).as_posix(),'questions':done['questions'],'scores_sha256':done['scores_sha256']})
    summary=json.loads((run/'test_evaluation/summary.json').read_text(encoding='utf8'))
    assert sha(run/'test_evaluation/predictions.jsonl')==summary['predictions_sha256']
    historical=json.loads((run/'historical_government/summary.json').read_text(encoding='utf8'))
    for name in ['scores','predictions']:assert sha(run/f'historical_government/{name}.jsonl')==historical[name+'_sha256']
    modelpath=Path(lock['model_path']);assert sha(modelpath/'model.safetensors')==lock['selected_weights_sha256']
    selected=load_file(modelpath/'model.safetensors');previous=load_file(ROOT/'../models/insurerag-domain-reranker-v2/epoch-1/model.safetensors')
    assert selected.keys()==previous.keys();count=0;tensors=0;sums=0.;maximum=0.
    for name,a in selected.items():
        b=previous[name];assert a.shape==b.shape
        changed=int(torch.count_nonzero(a!=b));count+=changed;tensors+=int(changed>0);diff=(a.float()-b.float()).abs()
        sums+=float(diff.sum());maximum=max(maximum,float(diff.max()))
    parameters=sum(t.numel() for t in selected.values())
    if lock['promoted']:assert count>0
    report={'status':'verified','created_utc':datetime.now(timezone.utc).isoformat(),'selected_weights_sha256':lock['selected_weights_sha256'],
        'selected_epoch':lock['epoch'],'promoted':lock['promoted'],'domain_weight':lock['domain_weight'],
        'parameter_delta_from_previous_v2':{'total_parameters':parameters,'changed_values':count,'changed_tensors':tensors,
             'mean_absolute_change':sums/parameters,'maximum_absolute_change':maximum},
        'training_optimizer_steps':completion['optimizer_steps'],'training_pair_exposures':completion['pair_exposures'],
        'data_audit_sha256':sha(run/'data_audit.json'),'selection_lock_sha256':sha(lockpath),'score_artifacts':checks,
        'test_evaluation_sha256':sha(run/'test_evaluation/summary.json'),'historical_government_sha256':sha(run/'historical_government/summary.json'),
        'verification_code_sha256':sha(Path(__file__))}
    write_json(report,run/'verification.json');print(json.dumps(report,indent=2))

if __name__=='__main__':run()
