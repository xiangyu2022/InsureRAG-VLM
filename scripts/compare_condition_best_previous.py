"""Supplementary control: the previous model at its own best validation setting."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_condition_listwise import predict,scored
from scripts.eval_insuranceqa_scale import aggregate
from scripts.eval_insuranceqa_reranker import paired_summary
from scripts.eval_retention_v2 import source_paired

def run(phase):
    run=ROOT/'reports/condition_listwise_v1';planpath=run/'best_previous_control_protocol.json'
    if phase=='plan':
        assert not planpath.exists() and not (run/'test_evaluation').exists()
        parent=ROOT/'reports/condition_v1/selection.lock.json';lock=json.loads(parent.read_text(encoding='utf8'))
        assert lock['selected_model']=='previous' and not lock['promoted_weights']
        write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Report the prior model at its own validation-optimal config in addition to identical-config and original-config controls.',
            'config':lock['config'],'previous_weights_sha256':lock['previous_weights_sha256'],'source_validation_selection_sha256':sha(parent),
            'used_for_new_model_selection':False,'test_has_not_been_scored':True,'code_sha256':sha(Path(__file__))},planpath);return
    plan=json.loads(planpath.read_text(encoding='utf8'));assert plan['code_sha256']==sha(Path(__file__))
    lock=json.loads((run/'selection.lock.json').read_text(encoding='utf8'));assert lock['best_previous_configuration']['config']==plan['config']
    predictions=read_jsonl(run/'test_evaluation/predictions.jsonl');selected={(r['cohort'],r['id']):r for r in predictions if r['arm']=='selected'}
    output={}
    for cohort,casepath,oldpath in [
        ('historical_insuranceqa','data/benchmarks/insuranceqa_v2/test.jsonl','reports/retention_v2/test_legacy_selected'),
        ('historical_multidomain','data/benchmarks/multidomain_v1/test.jsonl','reports/retention_v2/test_new_selected'),
        ('historical_government','data/benchmarks/hicric_government_qa_v1/questions.jsonl','reports/retention_v2/historical_government'),
        ('fresh_government','data/benchmarks/condition_v1/test.jsonl','reports/condition_listwise_v1/test_fresh_previous')]:
        cases=read_jsonl(ROOT/casepath)
        if cohort=='historical_government':cases=[{**c,'source_group':c['source_url']} for c in cases]
        rows=read_jsonl(ROOT/oldpath/'scores.jsonl') if cohort=='historical_government' else scored(ROOT/oldpath)
        old=predict(rows,cases,plan['config'],'previous_validation_best');new=[selected[(cohort,c['id'])] for c in cases]
        partitions={'all':list(range(len(cases)))} if cohort!='historical_multidomain' else {d:[i for i,c in enumerate(cases) if c['domain']==d] for d in ['government','general']}
        output[cohort]={}
        for name,ix in partitions.items():
            a=[new[i] for i in ix];b=[old[i] for i in ix];c=[cases[i] for i in ix]
            compare=paired_summary if cohort=='historical_insuranceqa' else source_paired
            output[cohort][name]={'previous_validation_best':aggregate(b),'selected_minus_previous_validation_best':compare(a,b,c)}
    write_json({'results':output,'config':plan['config'],'protocol_sha256':sha(planpath),'selection_lock_sha256':sha(run/'selection.lock.json'),
        'status':'supplementary validation-fixed comparison; no test-based choice'},run/'best_previous_control.json')
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['plan','test']);run(p.parse_args().phase)
