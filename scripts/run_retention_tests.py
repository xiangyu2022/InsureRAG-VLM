"""Run only locked model and predeclared controls on new and historical tests."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def call(*args):
    print('Running: '+' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def run():
    run=ROOT/'reports/retention_v2';lockpath=run/'selection.lock.json';lock=json.loads(lockpath.read_text(encoding='utf8'))
    candidates='reports/multidomain_v1/test_candidates'
    call('scripts/score_multidomain.py','candidates','--split','test','--model','../models/bge-small-en-v1.5','--output',candidates,'--selection',lockpath)
    for name,model in [('public','../models/ms-marco-MiniLM-L6-v2'),('previous','../models/insurerag-domain-reranker-v2/epoch-1'),('selected',lock['model_path'])]:
        call('scripts/score_multidomain.py','score','--split','test','--model',model,'--candidates',candidates,'--output',run/f'test_new_{name}','--selection',lockpath)
    call('scripts/score_domain_checkpoint.py','--split','test','--model',lock['model_path'],'--candidates','reports/insuranceqa_v2/rerank_test_v1',
         '--output',run/'test_legacy_selected','--selection',lockpath)
    call('scripts/eval_retention_v2.py','test')
    call('scripts/eval_retention_government_regression.py')

if __name__=='__main__':run()
