"""Run the finite, registered retention validation jobs sequentially on one GPU."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def call(*args):
    print('Running: '+' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def run():
    run=ROOT/'reports/retention_v2';completion=run/'train_run/completion.json'
    assert json.loads(completion.read_text(encoding='utf8'))['status']=='completed'
    candidates='reports/multidomain_v1/valid_candidates'
    call('scripts/score_multidomain.py','candidates','--split','valid','--model','../models/bge-small-en-v1.5','--output',candidates)
    jobs=[('public','../models/ms-marco-MiniLM-L6-v2'),('previous','../models/insurerag-domain-reranker-v2/epoch-1')]
    jobs += [(f'epoch_{e}',f'../models/insurerag-retention-v2/epoch-{e}') for e in [1,2]]
    for name,model in jobs:
        call('scripts/score_multidomain.py','score','--split','valid','--model',model,'--candidates',candidates,'--output',run/f'valid_new_{name}')
        if name.startswith('epoch_'):
            call('scripts/score_domain_checkpoint.py','--split','valid','--model',model,'--candidates','reports/insuranceqa_v2/rerank_valid_v2','--output',run/f'valid_legacy_{name}')
    call('scripts/eval_retention_v2.py','select')

if __name__=='__main__':run()
