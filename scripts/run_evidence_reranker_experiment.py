"""Run the fixed two-seed experiment through validation selection and once-only test."""
import json
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]


def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


if __name__=='__main__':
    target=ROOT/'data/training/evidence_reranker_v1/manifest.lock.json';deadline=time.monotonic()+7200
    while not target.exists():
        if time.monotonic()>deadline:raise TimeoutError('Training candidate mining did not finish')
        time.sleep(10)
    run('scripts/register_evidence_reranker_plan.py')
    run('scripts/eval_evidence_reranker.py','score','--split','valid','--model','previous')
    for seed in [42,123]:
        run('scripts/train_evidence_reranker.py','--seed',str(seed))
        for epoch in [1,2]:run('scripts/eval_evidence_reranker.py','score','--split','valid','--model',f'seed_{seed}_epoch_{epoch}')
    run('scripts/eval_evidence_reranker.py','evaluate','--split','valid')
    selection=json.loads((ROOT/'reports/evidence_reranker_v1/selection.lock.json').read_text(encoding='utf8'))
    run('scripts/eval_evidence_reranker.py','score','--split','test','--model','previous')
    if selection['config']['model']!='previous':run('scripts/eval_evidence_reranker.py','score','--split','test','--model',selection['config']['model'])
    run('scripts/eval_evidence_reranker.py','evaluate','--split','test')
