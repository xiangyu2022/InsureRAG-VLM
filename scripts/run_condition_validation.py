"""Run the two registered trainings and their validation, never opening test metrics."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def command(*args):
    print('Running: '+' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def run():
    assert (ROOT/'data/training/condition_v1/manifest.lock.json').is_file(),'Mining must complete first'
    for seed in [42,123]:
        command('scripts/train_condition_reranker.py','--seed',seed)
        model=f'../models/insurerag-condition-v1-seed-{seed}/epoch-1'
        for label,questions,answers,candidates in [
            ('legacy','data/benchmarks/insuranceqa_v2/valid.jsonl','data/benchmarks/insuranceqa_v2/answers.jsonl','reports/retention_v2/valid_legacy_epoch_2'),
            ('mixed','data/benchmarks/multidomain_v1/valid.jsonl','data/training/retention_v2/answers.jsonl','reports/retention_v2/valid_new_epoch_2')]:
            command('scripts/score_condition_model.py','score','--split','valid','--questions',questions,'--answers',answers,'--candidates',candidates,
                '--model',model,'--output',f'reports/condition_v1/valid_{label}_seed_{seed}')
    command('scripts/eval_condition_v1.py','select')
if __name__=='__main__':run()
