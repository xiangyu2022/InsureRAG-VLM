"""Execute the preregistered selected-model tests and matched controls once."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def command(*args):
    values=list(map(str,args))
    for flag in ['--questions','--answers','--candidates','--model','--output']:
        if flag in values:
            i=values.index(flag)+1;values[i]=str((ROOT/Path(values[i])).resolve())
    print('Running: '+' '.join(values),flush=True);subprocess.run([sys.executable,*values],cwd=ROOT,check=True)
def run():
    lock=json.loads((ROOT/'reports/condition_v1/selection.lock.json').read_text(encoding='utf8'));model=lock['model_path']
    # Each historical cohort keeps its original answer corpus and candidate rows.
    for name,questions,answers,candidates in [
        ('legacy','data/benchmarks/insuranceqa_v2/test.jsonl','data/benchmarks/insuranceqa_v2/answers.jsonl','reports/retention_v2/test_legacy_selected'),
        ('mixed','data/benchmarks/multidomain_v1/test.jsonl','data/training/retention_v2/answers.jsonl','reports/retention_v2/test_new_selected'),
        ('oldgov','data/benchmarks/hicric_government_qa_v1/questions.jsonl','reports/condition_v1/historical_government_answers.jsonl','reports/retention_v2/historical_government')]:
        if name=='oldgov':
            rows=[]
            for path in ['data/benchmarks/insuranceqa_v2/answers.jsonl','data/benchmarks/hicric_government_qa_v1/answers.jsonl']:
                rows.extend((ROOT/path).read_text(encoding='utf8').splitlines())
            (ROOT/answers).write_text('\n'.join(rows)+'\n',encoding='utf8')
        command('scripts/score_condition_model.py','score','--split','test','--questions',questions,'--answers',answers,'--candidates',candidates,
            '--model',model,'--output',f'reports/condition_v1/test_{name}_selected')
    question='data/benchmarks/condition_v1/test.jsonl';answers='data/benchmarks/condition_v1/answers.jsonl';candidates='reports/condition_v1/test_fresh_candidates'
    command('scripts/score_condition_model.py','candidates','--split','test','--questions',question,'--answers',answers,
        '--model','../models/bge-small-en-v1.5','--output',candidates)
    for name,path in [('public','../models/ms-marco-MiniLM-L6-v2'),('previous','../models/insurerag-retention-v2/epoch-2'),('selected',model)]:
        command('scripts/score_condition_model.py','score','--split','test','--questions',question,'--answers',answers,'--candidates',candidates,
            '--model',path,'--output',f'reports/condition_v1/test_fresh_{name}')
    command('scripts/eval_condition_v1.py','test')
if __name__=='__main__':run()
