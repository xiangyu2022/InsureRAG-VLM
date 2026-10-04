"""Conditionally launch a registered validation-only follow-up after the first recipe."""
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def command(*args):
    print('Running: '+' '.join(map(str,args)),flush=True);subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def run():
    parent=ROOT/'reports/condition_v1';start=time.monotonic()
    while not (parent/'selection.lock.json').exists():
        if time.monotonic()-start>3600:raise TimeoutError('Parent validation did not finish')
        time.sleep(5)
    lock=json.loads((parent/'selection.lock.json').read_text(encoding='utf8'))
    if lock['promoted_weights']:
        print('Parent passed promotion; follow-up not triggered.',flush=True);return
    out=ROOT/'reports/condition_listwise_v1';out.mkdir(parents=True,exist_ok=False)
    for name in ['fixture_verification.json','length_audit.json','failure_diagnosis.json','preflight_amendment.json','runtime.json','followup_intent.json']:
        shutil.copyfile(parent/name,out/name)
    command('scripts/eval_condition_listwise.py','plan')
    command('scripts/prepare_condition_listwise_training.py')
    command('scripts/run_condition_listwise_validation.py')
if __name__=='__main__':run()
