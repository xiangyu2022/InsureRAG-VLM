"""Wait for the frozen test, then audit, analyze, document and smoke-test its result."""
from datetime import datetime,timezone
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def run(python,*args):
    print(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'command':args}),flush=True)
    subprocess.run([str(python),*args],cwd=ROOT,check=True)


if __name__=='__main__':
    target=ROOT/'reports/evidence_reranker_v1/test_evaluation/summary.json'
    deadline=time.monotonic()+3*3600
    while not target.exists():
        if time.monotonic()>deadline:raise TimeoutError('Frozen test was not completed')
        time.sleep(10)
    summary=json.loads(target.read_text(encoding='utf8'))
    assert summary['status']=='completed' and summary['split']=='test'
    run(sys.executable,'scripts/verify_evidence_experiment.py','--require-test')
    run(sys.executable,'scripts/analyze_evidence_results.py')
    run(sys.executable,'scripts/review_evidence_test_cases.py')
    run(ROOT/'../.venv/Scripts/python.exe','scripts/report_evidence_reranker.py')
    run(sys.executable,'scripts/smoke_evidence_retrieval.py')
    print(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
                      'status':'ready_for_manual_case_and_figure_review',
                      'delivery_verified':False}),flush=True)
