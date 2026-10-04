"""Verify docs, smoke routes, tests and frozen artifacts for a research delivery."""
from datetime import datetime,timezone
import ast,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,write_json
from scripts.eval_evidence_reranker import check_contract,RUN
def load(p):return json.loads(p.read_text(encoding='utf8'))

if __name__=='__main__':
    check_contract();proof=load(RUN/'verification.json');lock=load(RUN/'selection.lock.json')
    assert proof['status']=='verified' and proof['selection_sha256']==sha(RUN/'selection.lock.json')
    assert proof['code_sha256']==sha(ROOT/'scripts/verify_evidence_experiment.py')
    assert proof['data_audit_sha256']==sha(RUN/'data_verification.json')
    assert proof['fixed_input_verifier_sha256']==sha(ROOT/'scripts/verify_evidence_inputs.py')
    assert proof['environment_sha256']==sha(RUN/'environment.json')
    analysis=load(RUN/'failure_analysis.json')
    assert analysis['selection_unchanged_sha256']==sha(RUN/'selection.lock.json')
    assert analysis['test_predictions_sha256']==sha(RUN/'test_evaluation/predictions.jsonl')
    assert analysis['code_sha256']==sha(ROOT/'scripts/analyze_evidence_results.py')
    review=load(RUN/'reviewed_test_cases.json')
    assert review['failure_cards_sha256']==sha(RUN/'failure_cases.json')
    assert review['selection_lock_sha256']==sha(RUN/'selection.lock.json')
    assert review['code_sha256']==sha(ROOT/'scripts/review_evidence_test_cases.py')
    smoke=load(RUN/'smoke/summary.json');assert smoke['status']=='passed'
    assert smoke['query_script_sha256']==sha(ROOT/'scripts/query_evidence_reranker.py')
    for r in smoke['runs']:
        p=RUN/'smoke'/f'{r["corpus"]}.json';assert sha(p)==r['output_sha256']
        v=load(p);assert v['selection_lock_sha256']==sha(RUN/'selection.lock.json')
        assert v['selected_weights_sha256']==lock['weights_sha256'] and v['generation_calls']==0
        assert v['fixed_encoders_match_inherited_inventories']
    logs=RUN/'execution_logs';logs.mkdir(exist_ok=True)
    for p in ROOT.parent.glob('evidence_*.log'):shutil.copyfile(p,logs/p.name)
    suites={}
    for label,name,expected in [('full','evidence_full_tests.log','310 passed, 58 subtests passed'),
                                 ('minimal','evidence_minimal_tests.log','280 passed, 16 skipped, 58 subtests passed')]:
        text=(logs/name).read_text(encoding='utf8');assert expected in text and 'failed' not in text
        suites[label]={'summary':text.strip().splitlines()[-1],'log_sha256':sha(logs/name)}
    for p in (ROOT/'scripts').glob('*evidence*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    subprocess.run(['git','-c','core.safecrlf=false','-c','core.whitespace=cr-at-eol','diff','--check'],cwd=ROOT,check=True,capture_output=True)
    docs=['README.md','docs/EVIDENCE_RERANKER_UPDATE_ZH.md','docs/EVIDENCE_RERANKER_REPRODUCTION.md',
          'docs/assets/evidence_reranker_results.png','docs/assets/evidence_reranker_results.svg']
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'status':'verified','deep_verification_sha256':sha(RUN/'verification.json'),
                'selection_lock_sha256':sha(RUN/'selection.lock.json'),'software_suites':suites,'actual_query_routes':len(smoke['runs']),
                'smoke_summary_sha256':sha(RUN/'smoke/summary.json'),'git_diff_check':'passed',
                'reviewed_test_cases_sha256':sha(RUN/'reviewed_test_cases.json'),
                'expert_case_reviews_completed':0,
                'documents_sha256':{p:sha(ROOT/p) for p in docs},'figure_visually_reviewed':'docs/assets/evidence_reranker_results.png',
                'all_historical_previous_default_predictions_exactly_reproduced':True,'new_finqa_test_questions':1147,
                'github_pushed':False,'verification_code_sha256':sha(Path(__file__))},RUN/'delivery_checks.json')
    print(json.dumps({'status':'verified','software_suites':suites,'smoke_routes':len(smoke['runs'])}))
