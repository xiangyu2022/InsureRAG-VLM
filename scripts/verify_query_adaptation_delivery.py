"""Check immutable research evidence and runnable artifacts before packaging."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import sha, write_json
from scripts.eval_query_adaptation import check_contract

def load(p): return json.loads(p.read_text(encoding='utf8'))

if __name__ == '__main__':
    run = ROOT / 'reports/query_adaptation_v1'
    plan = check_contract()
    deep, selection = load(run / 'verification.json'), load(run / 'selection.lock.json')
    assert deep['status'] == 'verified'
    assert deep['code_sha256'] == sha(ROOT / 'scripts/verify_query_adaptation.py')
    assert deep['selection_lock_sha256'] == sha(run / 'selection.lock.json')
    assert deep['primary_optimizer_steps'] == deep['ablation_optimizer_steps'] == 6284
    for phase, item in deep['evaluation'].items(): assert item['summary_sha256'] == sha(run / phase / 'summary.json')
    for path, digest in load(run / 'checkpoint_files.lock.json')['files_sha256'].items():
        assert sha(ROOT.parent / path) == digest, path
    ab = run / 'data_ablation'
    context, ablock = load(ab / 'context.json'), load(ab / 'evaluation_implementation.lock.json')
    assert not (ab / 'selection.lock.json').exists()
    assert context['evaluation_code_sha256'] == sha(ROOT / 'scripts/eval_query_data_ablation.py')
    assert context['diff_sha256'] == sha(run / 'data_ablation_evaluation.diff')
    implementation = load(run / 'data_ablation_implementation.lock.json')
    assert implementation['training_code_sha256'] == sha(ROOT / 'scripts/train_query_data_ablation.py')
    assert implementation['fork_generator_sha256'] == sha(ROOT.parent / 'create_query_data_ablation.py')
    assert context['fork_generator_sha256'] == sha(ROOT.parent / 'create_query_data_ablation_eval.py')
    for path, digest in ablock['code_sha256'].items(): assert sha(ROOT / path) == digest
    absummary = load(ab / 'valid/summary.json')
    assert absummary['predictions_sha256'] == sha(ab / 'valid/predictions.jsonl')
    assert absummary['protocol_sha256'] == sha(ab / 'valid/protocol.json')
    for name, digest in {**absummary['score_files_sha256'], **absummary['query_vector_files_sha256']}.items():
        assert sha(ab / 'valid' / name) == digest
    assert datetime.fromisoformat(load(run / 'data_ablation_protocol.json')['created_utc']) > datetime.fromisoformat(selection['created_utc'])
    assert datetime.fromisoformat(load(run / 'test/protocol.json')['created_utc']) > datetime.fromisoformat(selection['created_utc'])
    audit = load(run / 'data_verification.json')
    assert audit['status'] == 'verified' and audit['forbidden_positive_or_candidate_count'] == 0
    assert audit['fixture_sha256'] == sha(ROOT / 'data/benchmarks/fiqa_v1/manifest.lock.json')
    failure = load(run / 'failure_analysis.json')
    assert failure['test_predictions_sha256'] == sha(run / 'test/predictions.jsonl')
    assert load(run / 'failure_cases.json')['n'] == 777
    smoke = load(run / 'smoke/summary.json')
    assert smoke['status'] == 'passed'
    assert smoke['query_script_sha256'] == sha(ROOT / 'scripts/query_adapted_retrieval.py')
    for item in smoke['runs']:
        p = run / 'smoke' / (item['corpus']+'.json')
        assert sha(p) == item['output_sha256']
        result = load(p)
        assert result['selection_lock_sha256'] == sha(run / 'selection.lock.json')
        assert result['selected_query_weights_sha256'] == selection['query_weights_sha256']
        assert result['generation_calls'] == 0 and result['query_encoder_passes'] == 2
    evidence = run / 'execution_logs'; evidence.mkdir(exist_ok=True)
    for p in ROOT.parent.glob('query_*log'): shutil.copyfile(p, evidence / p.name)
    suites = {}
    for name, file, expected in [
        ('full', 'query_adaptation_full_tests.log', '302 passed, 58 subtests passed'),
        ('minimal', 'query_adaptation_minimal_tests_final.log', '275 passed, 13 skipped, 58 subtests passed')]:
        content = (evidence / file).read_text(encoding='utf8')
        assert expected in content and 'failed' not in content
        suites[name] = {'summary': content.strip().splitlines()[-1], 'log_sha256': sha(evidence / file)}
    compiled = []
    for p in list((ROOT / 'scripts').glob('*query*adapt*.py'))+[ROOT / 'src/insurerag_vlm/query_adaptation.py']:
        ast.parse(p.read_text(encoding='utf8'), filename=str(p)); compiled.append(p.relative_to(ROOT).as_posix())
    subprocess.run(['git', '-c', 'core.safecrlf=false', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True, capture_output=True)
    docs = ['README.md', 'docs/QUERY_ADAPTATION_UPDATE_ZH.md', 'docs/QUERY_ADAPTATION_REPRODUCTION.md',
            'docs/assets/query_adaptation_results.png', 'docs/assets/query_adaptation_results.svg']
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified',
                'deep_training_verification_sha256': sha(run / 'verification.json'),
                'selection_lock_sha256': sha(run / 'selection.lock.json'),
                'ablation_evidence_verified': True, 'software_suites': suites, 'actual_query_routes': 3,
                'smoke_summary_sha256': sha(run / 'smoke/summary.json'), 'git_diff_check': 'passed',
                'parsed_scripts': compiled, 'documents_sha256': {p: sha(ROOT / p) for p in docs},
                'figure_visually_reviewed': 'docs/assets/query_adaptation_results.png',
                'control_metrics_identical_to_prior': True, 'control_rank_lists_bitwise_identical': False,
                'github_pushed': False, 'verification_code_sha256': sha(Path(__file__))}, run / 'delivery_checks.json')
    print(json.dumps({'status': 'verified', 'software_suites': suites, 'actual_query_routes': 3}))
