"""Verify final smoke evidence and immutable experiment artifacts for delivery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import sha, write_json


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    run = ROOT / 'reports/condition_listwise_v1'
    verification = load(run / 'verification.json')
    lock = load(run / 'selection.lock.json')
    assert verification['status'] == 'verified'
    assert sha(run / 'selection.lock.json') == verification['selection_lock_sha256']
    assert sha(run / 'test_evaluation/summary.json') == verification['test_evaluation_sha256']
    assert sha(ROOT / 'data/benchmarks/condition_v1/manifest.lock.json') == verification['fixture_manifest_sha256']
    assert sha(ROOT / 'data/training/condition_listwise_v1/manifest.lock.json') == verification['training_manifest_sha256']
    model = ROOT / '../models/insurerag-condition-listwise-v1-seed-123/epoch-1/model.safetensors'
    assert sha(model) == lock['selected_weights_sha256']
    for name, digest in lock['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for row in verification['score_artifacts']:
        assert sha(run / row['artifact'] / 'scores.jsonl') == row['scores_sha256'], row['artifact']
    secondary = load(run / 'secondary_metrics.json')
    assert secondary['analysis_code_sha256'] == sha(ROOT / 'scripts/analyze_condition_secondary_metrics.py')
    assert secondary['prediction_sha256'] == sha(run / 'test_evaluation/predictions.jsonl')
    repair = load(run / 'boundary_repair_diagnostic.json')
    assert repair['repair_parser_sha256'] == sha(ROOT / 'scripts/extract_government_qa_boundaries_v2.py')
    for name, digest in repair['frozen_fixture_hashes_before_and_after'].items():
        assert sha(ROOT / 'data/benchmarks/condition_v1' / name) == digest, name
    queries = []
    for corpus, count in [('condition', 48172), ('insuranceqa', 27413), ('hicric', 78830)]:
        path = run / f'query_{corpus}_smoke.json'
        result = load(path)
        assert result['corpus'] == corpus and result['answer_candidates'] == count
        assert result['config'] == lock['config']
        assert result['generation_calls'] == 0 and result['research_retrieval_only']
        assert result['model_weights_sha256'] == lock['selected_weights_sha256']
        assert result['selection_lock_sha256'] == sha(run / 'selection.lock.json')
        assert [r['rank'] for r in result['results']] == [1, 2, 3]
        assert len({r['answer_id'] for r in result['results']}) == 3
        assert all(r['text'].strip() for r in result['results'])
        queries.append({'corpus': corpus, 'answer_candidates': count,
                        'artifact': path.name, 'sha256': sha(path), 'execution_passed': True,
                        'expert_answer_quality_judged': False})
    suites = {}
    for name, expected in [('full', '294 passed, 58 subtests passed'),
                           ('minimal', '273 passed, 9 skipped, 58 subtests passed')]:
        path = run / f'pytest_delivery_{name}.txt'
        content = path.read_text(encoding='utf8')
        assert expected in content and 'failed' not in content
        suites[name] = {'summary': content.strip().splitlines()[-1], 'log': path.name, 'sha256': sha(path)}
    subprocess.run(['git', '-c', 'core.safecrlf=false', '-c', 'core.whitespace=cr-at-eol',
                    'diff', '--check'], cwd=ROOT, check=True, capture_output=True)
    write_json({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified',
        'immutable_experiment_artifacts_rechecked': True,
        'deep_training_verification_sha256': sha(run / 'verification.json'),
        'selected_weights_sha256': lock['selected_weights_sha256'],
        'smoke_queries': queries, 'software_suites': suites, 'git_diff_check': 'passed',
        'future_parser_changes_frozen_fixture': False,
        'post_test_parser_repair_used_to_rescore': False,
        'figure_visually_reviewed': 'docs/assets/condition_results.png',
        'verification_code_sha256': sha(Path(__file__)),
    }, run / 'delivery_checks.json')
    print(json.dumps({'status': 'verified', 'software_suites': suites, 'smoke_queries': len(queries)}))


if __name__ == '__main__':
    main()
