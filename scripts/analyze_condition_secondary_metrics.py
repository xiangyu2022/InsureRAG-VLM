"""Post-test uncertainty for already-reported Hit@1 and MRR; no model selection."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json


def clusters(cases, shared_labels=False):
    if not shared_labels:
        groups = defaultdict(list)
        for i, c in enumerate(cases):
            groups[c.get('source_group') or c['source_url']].append(i)
        return list(groups.values())
    parents = list(range(len(cases)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owner = {}
    for i, c in enumerate(cases):
        for label in c['gold_answer_ids']:
            if label in owner:
                parents[find(i)] = find(owner[label])
            owner[label] = i
    groups = defaultdict(list)
    for i in range(len(cases)):
        groups[find(i)].append(i)
    return list(groups.values())


def main():
    run = ROOT / 'reports/condition_listwise_v1'
    lock = json.loads((run / 'selection.lock.json').read_text(encoding='utf8'))
    best = json.loads((run / 'best_previous_control.json').read_text(encoding='utf8'))
    summary = json.loads((run / 'test_evaluation/summary.json').read_text(encoding='utf8'))
    assert best['config'] == lock['config'], 'Matched is not the validation-best control'
    path = run / 'test_evaluation/predictions.jsonl'
    assert sha(path) == summary['predictions_sha256']
    predictions = defaultdict(dict)
    for r in read_jsonl(path):
        key = (r['cohort'], r['arm'])
        assert r['id'] not in predictions[key]
        predictions[key][r['id']] = r
    results = {}
    cohorts = [
        ('historical_insuranceqa', 'data/benchmarks/insuranceqa_v2/test.jsonl', ['all']),
        ('historical_multidomain', 'data/benchmarks/multidomain_v1/test.jsonl', ['government', 'general']),
        ('historical_government', 'data/benchmarks/hicric_government_qa_v1/questions.jsonl', ['all']),
        ('fresh_government', 'data/benchmarks/condition_v1/test.jsonl', ['all']),
    ]
    for cohort, fixture, parts in cohorts:
        all_cases = read_jsonl(ROOT / fixture)
        for arm in ['selected', 'previous_matched']:
            assert set(predictions[(cohort, arm)]) == {c['id'] for c in all_cases}
        for part in parts:
            cases = [c for c in all_cases if part == 'all' or c['domain'] == part]
            groups = clusters(cases, cohort == 'historical_insuranceqa')
            sizes = np.array([len(g) for g in groups])
            seed = 42 if cohort == 'historical_insuranceqa' else 20261001
            draws = np.random.default_rng(seed).integers(0, len(groups), (5000, len(groups)))
            output = {}
            for metric in ['hit_at_1', 'mrr_at_100']:
                old = np.array([predictions[(cohort, 'previous_matched')][c['id']][metric] for c in cases])
                new = np.array([predictions[(cohort, 'selected')][c['id']][metric] for c in cases])
                assert np.isclose(old.mean(), best['results'][cohort][part]['previous_validation_best'][metric])
                assert np.isclose(new.mean(), summary['summaries'][cohort][part]['selected'][metric])
                delta = new - old
                sums = np.array([delta[g].sum() for g in groups])
                boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
                output[metric] = {
                    'previous_validation_best': float(old.mean()),
                    'selected': float(new.mean()),
                    'difference': float(delta.mean()),
                    'cluster_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
                    'improved_questions': int((delta > 0).sum()),
                    'worsened_questions': int((delta < 0).sum()),
                    'unchanged_questions': int((delta == 0).sum()),
                }
            results[f'{cohort}/{part}'] = {
                'n': len(cases), 'clusters': len(groups), 'seed': seed,
                'cluster_unit': 'shared-gold-answer connected component' if cohort == 'historical_insuranceqa' else 'source URL/title',
                'metrics': output,
            }
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Post-test descriptive uncertainty, not used for selection or training',
        'comparison': 'Selected versus previous weights at their validation-best configuration; same as matched configuration in this run',
        'bootstrap_repetitions': 5000, 'multiple_comparisons_adjusted': False,
        'interpretation': 'Nominal marginal intervals, not simultaneous intervals. Historical tests have been inspected repeatedly. Fresh government has 28 URLs, not 157 independent sources.',
        'results': results, 'prediction_sha256': sha(path),
        'selection_lock_sha256': sha(run / 'selection.lock.json'),
        'analysis_code_sha256': sha(Path(__file__)),
    }
    write_json(report, run / 'secondary_metrics.json')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
