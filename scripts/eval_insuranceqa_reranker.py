"""Validation-only selection and one frozen, paired historical-test comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha, verify_fixture
from scripts.eval_insuranceqa_scale import aggregate, metrics
from src.insurerag_vlm.reranker import blend_scores


def configurations():
    return [{'pool': pool, 'lexical_weight': lw, 'cross_weight': cw}
            for pool in ['bge100', 'hybrid100', 'union200']
            for lw in [0.0, 0.2] for cw in [0.25, 0.5, 0.75, 1.0]]


def rank_row(row, config):
    ids = np.array(row['candidate_ids']); d = np.array(row['dense']); b = np.array(row['sparse']); c = np.array(row['cross'])
    if config['pool'] == 'bge100':
        keep = np.isin(ids, row['bge_ids'])
    elif config['pool'] == 'hybrid100':
        first = blend_scores(d, b, lexical_weight=0.2)
        indices = np.argsort(-first, kind='stable')[:100]
        keep = np.zeros(len(ids), dtype=bool); keep[indices] = True
    elif config['pool'] == 'union200':
        keep = np.ones(len(ids), dtype=bool)
    else:
        raise ValueError('Unknown candidate policy')
    scores = blend_scores(d[keep], b[keep], c[keep], config['lexical_weight'], config['cross_weight'])
    return ids[keep][np.argsort(-scores, kind='stable')[:100]].tolist()


def summarize(rows, cases):
    groups = {c['id']: c for c in cases}
    return {**aggregate(rows),
            'no_exact_overlap': aggregate([r for r in rows if not groups[r['id']]['exact_question_overlap_with_train_or_valid']]),
            'by_domain': {domain: aggregate([r for r in rows if groups[r['id']]['domain'] == domain])
                          for domain in sorted({c['domain'] for c in cases})}}


def paired_summary(new, baseline, cases):
    from scipy.stats import binomtest
    ids = [c['id'] for c in cases]
    a = {r['id']: r['hit_at_10'] for r in new}; b = {r['id']: r['hit_at_10'] for r in baseline}
    delta = np.array([a[i]-b[i] for i in ids]); wins = int(np.sum(delta == 1)); losses = int(np.sum(delta == -1))
    # Connected components of shared gold answer IDs are resampled as clusters.
    parents = list(range(len(cases)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owner = {}
    for i, case in enumerate(cases):
        for label in case['gold_answer_ids']:
            if label in owner:
                parents[find(i)] = find(owner[label])
            owner[label] = i
    clusters = {}
    for i in range(len(cases)):
        clusters.setdefault(find(i), []).append(i)
    sums = np.array([delta[g].sum() for g in clusters.values()]); sizes = np.array([len(g) for g in clusters.values()])
    rng = np.random.default_rng(42); bootstrap = []
    for _ in range(5000):
        sample = rng.integers(0, len(sums), len(sums)); bootstrap.append(sums[sample].sum()/sizes[sample].sum())
    return {'n': len(cases), 'wins': wins, 'losses': losses, 'ties': len(cases)-wins-losses,
            'hit10_difference': float(delta.mean()),
            'label_cluster_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'clusters': len(clusters), 'draws': 5000, 'seed': 42,
            'paired_exact_binomial_p': float(binomtest(wins, wins+losses, .5).pvalue) if wins+losses else 1.0,
            'interpretation': 'Historical test was inspected previously. Cluster intervals do not establish independent insurance-domain generalization.'}


def run(args):
    verify_fixture(args.fixture)
    protocol = json.loads((args.scores/'protocol.json').read_text())
    completion = json.loads((args.scores/'completion.json').read_text())
    if completion['status'] != 'completed' or completion['scores_sha256'] != sha(args.scores/'scores.jsonl'):
        raise ValueError('Incomplete or modified neural scoring artifact')
    if protocol['fixture_lock_sha256'] != sha(args.fixture/'manifest.lock.json'):
        raise ValueError('Fixture differs from scored data')
    split = protocol['split']; cases = read_jsonl(args.fixture/f'{split}.jsonl')
    scored = read_jsonl(args.scores/'scores.jsonl')
    if [r['id'] for r in scored] != [c['id'] for c in cases]:
        raise ValueError('Scoring rows do not exactly match the frozen split')
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'src/insurerag_vlm/reranker.py', ROOT/'scripts/eval_insuranceqa_scale.py']
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    gold = {c['id']: set(c['gold_answer_ids']) for c in cases}
    # Require exact parity with the original retrieval rankings before comparing.
    old_rows = [r for r in read_jsonl(args.baseline/'predictions.jsonl')
                if r['split'] == split and r['scope'] == 'all_27413_answers']
    old = {(r['id'], r['arm']): r for r in old_rows}
    for row in scored:
        for arm, field in [('bge', 'bge_ids'), ('bm25', 'bm25_ids')]:
            if row[field] != old[(row['id'], arm)]['top_answer_ids']:
                raise ValueError(f'Cached embedding or BM25 ranking parity failed: {row["id"]} {arm}')
    if args.select:
        if split != 'valid' or args.selection:
            raise ValueError('Hyperparameter selection is allowed only on validation')
        write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'selection_split': 'valid',
                    'configurations': configurations(), 'rule': 'Max Hit@10, then MRR@100, then Hit@1, stable configuration order',
                    'code_sha256': code}, args.output/'selection_protocol.json')
        sweep = []
        for cfg in configurations():
            rows = [metrics(rank_row(r, cfg), gold[r['id']]) for r in scored]
            sweep.append({'config': cfg, **aggregate(rows)})
        sweep.sort(key=lambda r: (r['hit_at_10'], r['mrr_at_100'], r['hit_at_1']), reverse=True)
        write_json(sweep, args.output/'validation_sweep.json')
        selected = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_on': 'valid',
            'config': sweep[0]['config'], 'validation_metrics': sweep[0],
            'fixture_lock_sha256': sha(args.fixture/'manifest.lock.json'),
            'scoring_protocol_sha256': sha(args.scores/'protocol.json'),
            'scored_validation_sha256': completion['scores_sha256'], 'code_sha256': code,
            'ablation_configs': {
                'bge_plus_cross_same_budget': {**sweep[0]['config'], 'pool': 'bge100', 'lexical_weight': 0.0},
                'pure_cross_union': {'pool': 'union200', 'lexical_weight': 0.0, 'cross_weight': 1.0},
                'linear_no_cross': {'pool': 'union200', 'lexical_weight': 0.2, 'cross_weight': 0.0}},
            'test_status': 'Original historical test inspected before this work; no new-config test metrics used for selection.',
            'model_training': 'Public MS MARCO checkpoint, no InsuranceQA fine-tuning'}
        write_json(selected, args.output/'selection.lock.json')
    else:
        if not args.selection:
            raise ValueError('A validation selection is required')
        selected = json.loads(args.selection.read_text())
        if selected['selected_on'] != 'valid' or selected['fixture_lock_sha256'] != sha(args.fixture/'manifest.lock.json'):
            raise ValueError('Invalid validation lock')
        if selected['code_sha256'] != code:
            raise ValueError('Evaluation implementation differs from validation selection')
        if protocol['selection_lock_sha256'] != sha(args.selection):
            raise ValueError('Neural test scoring did not use this selection lock')
    arms = {'selected': selected['config'], **selected['ablation_configs']}
    predictions = []; summaries = {}
    for arm in ['bge', 'bm25', 'rrf']:
        rows = [old[(r['id'], arm)] for r in scored]
        summaries[arm] = summarize(rows, cases)
        predictions.extend(rows)
    for arm, config in arms.items():
        rows = []
        for row in scored:
            order = rank_row(row, config)
            rows.append({'id': row['id'], 'split': split, 'arm': arm, 'scope': 'all_27413_answers',
                         'top_answer_ids': order, **metrics(order, gold[row['id']])})
        predictions.extend(rows); summaries[arm] = summarize(rows, cases)
    with (args.output/'predictions.jsonl').open('w', encoding='utf8') as handle:
        for row in predictions:
            handle.write(json.dumps(row)+'\n')
    selected_rows = [r for r in predictions if r['arm'] == 'selected']
    paired = {arm: paired_summary(selected_rows, [r for r in predictions if r['arm'] == arm], cases)
              for arm in ['bge', 'rrf', 'bge_plus_cross_same_budget']}
    sensitivity = None
    audit_path = args.baseline/'sensitivity_audit.json'
    if split == 'test' and audit_path.exists():
        previous_audit = json.loads(audit_path.read_text())
        excluded = {r['id'] for r in previous_audit['near_duplicates']}
        sensitivity = {'rule': previous_audit['near_duplicate_rule'], 'audit_sha256': sha(audit_path),
            'selection_used_this_slice': False,
            'summaries': {arm: aggregate([r for r in predictions if r['arm'] == arm and r['id'] not in excluded])
                          for arm in summaries}}
    write_json({'status': 'completed', 'split': split, 'config': selected['config'], 'summaries': summaries,
                'paired_selected_minus_baseline': paired, 'baseline_rank_parity': True,
                'historical_near_duplicate_exclusion': sensitivity,
                'code_sha256': code, 'predictions_sha256': sha(args.output/'predictions.jsonl'),
                'selection_lock_sha256': sha(args.selection or args.output/'selection.lock.json'),
                'scoring_protocol_sha256': sha(args.scores/'protocol.json')}, args.output/'summary.json')
    print(json.dumps({'split': split, 'selected': selected['config'],
        'metrics': {a: {k: s[k] for k in ['n', 'hit_at_1', 'hit_at_10', 'mrr_at_100']} for a, s in summaries.items()},
        'paired': paired}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, default=ROOT/'data/benchmarks/insuranceqa_v2')
    p.add_argument('--baseline', type=Path, default=ROOT/'reports/insuranceqa_v2/retrieval_frozen')
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--select', action='store_true')
    p.add_argument('--selection', type=Path)
    run(p.parse_args())
