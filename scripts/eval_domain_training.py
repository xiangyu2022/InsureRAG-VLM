"""Select domain checkpoints on validation only, then evaluate frozen historical test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha
from scripts.eval_insuranceqa_scale import aggregate, metrics
from scripts.eval_insuranceqa_reranker import rank_row, summarize, paired_summary

FILES = [Path(__file__), ROOT/'scripts/eval_insuranceqa_reranker.py',
         ROOT/'scripts/eval_insuranceqa_scale.py', ROOT/'src/insurerag_vlm/reranker.py']


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in FILES}


def verified_scores(folder, split):
    completion = json.loads((folder/'completion.json').read_text(encoding='utf8'))
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf8'))
    if completion['status'] != 'completed' or completion['scores_sha256'] != sha(folder/'scores.jsonl'):
        raise ValueError('Scoring artifact incomplete or changed')
    if protocol['split'] != split:
        raise ValueError('Wrong split')
    return read_jsonl(folder/'scores.jsonl'), protocol


def predictions(scored, cases, config, arm):
    if [r['id'] for r in scored] != [c['id'] for c in cases]:
        raise ValueError('Question identity/order changed')
    rows = []
    for row, case in zip(scored, cases):
        order = rank_row(row, config)
        rows.append({'id': row['id'], 'arm': arm, 'top_answer_ids': order,
                     **metrics(order, set(case['gold_answer_ids']))})
    return rows


def run(args):
    folder = args.run
    plan_path = folder/'selection_protocol.json'
    if args.phase == 'plan':
        if plan_path.exists():
            raise ValueError('Do not overwrite a registered selection protocol')
        write_json({'created_utc': datetime.now(timezone.utc).isoformat(),
            'selected_on': 'valid', 'epochs': [1, 2, 3],
            'configurations': [{'pool': 'hybrid100', 'lexical_weight': .2, 'cross_weight': w}
                               for w in [.25, .5, .75, 1.]],
            'rule': 'Maximum validation Hit@10, then MRR@100, then Hit@1; stable epoch/config order',
            'fallback': 'Keep previous checkpoint if the best trained model fails to exceed previous validation Hit@10',
            'primary_endpoint': 'Hit@10 over the complete 27,413-answer corpus',
            'test_use': 'No test metrics used for checkpoint or fusion selection',
            'code_sha256': code_hashes()}, plan_path)
        return
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if plan['code_sha256'] != code_hashes():
        raise ValueError('Registered evaluation implementation changed')
    fixture = ROOT/'data/benchmarks/insuranceqa_v2'
    if args.phase == 'select':
        cases = read_jsonl(fixture/'valid.jsonl'); sweep = []; sources = {}
        for epoch in plan['epochs']:
            scores_path = folder/f'valid_epoch_{epoch}'
            rows, protocol = verified_scores(scores_path, 'valid')
            sources[str(epoch)] = {'protocol_sha256': sha(scores_path/'protocol.json'),
                                  'scores_sha256': sha(scores_path/'scores.jsonl')}
            for cfg in plan['configurations']:
                result = predictions(rows, cases, cfg, 'candidate')
                sweep.append({'epoch': epoch, 'config': cfg, **aggregate(result)})
        sweep.sort(key=lambda r: (r['hit_at_10'], r['mrr_at_100'], r['hit_at_1']), reverse=True)
        previous = json.loads((ROOT/'reports/insuranceqa_v2/rerank_selection_v2/selection.lock.json').read_text(encoding='utf8'))
        write_json(sweep, folder/'validation_sweep.json')
        if sweep[0]['hit_at_10'] <= previous['validation_metrics']['hit_at_10']:
            write_json({'status': 'no_validation_improvement', 'best': sweep[0]}, folder/'selection_outcome.json')
            raise ValueError('Validation fallback triggered: trained checkpoint not promoted')
        best = sweep[0]; checkpoints = json.loads((folder/'completion.json').read_text(encoding='utf8'))['checkpoints']
        selected = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_on': 'valid',
            'epoch': best['epoch'], 'config': best['config'], 'validation_metrics': best,
            'selected_weights_sha256': checkpoints[best['epoch']-1]['weights_sha256'],
            'training_manifest_sha256': checkpoints[best['epoch']-1]['training_manifest_sha256'],
            'fixture_lock_sha256': sha(fixture/'manifest.lock.json'), 'validation_artifacts': sources,
            'selection_protocol_sha256': sha(plan_path), 'code_sha256': code_hashes(),
            'historical_test_status': 'Previously inspected in earlier project iterations; not a blind test',
            'external_transfer_status': 'Not scored before checkpoint selection'}
        lock = folder/'selection.lock.json'
        if lock.exists():
            raise ValueError('Selection lock already exists')
        write_json(selected, lock); print(json.dumps(selected, indent=2)); return
    selection_path = folder/'selection.lock.json'
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    if selection['code_sha256'] != code_hashes():
        raise ValueError('Frozen evaluation code changed')
    if selection['fixture_lock_sha256'] != sha(fixture/'manifest.lock.json'):
        raise ValueError('Fixture changed after selection')
    cases = read_jsonl(fixture/'test.jsonl')
    rows, protocol = verified_scores(folder/'test_selected', 'test')
    if protocol['selection_lock_sha256'] != sha(selection_path):
        raise ValueError('Test scorer used a different selection')
    original, _ = verified_scores(ROOT/'reports/insuranceqa_v2/rerank_test_v1', 'test')
    for a, b in zip(rows, original):
        for key in ['id', 'candidate_ids', 'dense', 'sparse', 'bge_ids', 'bm25_ids']:
            if a[key] != b[key]:
                raise ValueError('Training comparison changed retrieval candidates')
    previous_config = json.loads((ROOT/'reports/insuranceqa_v2/rerank_selection_v2/selection.lock.json').read_text(encoding='utf8'))['config']
    arms = {
        'trained_selected': predictions(rows, cases, selection['config'], 'trained_selected'),
        'previous_untrained': predictions(original, cases, previous_config, 'previous_untrained'),
        'untrained_same_fusion': predictions(original, cases, selection['config'], 'untrained_same_fusion'),
        'trained_bge_candidates': predictions(rows, cases, {**selection['config'], 'pool': 'bge100', 'lexical_weight': 0.}, 'trained_bge_candidates')}
    old = read_jsonl(ROOT/'reports/insuranceqa_v2/retrieval_frozen/predictions.jsonl')
    for arm in ['bge', 'bm25', 'rrf']:
        arms[arm] = [r for r in old if r['split'] == 'test' and r['scope'] == 'all_27413_answers' and r['arm'] == arm]
    summaries = {name: summarize(values, cases) for name, values in arms.items()}
    paired = {name: paired_summary(arms['trained_selected'], values, cases)
              for name, values in arms.items() if name != 'trained_selected'}
    audit_path = ROOT/'reports/insuranceqa_v2/retrieval_frozen/sensitivity_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf8')); excluded = {r['id'] for r in audit['near_duplicates']}
    sensitivity = {name: aggregate([r for r in values if r['id'] not in excluded]) for name, values in arms.items()}
    output = folder/'test_evaluation'; output.mkdir(exist_ok=False)
    with (output/'predictions.jsonl').open('w', encoding='utf8') as handle:
        for values in arms.values():
            for row in values: handle.write(json.dumps(row)+'\n')
    summary = {'status': 'completed', 'summaries': summaries, 'trained_minus_comparator': paired,
        'historical_near_duplicate_exclusion': sensitivity, 'sensitivity_audit_sha256': sha(audit_path),
        'selection_lock_sha256': sha(selection_path), 'test_scores_sha256': sha(folder/'test_selected/scores.jsonl'),
        'predictions_sha256': sha(output/'predictions.jsonl'), 'code_sha256': code_hashes(),
        'limitation': 'Historical test was inspected in previous iterations. Training loss is not a generalization metric.'}
    write_json(summary, output/'summary.json')
    print(json.dumps({'summaries': {n: {k: v[k] for k in ['n', 'hit_at_1', 'hit_at_10', 'mrr_at_100']} for n,v in summaries.items()}, 'paired': paired}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['plan', 'select', 'test'])
    p.add_argument('--run', type=Path, default=ROOT/'reports/domain_training_v1')
    run(p.parse_args())
