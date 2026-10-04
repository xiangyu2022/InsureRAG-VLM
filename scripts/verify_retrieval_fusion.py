"""Verify frozen scores, code, selection, row coverage, and independent Hit@10 counts."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, verify_fixture, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    base = ROOT/'reports/insuranceqa_v2'; fixture = ROOT/'data/benchmarks/insuranceqa_v2'
    verify_fixture(fixture)
    selection_file = base/'rerank_selection_v2/selection.lock.json'
    selection = json.loads(selection_file.read_text())
    valid_protocol = json.loads((base/'rerank_valid_v2/protocol.json').read_text())
    require(selection['scoring_protocol_sha256'] == sha(base/'rerank_valid_v2/protocol.json'),
            'Selection does not bind this validation run')
    scores_count = {}
    for folder in ['rerank_valid_v2', 'rerank_test_v1']:
        run = base/folder
        protocol = json.loads((run/'protocol.json').read_text())
        done = json.loads((run/'completion.json').read_text())
        require(done['status'] == 'completed', 'Incomplete scoring run')
        require(sha(run/'protocol.json') == done['protocol_sha256'], 'Scoring protocol changed')
        require(sha(run/'scores.jsonl') == done['scores_sha256'], 'Raw scores changed')
        for path, expected in protocol['code_sha256'].items():
            require(sha(ROOT/path) == expected, 'Scoring code changed: '+path)
        if protocol['split'] == 'test':
            require(protocol['selection_lock_sha256'] == sha(selection_file), 'Test used another selection')
            require(protocol['reranker'] == valid_protocol['reranker'], 'Reranker changed after validation')
            require(protocol['embedding'] == valid_protocol['embedding'], 'BGE changed after validation')
            require(protocol['embedding_cache_sha256'] == valid_protocol['embedding_cache_sha256'],
                    'Embedding cache changed after validation')
        rows = read_jsonl(run/'scores.jsonl'); cases = read_jsonl(fixture/f'{protocol["split"]}.jsonl')
        require([r['id'] for r in rows] == [c['id'] for c in cases], 'Missing/duplicate/reordered score rows')
        for row in rows:
            require(len(row['candidate_ids']) == len(set(row['candidate_ids'])), 'Duplicate candidate')
            require(len(row['candidate_ids']) <= 200, 'Candidate budget exceeded')
            require(all(len(row[k]) == len(row['candidate_ids']) for k in ['dense', 'sparse', 'cross']), 'Misaligned scores')
            require(set(row['candidate_ids']) == set(row['bge_ids']) | set(row['bm25_ids']), 'Candidate injection or omission')
        require(sum(len(r['candidate_ids']) for r in rows) == done['pairs'], 'Pair count mismatch')
        scores_count[protocol['split']] = {'questions': len(rows), 'pairs': done['pairs']}
    eval_folder = base/'rerank_test_evaluation_v1'
    summary = json.loads((eval_folder/'summary.json').read_text())
    require(sha(eval_folder/'predictions.jsonl') == summary['predictions_sha256'], 'Predictions changed')
    for path, expected in selection['code_sha256'].items():
        require(sha(ROOT/path) == expected, 'Selection/evaluation code changed: '+path)
    gold = {c['id']: set(c['gold_answer_ids']) for c in read_jsonl(fixture/'test.jsonl')}
    predictions = read_jsonl(eval_folder/'predictions.jsonl'); recount = {}
    for arm, reported in summary['summaries'].items():
        rows = [r for r in predictions if r['arm'] == arm]
        require(len(rows) == len(gold) and {r['id'] for r in rows} == set(gold), 'Arm lacks complete test coverage')
        hits = sum(bool(set(r['top_answer_ids'][:10]) & gold[r['id']]) for r in rows)
        require(abs(hits/len(rows)-reported['hit_at_10']) < 1e-12, 'Hit@10 recount mismatch')
        first_hits = sum(bool(set(r['top_answer_ids'][:1]) & gold[r['id']]) for r in rows)
        reciprocal_sum = sum(next((1/(i+1) for i, id in enumerate(r['top_answer_ids'][:100])
                                   if id in gold[r['id']]), 0) for r in rows)
        require(abs(first_hits/len(rows)-reported['hit_at_1']) < 1e-12, 'Hit@1 recount mismatch')
        require(abs(reciprocal_sum/len(rows)-reported['mrr_at_100']) < 1e-12, 'MRR recount mismatch')
        recount[arm] = {'n': len(rows), 'hits_at_10': hits, 'hit_at_10': hits/len(rows),
                       'hit_at_1': first_hits/len(rows), 'mrr_at_100': reciprocal_sum/len(rows)}
    record = {'status': 'verified', 'scoring': scores_count, 'independent_test_hit10_recount': recount,
              'fixture_sha256': sha(fixture/'manifest.lock.json'), 'selection_sha256': sha(selection_file),
              'test_summary_sha256': sha(eval_folder/'summary.json')}
    if args.output:
        write_json(record, args.output)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    run(p.parse_args())
