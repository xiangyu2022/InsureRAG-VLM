"""Historical recall failures: descriptive evidence, never training examples."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.eval_insuranceqa_scale import SparseBM25


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    out = ROOT / 'reports/query_adaptation_v1'
    out.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(ROOT / 'data/benchmarks/insuranceqa_v2/test.jsonl')
    answers = sorted(read_jsonl(ROOT / 'data/benchmarks/insuranceqa_v2/answers.jsonl'), key=lambda a: int(a['id']))
    lookup = {a['id']: i for i, a in enumerate(answers)}
    failures = json.loads((ROOT / 'reports/condition_listwise_v1/failure_cases.json').read_text(encoding='utf8'))
    missed = {c['id'] for c in failures if c['cohort'] == 'historical_insuranceqa' and c['stage'] == 'absent_from_union'}
    vectors = ROOT / 'reports/insuranceqa_v2/retrieval_frozen'
    dense = np.load(vectors / 'answer_embeddings.npy')
    query = np.load(vectors / 'query_embeddings.npy')[2000:]
    assert len(query) == len(cases)
    bm25 = SparseBM25([a['text'] for a in answers])
    matrix = bm25.matrix.tocsc()
    details = []
    for i, c in enumerate(cases):
        if c['id'] not in missed:
            continue
        d = dense @ query[i]
        q = bm25.vectorizer.transform([c['question']]).tocsr()
        b = np.asarray(matrix[:, q.indices] @ q.data).ravel()
        gold = [lookup[a] for a in c['gold_answer_ids']]
        # Stable ties use corpus order, identical to the evaluation ranker.
        def goldrank(values, positive_only=False):
            return min([1+int((values > values[j]).sum())+int((values[:j] == values[j]).sum())
                        for j in gold if not positive_only or values[j] > 0] or [len(answers)+1])
        details.append({'id': c['id'], 'question': c['question'], 'domain': c['domain'],
                        'best_gold_dense_rank': goldrank(d), 'best_gold_bm25_rank': goldrank(b, True),
                        'gold_answers': [answers[j] for j in gold],
                        'dense_top1': answers[int(np.argmax(d))]})
    assert len(details) == 194
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'historical_only': True,
                'used_for_training': False, 'cases': details,
                'domain_counts': dict(Counter(r['domain'] for r in details)),
                'missed_by_both_top100': len(details),
                'dense_rank_quantiles': np.quantile([r['best_gold_dense_rank'] for r in details], [0, .25, .5, .75, 1]).tolist(),
                'would_enter_expanded_top200_union': sum(min(r['best_gold_dense_rank'], r['best_gold_bm25_rank']) <= 200 for r in details),
                'would_enter_expanded_top500_union': sum(min(r['best_gold_dense_rank'], r['best_gold_bm25_rank']) <= 500 for r in details),
                'interpretation': 'Candidate absence can reflect representation mismatch, truncation, or incomplete labels; these diagnostics do not adjudicate correctness or justify test-specific query rules.',
                'source_failure_sha256': sha(ROOT / 'reports/condition_listwise_v1/failure_cases.json'),
                'code_sha256': sha(Path(__file__))}, out / 'recall_failure_diagnosis.json')
    print(json.dumps({'misses': len(details), 'dense_rank_median': float(np.median([r['best_gold_dense_rank'] for r in details]))}))


if __name__ == '__main__':
    main()
