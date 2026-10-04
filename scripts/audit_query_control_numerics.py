"""Explain rank-list drift without claiming bitwise-identical historical inference."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.eval_query_adaptation import specs, CFG
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.reranker import blend_scores


if __name__ == '__main__':
    result = {}
    for cohort, _, _, priorfolder in specs('test'):
        if not priorfolder: continue
        old = {r['id']: r for r in read_jsonl(ROOT / 'reports/condition_listwise_v1' / priorfolder / 'scores.jsonl')}
        counts = Counter(); maxima = Counter(); example = []
        for row in read_jsonl(ROOT / 'reports/query_adaptation_v1/test' / f'{cohort}_scores.jsonl'):
            a = row['arms']['original']; b = old[row['id']]
            cross = dict(zip(row['candidate_ids'], row['cross']))
            a = {**a, 'cross': [cross[x] for x in a['candidate_ids']]}
            counts['questions'] += 1
            counts['candidate_set_changed'] += int(set(a['candidate_ids']) != set(b['candidate_ids']))
            ad, bd = {}, {}
            for field in ['dense', 'sparse', 'cross']:
                ad[field] = dict(zip(a['candidate_ids'], a[field])); bd[field] = dict(zip(b['candidate_ids'], b[field]))
                common = set(ad[field]) & set(bd[field])
                maxima[field+'_max_absolute_change'] = max(maxima[field+'_max_absolute_change'],
                                                          max(abs(ad[field][x]-bd[field][x]) for x in common))
            ao, bo = rank_row(a, CFG), rank_row(b, CFG)
            if ao == bo: continue
            counts['top100_changed'] += 1
            if set(a['candidate_ids']) == set(b['candidate_ids']):
                # Maximum old score gap among pairs whose relative order changed.
                values = blend_scores(np.array(b['dense']), np.array(b['sparse']), np.array(b['cross']), .2, .5)
                scores = dict(zip(b['candidate_ids'], values))
                position = {aid: i for i, aid in enumerate(bo)}
                gap = max([abs(scores[x]-scores[y]) for i, x in enumerate(ao) for y in ao[i+1:]
                           if x in position and y in position and position[x] > position[y]] or [0.])
                maxima['largest_inverted_old_blend_score_gap'] = max(maxima['largest_inverted_old_blend_score_gap'], gap)
                if len(example) < 3: example.append({'id': row['id'], 'largest_inverted_old_score_gap': float(gap)})
        assert maxima['cross_max_absolute_change'] == 0
        result[cohort] = {**dict(counts), **dict(maxima), 'examples': example}
    write_json({'cohorts': result, 'reranker_scores_for_shared_candidates_exactly_reused': True,
                'interpretation': 'Original-query embeddings/matrix multiplication are recomputed; small dense-score changes can reorder near ties. Historical per-question retrieval metrics are verified separately, not assumed from top-list equality.',
                'code_sha256': sha(Path(__file__))}, ROOT / 'reports/query_adaptation_v1/control_numerics.json')
    print(json.dumps(result))
