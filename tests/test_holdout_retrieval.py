import numpy as np
import pytest
from src.insurerag_vlm.holdout_retrieval import ARMS, rank_six_arms


def test_distinct_union_pools_are_bounded_and_shared_cross_work_reused():
    ids = [f'c{i:03d}' for i in range(320)]
    d = np.arange(320, dtype=float)
    a = -d
    b = np.zeros(320); b[110:210] = 1
    seen = []
    def cross(selected):
        seen.append(selected)
        return [float(int(s[1:]) % 17) for s in selected]
    result = rank_six_arms(ids, d, a, b, cross)
    assert set(result['orders']) == set(ARMS)
    pools = result['candidate_scores']
    assert len(pools['original']['candidate_ids']) == 200
    assert len(pools['adapted']['candidate_ids']) == 200
    assert len(seen) == 1 and len(seen[0]) == 300
    assert set(result['orders']['rerank_original']) <= set(pools['original']['candidate_ids'])
    assert set(result['orders']['rerank_adapted']) <= set(pools['adapted']['candidate_ids'])


def test_zero_lexical_matches_stay_empty_and_ties_use_ids_not_input_order():
    ids=['c','a','b']; scores=[0,0,0]
    result=rank_six_arms(ids,scores,scores,scores,lambda x:[0]*len(x))
    assert result['orders']['bm25']==[]
    assert all(order==['a','b','c'] for arm,order in result['orders'].items() if arm!='bm25')
    assert result==rank_six_arms(list(reversed(ids)),scores,scores,scores,lambda x:[0]*len(x))


def test_nonfinite_missing_or_duplicate_global_scores_fail():
    for ids,d,a,b in [(['x','x'],[1,2],[1,2],[1,2]),
                       (['x','y'],[1],[1,2],[1,2]),
                       (['x','y'],[1,np.nan],[1,2],[1,2]),
                       (['x','y'],[1,2],[1,2],[-1,0])]:
        with pytest.raises(ValueError): rank_six_arms(ids,d,a,b,lambda x:[0]*len(x))
    with pytest.raises(ValueError):
        rank_six_arms(['x','y'],[1,2],[2,1],[0,0],lambda x:[np.inf]*len(x))
