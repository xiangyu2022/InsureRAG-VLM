import json
import numpy as np
import pytest

from src.insurerag_vlm.holdout_dev_runner import fingerprint
from src.insurerag_vlm.holdout_retrieval import ARMS, rank_arm, rank_six_arms
from src.insurerag_vlm.holdout_standalone import read_records, run_requests


def test_independent_arms_equal_shared_ranks_with_distinct_global_candidate_pools():
    ids = [f'global-{i:03}' for i in range(320)]
    original = np.arange(320, dtype=float)
    adapted = original[::-1]
    sparse = np.zeros(320); sparse[110:210] = 1
    cross = lambda selected: [((int(i[-3:])*37)%101)/101 for i in selected]
    shared = rank_six_arms(ids, original, adapted, sparse, cross)
    for arm in ARMS:
        calls = []
        def scorer(selected):
            calls.append(selected)
            return cross(selected)
        solo = rank_arm(arm, ids, dense=adapted if arm.endswith('_adapted') else original,
                        sparse=sparse, cross_score=scorer)
        assert solo['order'] == shared['orders'][arm]
        assert bool(calls) == arm.startswith('rerank_')
        assert solo['cross_pairs'] <= 200


def test_unused_dense_sparse_and_cross_channels_are_not_required():
    assert rank_arm('bm25', ['a','b'], sparse=[0,1])['order'] == ['b']
    assert rank_arm('dense_original', ['b','a'], dense=[1,1])['order'] == ['a','b']
    with pytest.raises(ValueError): rank_arm('bm25', [[]], sparse=[1])
    with pytest.raises(ValueError): rank_arm('bm25', ['a'], sparse=[-1])
    with pytest.raises(ValueError): rank_arm('rerank_original', ['a'], dense=[1], sparse=[1])


def test_failure_is_local_to_one_arm_and_exact_resume_never_retries(tmp_path):
    requests = [dict(id='q', question='Invented question?')]
    calls, syncs = [], []
    def infer(arm, request):
        calls.append(arm)
        assert set(request) == {'id','question'}
        if arm == 'hybrid_adapted': raise RuntimeError('Synthetic backend failure')
        return dict(order=['global'])
    summary = run_requests(requests, ['global'], infer, tmp_path, {}, synchronize=lambda:syncs.append(1))
    assert calls == list(ARMS) and len(syncs) == 11
    assert summary['arms']['hybrid_adapted']['failed'] == 1
    assert sum(a['successful'] for a in summary['arms'].values()) == 5
    def forbidden(*_): raise AssertionError('No repeated backend work')
    resumed = run_requests(requests, ['global'], forbidden, tmp_path, {})
    assert resumed['reused_arm_requests'] == 6
    _, records = read_records(requests, ['global'], tmp_path, {})
    assert all(r['elapsed_seconds'] >= 0 for r in records)
    with pytest.raises(ValueError, match='differ'):
        run_requests(requests, ['global'], forbidden, tmp_path, {'changed':True})


def test_interrupt_resumes_missing_arms_only(tmp_path):
    requests = [dict(id='q', question='Invented?')]
    calls = []
    def interrupted(arm, request):
        calls.append(arm)
        if arm == ARMS[2]: raise KeyboardInterrupt()
        return dict(order=['c'])
    with pytest.raises(KeyboardInterrupt): run_requests(requests, ['c'], interrupted, tmp_path, {})
    with pytest.raises(ValueError, match='Incomplete'): read_records(requests, ['c'], tmp_path, {})
    def finish(arm, request):
        calls.append(arm)
        return dict(order=['c'])
    summary = run_requests(requests, ['c'], finish, tmp_path, {})
    assert summary['reused_arm_requests'] == 2
    assert calls == list(ARMS[:3])+list(ARMS[2:])


@pytest.mark.parametrize('result', [dict(order=['outside']), dict(order=['c','c']),
                                  dict(order=['c'],trace={'bad':float('nan')})])
def test_invalid_backend_output_is_preserved_as_failure(tmp_path, result):
    requests = [dict(id='q',question='Invented?')]
    summary = run_requests(requests, ['c'], lambda *_:result, tmp_path, {})
    assert all(a['failed'] == 1 for a in summary['arms'].values())
    _, records = read_records(requests,['c'],tmp_path,{})
    assert all(r['order'] == [] for r in records)


@pytest.mark.parametrize('rehash', [False,True])
def test_tampered_or_negative_saved_time_cannot_resume(tmp_path, rehash):
    requests = [dict(id='q',question='Invented?')]
    run_requests(requests,['c'],lambda *_:dict(order=['c']),tmp_path,{})
    path = tmp_path/ARMS[0]/(fingerprint(requests[0])+'.json')
    row = json.loads(path.read_text()); row['elapsed_seconds'] = -1
    if rehash:
        row.pop('record_sha256'); row['record_sha256'] = fingerprint(row)
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError): read_records(requests,['c'],tmp_path,{})
