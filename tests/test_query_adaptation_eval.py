import math
import pytest
pytest.importorskip('sklearn', reason='Optional retrieval evaluation dependency')
from scripts.eval_query_adaptation import configurations, measure, partitions


def test_retrieval_metrics_respect_multiple_original_labels():
    result = measure(['x', 'a', 'b'], {'a', 'b'}, ['x', 'a', 'b'])
    expected = (1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
    assert math.isclose(result['ndcg_at_10'], expected)
    assert result['hit_at_1'] == 0 and result['hit_at_10'] == 1
    assert result['label_recall_at_10'] == 1
    assert result['mrr_at_100'] == .5 and result['candidate_hit'] == 1


def test_missing_positive_is_not_injected_by_metrics():
    order = ['x', 'y']
    result = measure(order, {'z'}, order)
    assert result['candidate_hit'] == 0 and result['hit_at_10'] == 0 and result['ndcg_at_10'] == 0
    assert order == ['x', 'y']


def test_registered_model_alpha_grid_and_domain_partition_are_complete():
    configs = configurations({'seeds': [42, 123], 'epochs_considered': [1, 2], 'query_blend_alphas': [.5, 1.]})
    assert len(configs) == len({c['name'] for c in configs}) == 9
    assert configs[0]['name'] == 'original'
    assert {c['alpha'] for c in configs} == {0., .5, 1.}
    rows = [{'domain': 'government'}, {'domain': 'general'}, {'domain': 'government'}]
    assert partitions('multidomain', rows) == {'government': [0, 2], 'general': [1]}
