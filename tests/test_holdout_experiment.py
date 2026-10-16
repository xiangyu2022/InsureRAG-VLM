import pytest
from src.insurerag_vlm.holdout_experiment import score_retrieval, paired_family_delta
from src.insurerag_vlm.evidence_evaluation import retrieval_metrics


def test_recall_is_not_hit_and_short_ranking_has_no_invented_evidence():
    result = score_retrieval(['a'], ['a', 'b'], answerable=True)
    assert result['recall_at_5'] == .5 and result['hit_at_5'] == 1
    assert result['complete_at_5'] == 0 and 0 < result['ndcg_at_10'] < 1
    assert score_retrieval([], ['a'], answerable=True)['recall_at_5'] == 0


def test_failure_stays_zero_and_refusal_is_not_scored_as_answer_retrieval():
    failed = score_retrieval(['a'], ['a'], answerable=True, failed=True)
    assert failed['recall_at_5'] == failed['ndcg_at_10'] == 0
    refusal = score_retrieval(['a'], ['a'], answerable=False)
    assert refusal['recall_at_5'] is None and refusal['complete_at_5'] is None
    with pytest.raises(ValueError): score_retrieval([], [], answerable=True)
    with pytest.raises(ValueError): score_retrieval(['a','a'], ['a'], answerable=True)


def test_full_depth_matches_existing_metric_contract():
    ranking = list('abcdefghijk')
    previous = retrieval_metrics(ranking, ['c','h'])
    current = score_retrieval(ranking, ['c','h'], answerable=True)
    assert all(current[k] == previous[k] for k in ('recall_at_5','hit_at_5','complete_at_5','ndcg_at_10'))


def records():
    baseline = [dict(id=str(i), document_family='A' if i < 3 else 'B', score=0.) for i in range(4)]
    candidate = [r | {'score':float(r['document_family']=='A')} for r in baseline]
    return baseline, candidate, [r['id'] for r in baseline]


def test_cluster_estimand_and_pairing_are_explicit():
    b,c,ids = records()
    result = paired_family_delta(b,c,ids,'score',draws=200)
    assert result['question_macro_delta'] == .75
    assert result['family_macro_delta'] == .5
    assert result['ci95'] == [0.,1.]
    assert result == paired_family_delta(list(reversed(b)),list(reversed(c)),list(reversed(ids)),'score',draws=200)
    with pytest.raises(ValueError): paired_family_delta(b,c[:-1],ids,'score')
    with pytest.raises(ValueError): paired_family_delta(b,c+[c[0]],ids,'score')
    c[0] = c[0] | {'document_family':'C'}
    with pytest.raises(ValueError): paired_family_delta(b,c,ids,'score')


def test_single_family_uncertainty_and_nonfinite_scores_are_not_fabricated():
    b,c,ids = records()
    one = paired_family_delta(b[:3],c[:3],ids[:3],'score',draws=20)
    assert one['ci95'] is None and 'ci_unavailable_reason' in one
    c[0] = c[0] | {'score':float('nan')}
    with pytest.raises(ValueError): paired_family_delta(b,c,ids,'score')


def test_aliases_do_not_multiply_units_and_split_spans_need_all_chunks():
    from src.insurerag_vlm.holdout_experiment import evidence_unit_metrics
    units = [dict(id='rule',alternative_chunk_sets=[['a','b'],['alias']]),
             dict(id='exception',alternative_chunk_sets=[['c']])]
    assert evidence_unit_metrics(['a','c'],units)['unit_recall'] == .5
    assert evidence_unit_metrics(['a','b','c'],units)['complete_evidence'] == 1
    alias = evidence_unit_metrics(['alias','a','b','c'],units)
    assert alias['retrieved_units'] == alias['necessary_units'] == 2
    assert evidence_unit_metrics(['alias','c'],units,failed=True)['complete_evidence'] == 0


def test_invalid_support_annotations_fail_before_any_score_is_reported():
    from src.insurerag_vlm.holdout_experiment import evidence_unit_metrics
    with pytest.raises(ValueError): evidence_unit_metrics([],[])
    with pytest.raises(ValueError): evidence_unit_metrics(['a'],[dict(id='rule',alternative_chunk_sets=[[]])])
    with pytest.raises(ValueError): evidence_unit_metrics(['a'],[dict(id='rule',alternative_chunk_sets=[['a'],['a']])])
