import pytest
from src.insurerag_vlm.evidence_evaluation import (
    retrieval_metrics, packing_metrics, aggregate_by_publisher, abstention_metrics, citation_markers, publisher_cluster_delta,
)


def test_partial_multigold_is_hit_but_not_full_recall():
    result=retrieval_metrics(['a']+[str(i) for i in range(9)],['a','b'])
    assert result['hit_at_10']==1
    assert result['recall_at_10']==.5
    assert result['complete_at_10']==0
    assert result['ndcg_at_10']<1


def test_cutoff_cannot_be_reported_from_short_or_duplicate_rankings():
    with pytest.raises(ValueError): retrieval_metrics(['a']*10,['a'])
    with pytest.raises(ValueError): retrieval_metrics(['a'],['a'])
    with pytest.raises(ValueError): retrieval_metrics(list('abcdefghij'),[])


def test_retrieval_loss_and_packing_loss_are_separate():
    gold={'a':'Complete first evidence.','b':'Second full evidence.'}
    missing=packing_metrics(list('acdef'),gold,'Complete first evidence.')
    clipped=packing_metrics(list('abcde'),gold,'Complete first evidence. Second full')
    assert missing['retrieval_incomplete']==1
    assert missing['packing_lost_complete_evidence']==0
    assert clipped['retrieval_incomplete']==0
    assert clipped['packing_lost_complete_evidence']==1
    assert clipped['chunk_recall_after_packing']==.5


def test_context_text_alone_cannot_credit_an_unretrieved_gold_id():
    result=packing_metrics(list('abcde'),{'f':'same words'},'same words')
    assert result['complete_after_packing']==0


def test_publisher_macro_does_not_weight_larger_publisher_more():
    rows=[{'id':'a','publisher':'A','score':1},
          {'id':'b','publisher':'B','score':0},{'id':'c','publisher':'B','score':0}]
    result=aggregate_by_publisher(rows,['a','b','c'],['score'])
    assert result['publisher_macro']['score']==.5
    assert result['question_macro']['score']==pytest.approx(1/3)
    with pytest.raises(ValueError):aggregate_by_publisher(rows[:2],['a','b','c'],['score'])


def test_all_abstain_is_not_successful_answerable_coverage():
    result=abstention_metrics([{'answerable':True,'abstained':True},
                              {'answerable':False,'abstained':True}])
    assert result['abstention_recall']==1
    assert result['abstention_precision']==.5
    assert result['answerable_coverage']==0


def test_citation_parser_keeps_unknown_ids_and_handles_repeated_source_labels():
    result=citation_markers('A supported claim. SOURCE: known, SOURCE: invented\nSOURCE: third')
    assert result['ids']==['known','invented','third']
    assert result['sentinels']==[]


def test_no_source_sentinel_is_not_a_valid_citation_or_proof_of_refusal():
    result=citation_markers('An unsupported factual assertion. SOURCE: N/A')
    assert result['ids']==[] and result['sentinels']==['N/A']
    assert citation_markers('SOURCE: none, unknown')['ids']==['unknown']


def test_empty_source_field_does_not_swallow_the_next_paragraph_as_an_identifier():
    raw='SOURCE:\n\nAdditional explanatory prose.'
    assert citation_markers(raw)['ids']==[]
    assert citation_markers(raw,version=2)['ids']==['Additional explanatory prose']


def test_cluster_contrast_is_paired_and_equal_publisher_weighted():
    base=[{'id':'a','publisher':'A','score':0},{'id':'b','publisher':'B','score':1},{'id':'c','publisher':'B','score':1}]
    candidate=[dict(r,score=1) for r in base]
    result=publisher_cluster_delta(base,candidate,'score')
    assert result['publisher_macro_delta']==.5
    assert result['questions']==3 and result['publisher_groups']==2
    assert result['paired_publisher_cluster_bootstrap_95ci']==[0.,1.]
    with pytest.raises(ValueError):publisher_cluster_delta(base,candidate[:2],'score')


@pytest.mark.parametrize('gold',[None,[],[''],[' '],['a',' a'],['a','a'],'a'])
def test_empty_duplicate_or_ambiguous_gold_ids_are_rejected(gold):
    with pytest.raises(ValueError):retrieval_metrics(list('abcdefghij'),gold)


def test_opaque_ids_do_not_gain_guessed_numeric_alias_equivalence():
    result=retrieval_metrics(['01']+[str(i) for i in range(2,11)],['1'])
    assert result['recall_at_10']==0


@pytest.mark.parametrize('cutoffs',[(0,),(1,1),(1.5,),(True,),()])
def test_invalid_cutoffs_fail_explicitly(cutoffs):
    with pytest.raises(ValueError):retrieval_metrics(list('abcdefghij'),['a'],cutoffs)


def test_empty_gold_text_cannot_be_retained_vacuously():
    with pytest.raises(ValueError):packing_metrics(list('abcde'),{'a':'  '},'')
    with pytest.raises(ValueError):packing_metrics(['a']*5,{'a':'Evidence'},'Evidence')


def test_all_gold_chunks_are_required_even_when_two_of_three_are_present():
    result=retrieval_metrics(list('abcdefghij'),['a','b','missing'])
    assert result['hit_at_10']==1 and result['recall_at_10']==pytest.approx(2/3)
    assert result['complete_at_10']==0


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_values_cannot_enter_aggregate_reports(value):
    with pytest.raises(ValueError):aggregate_by_publisher([{'id':'a','publisher':'A','score':value}],['a'],['score'])


def test_refusal_metrics_preserve_undefined_denominators_and_reject_truthy_strings():
    no_refusals=abstention_metrics([{'answerable':True,'abstained':False}])
    assert no_refusals['abstention_precision'] is None and no_refusals['abstention_recall'] is None
    no_positives=abstention_metrics([{'answerable':False,'abstained':True}])
    assert no_positives['answerable_coverage'] is None and no_positives['abstention_precision']==1
    with pytest.raises(ValueError):abstention_metrics([{'answerable':'false','abstained':True}])


def test_one_cluster_cannot_produce_a_spuriously_precise_interval():
    rows=[{'id':'a','publisher':'A','score':1}]
    with pytest.raises(ValueError):publisher_cluster_delta(rows,rows,'score')
    with pytest.raises(ValueError):publisher_cluster_delta(rows,rows,'score',draws=0)


def test_failed_requests_remain_in_denominators_but_are_neither_answers_nor_successful_refusals():
    rows=[{'answerable':True,'abstained':False,'failed':True},
          {'answerable':False,'abstained':True,'failed':True}]
    result=abstention_metrics(rows)
    assert result['n']==2 and result['failed_outputs']==2
    assert result['answerable_coverage']==0 and result['abstention_recall']==0
    assert result['abstention_precision'] is None and result['abstentions']==0
