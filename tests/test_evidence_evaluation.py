import pytest
from src.insurerag_vlm.evidence_evaluation import (
    retrieval_metrics, packing_metrics, aggregate_by_publisher, abstention_metrics, citation_markers,
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
