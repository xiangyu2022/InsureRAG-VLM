import copy
import pytest
from src.insurerag_vlm.source_scope_audit import audit_source_scope


def test_mixed_scopes_warn_without_mutating_or_filtering_sources():
    sources=[{'source':'a','jurisdiction':'IE','publisher':'Authority A'},
             {'source':'b','jurisdiction':'US','publisher':'Authority B'}]
    original=copy.deepcopy(sources)
    result=audit_source_scope(sources,'IE')
    assert result['mismatched_source_ids']==['b']
    assert 'multiple_declared_jurisdictions' in result['flags']
    assert result['serving_action']=='none' and sources==original


def test_missing_metadata_is_not_guessed_from_program_name_or_url():
    sources=[{'source':'a','text':'Medicare Part B','source_url':'https://example.gov/coverage'}]
    result=audit_source_scope(sources)
    assert result['declared_jurisdictions']==[]
    assert result['unknown_jurisdiction_source_ids']==['a']
    assert 'requested_jurisdiction_not_explicit' in result['flags']


def test_same_scope_never_establishes_entailment_or_policy_applicability():
    result=audit_source_scope([{'source':'a','jurisdiction':' ie ','publisher':'Authority A'}],'IE')
    assert not result['flags']
    assert result['semantic_support']=='not_assessed' and result['serving_action']=='none'


def test_scope_granularity_is_not_silently_assumed_compatible():
    result=audit_source_scope([{'source':'a','jurisdiction':'CA'}],'CA-ON')
    assert result['mismatched_source_ids']==['a']


def test_empty_duplicate_or_missing_source_ids_cannot_look_verified():
    assert 'no_sources' in audit_source_scope([],'IE')['flags']
    with pytest.raises(ValueError):audit_source_scope([{'jurisdiction':'IE'}])
    with pytest.raises(ValueError):audit_source_scope([{'source':'a'},{'source':'a'}])


def test_empty_metadata_remains_unknown_without_triggering_a_serving_decision():
    result=audit_source_scope([{'source':'a','jurisdiction':' ','publisher':None}])
    assert result['unknown_jurisdiction_source_ids']==['a']
    assert result['unknown_publisher_source_ids']==['a']
    assert 'publisher_metadata_missing' in result['flags']
    assert result['serving_action']=='none' and result['semantic_support']=='not_assessed'


@pytest.mark.parametrize('source',[{'source':' '},{'source':' a'}, {'source':17},
    {'source':'a','jurisdiction':{'country':'IE'}},{'source':'a','publisher':['Authority A']}])
def test_ill_typed_metadata_cannot_be_coerced_into_apparently_valid_labels(source):
    with pytest.raises(ValueError):audit_source_scope([source])
