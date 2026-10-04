from copy import deepcopy
import pytest
from src.insurerag_vlm.holdout_quality import digest,verify_item,verify_dataset

def fixture(task='ordinary_qa'):
    text='The synthetic policy requires a deductible of 100 dollars. A separate exclusion applies to flood.'
    document={'id':'doc','publisher':'p','jurisdiction':'TEST','source_title':'Synthetic fixture',
              'source_url':'https://example.invalid/synthetic','source_sha256':'a'*64,'acquired_utc':'2026-10-04',
              'text':text,'normalized_text_sha256':digest(text)}
    item={'id':'one','publisher':'p','jurisdiction':'TEST','insurance_type':'home','question':'What deductible does the synthetic policy require?',
          'answer':'100 dollars','document_group':'doc','origin':'synthetic_unit_test','task':task,
          'evidence':[{'document_id':'doc','start':0,'end':len(text),'sha256':digest(text)}],
          'contamination_audit':{'status':'clear_in_accessible_scope','report_sha256':'b'*64},
          'duplicate_audit':{'status':'unique_information_need'},'content_review':{'decision':'pass','reviewer_type':'codex_agent_content_review','rationale':'Synthetic fixture only.'}}
    return item,{'doc':document},{'p'}

def test_structural_pass_is_explicitly_separate_from_content_truth():
    item,docs,sources=fixture()
    assert verify_item(item,docs,sources)==[]

@pytest.mark.parametrize('mutation,expected',[
    (lambda r,d:r.update(insurance_type=None),'missing_insurance_type'),
    (lambda r,d:r.update(automatic_flags=['unresolved']),'unresolved_automatic_flags'),
    (lambda r,d:r.update(contamination_audit={}), 'missing_contamination_clearance'),
    (lambda r,d:r.update(duplicate_audit={}), 'missing_duplicate_clearance'),
    (lambda r,d:r['evidence'][0].update(start=-1),'invalid_evidence_offsets'),
    (lambda r,d:r['evidence'][0].update(sha256='wrong'),'evidence_hash_mismatch'),
    (lambda r,d:d['doc'].update(text='tampered'), 'document_text_hash_mismatch'),
    (lambda r,d:d['doc'].update(jurisdiction='OTHER'),'jurisdiction_mismatch'),
    (lambda r,d:r.update(content_review={'decision':'reject'}),'content_review_rejection_or_hold'),
])
def test_corrupt_or_unreviewed_candidate_cannot_pass(mutation,expected):
    item,docs,sources=fixture();mutation(item,docs)
    assert expected in verify_item(item,docs,sources)

def test_duplicate_citation_does_not_make_a_multi_evidence_question():
    item,docs,sources=fixture('multi_evidence');item['evidence']*=2
    item['evidence_roles']=['rule','same rule'];item['content_review']['each_evidence_necessary']=True
    assert 'duplicate_evidence' in verify_item(item,docs,sources)

def test_refusal_needs_real_missing_facts_and_nonempty_relevant_evidence():
    item,docs,sources=fixture('insufficient_evidence')
    errors=verify_item(item,docs,sources)
    assert {'unsubstantiated_information_gap','refusal_gap_unreviewed','empty_context_shortcut'}<=set(errors)
    item['information_gap']={'missing_facts':['insured cause of loss'],'why_required':'Flood is excluded, so cause determines coverage.',
      'available_evidence_not_sufficient':'The policy describes rules but no loss cause was supplied.','empty_context_only':False}
    item['content_review'].update(realistic_information_gap=True,not_answerable_from_corpus=True)
    assert verify_item(item,docs,sources)==[]

def test_calculation_requires_independently_checked_result_and_operand_provenance():
    item,docs,sources=fixture('numerical_calculation')
    item['calculation']={'formula':'200 - 100','unit':'USD','rounding':'exact integer',
      'independent_verification':'Decimal subtraction in synthetic unit test','operands':[{'name':'loss','provenance':'explicit hypothetical input'}],
      'result':'100','verified_result':'99'}
    assert 'calculation_result_mismatch' in verify_item(item,docs,sources)
    item['calculation']['verified_result']='100'
    assert verify_item(item,docs,sources)==[]

def test_subthreshold_or_cross_split_rows_never_report_accepted_thousand():
    item,docs,sources=fixture();other=deepcopy(item);other['id']='dev'
    protocol={'target_test_accepted':1000,'target_dev_accepted':150,'source_targets':{'minimum_independent_test_publishers':8,
      'maximum_publisher_fraction':.2,'minimum_test_document_groups':100,'maximum_questions_per_document_group':20},
      'test_task_targets':{'ordinary_qa':600,'numerical_calculation':150,'multi_evidence':150,'insufficient_evidence':100}}
    report=verify_dataset([item],[other],docs,sources,protocol)
    assert report['accepted_test_items']==0 and not report['passed']
    assert {'test_count_below_target','cross_split_publisher','cross_split_document_group','cross_split_evidence_document','cross_split_question_duplicate'}<=set(report['errors'])
