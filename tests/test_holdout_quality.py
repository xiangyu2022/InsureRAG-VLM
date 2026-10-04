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
    (lambda r,d:r.update(content_review={'decision':'hold'}),'content_review_rejection_or_hold'),
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
    item['calculation']={'formula':'loss - deductible','unit':'USD','rounding':'exact integer',
      'independent_verification':'Decimal subtraction in synthetic unit test','operands':[{'name':'loss','value':'200','provenance':'explicit hypothetical input'},{'name':'deductible','value':'100','provenance':'synthetic source clause'}],
      'result':'100','verified_result':'99'}
    assert 'calculation_result_mismatch' in verify_item(item,docs,sources)
    item['calculation']['verified_result']='100'
    assert 'calculation_retrieval_necessity_unreviewed' in verify_item(item,docs,sources)
    item['calculation']['retrieval_required_rule']='The deductible must be retrieved from the synthetic policy.'
    item['content_review']['numeric_retrieval_necessary']=True
    assert verify_item(item,docs,sources)==[]
    item['calculation']['operands'][1]['value']='50'
    assert 'calculation_formula_mismatch' in verify_item(item,docs,sources)

def test_matching_declared_numbers_do_not_hide_wrong_formula():
    from src.insurerag_vlm.holdout_quality import evaluate_calculation
    assert evaluate_calculation('loss - deductible',[{'name':'loss','value':'200'},{'name':'deductible','value':'50'}])==150
    with pytest.raises(ValueError):evaluate_calculation('__import__("os").getcwd()',[])
    with pytest.raises(ValueError):evaluate_calculation('x',[{'name':'x','value':'NaN'}])
    with pytest.raises(ValueError):evaluate_calculation('100',[{'name':'ignored','value':'200'}])
    assert evaluate_calculation('0.1 + 0.2',[])==__import__('decimal').Decimal('0.3')

def test_non_decimal_step_cannot_silently_use_decimal_places_rounding():
    item,docs,sources=fixture('numerical_calculation')
    item['calculation']={'formula':'x','operands':[{'name':'x','value':'1.03','provenance':'synthetic input'}],
      'result':'1.03','verified_result':'1.03','unit':'USD','rounding':'nearest five cents',
      'rounding_quantum':'0.05','rounding_mode':'ROUND_HALF_UP','independent_verification':'synthetic fixture'}
    assert 'invalid_calculation_result' in verify_item(item,docs,sources)

def test_reviewing_one_document_does_not_clear_an_unreviewed_document():
    item,docs,sources=fixture();docs['unreviewed_doc']={**docs['doc'],'id':'unreviewed_doc'};rows=[]
    for i in range(8):
        row=deepcopy(item);row['id']=str(i)
        if i>=4:
            row['document_group']='unreviewed_doc';row['evidence'][0]['document_id']='unreviewed_doc';row['content_review']=None
        rows.append(row)
    protocol={'target_test_accepted':1,'target_dev_accepted':0,'source_targets':{'minimum_independent_test_publishers':1,
      'maximum_publisher_fraction':1,'minimum_test_document_groups':1,'maximum_questions_per_document_group':20},'test_task_targets':{}}
    result=verify_dataset(rows,[],docs,sources,protocol)
    assert any('insufficient_ordinary_review:' in e and 'unreviewed_doc' in e for e in result['errors'])

def test_subthreshold_or_cross_split_rows_never_report_accepted_thousand():
    item,docs,sources=fixture();other=deepcopy(item);other['id']='dev'
    protocol={'target_test_accepted':1000,'target_dev_accepted':150,'source_targets':{'minimum_independent_test_publishers':8,
      'maximum_publisher_fraction':.2,'minimum_test_document_groups':100,'maximum_questions_per_document_group':20},
      'test_task_targets':{'ordinary_qa':600,'numerical_calculation':150,'multi_evidence':150,'insufficient_evidence':100}}
    report=verify_dataset([item],[other],docs,sources,protocol)
    assert report['accepted_test_items']==0 and not report['passed']
    assert {'test_count_below_target','cross_split_publisher','cross_split_document_group','cross_split_evidence_document','cross_split_question_duplicate'}<=set(report['errors'])


def test_dev_size_does_not_replace_document_and_task_coverage():
    item,docs,sources=fixture()
    dev=[{**deepcopy(item),'id':str(i)} for i in range(50)]
    protocol={'target_test_accepted':0,'target_dev_accepted':50,'source_targets':{'minimum_independent_test_publishers':0,
      'maximum_publisher_fraction':1,'minimum_test_document_groups':0,'maximum_questions_per_document_group':20},
      'test_task_targets':{},'dev_source_targets':{'minimum_document_groups':10,'maximum_questions_per_document_group':20},
      'dev_task_minimums':{'ordinary_qa':10,'numerical_calculation':10,'multi_evidence':10,'insufficient_evidence':10}}
    report=verify_dataset([],dev,docs,sources,protocol)
    assert 'dev_count_below_target' not in report['errors']
    assert 'too_few_dev_document_groups' in report['errors']
    assert 'dev_document_overrepresentation' in report['errors']
    assert 'dev_task_count_below_target:numerical_calculation' in report['errors']
    assert 'dev_task_count_below_target:ordinary_qa' not in report['errors']


def test_alias_documents_do_not_hide_family_leakage_or_inflate_coverage():
    item,docs,sources=fixture();item['document_family']='shared'
    second=deepcopy(item);second.update(id='second',document_group='alias')
    dev=deepcopy(item);dev.update(id='dev',publisher='other',document_group='different',question='A different synthetic question?')
    dev['evidence']=[]
    protocol={'target_test_accepted':0,'target_dev_accepted':0,'source_targets':{'minimum_independent_test_publishers':0,
      'maximum_publisher_fraction':1,'minimum_test_document_groups':2,'maximum_questions_per_document_group':20},'test_task_targets':{}}
    report=verify_dataset([item,second],[dev],docs,sources,protocol)
    assert 'cross_split_document_family' in report['errors']
    assert 'cross_split_document_group' not in report['errors']
    assert 'too_few_document_groups' in report['errors']
    assert report['document_groups']==1


def test_final_registry_gates_cannot_be_bypassed_with_missing_or_forged_item_labels():
    item,docs,sources=fixture()
    protocol={'target_test_accepted':1,'target_dev_accepted':0,'source_targets':{'minimum_independent_test_publishers':1,
      'maximum_publisher_fraction':1,'minimum_test_document_groups':1,'maximum_questions_per_document_group':20},
      'test_task_targets':{},'require_source_review_registries':True}
    families={'groups':[],'document_text_sha256':{}}
    assert 'missing_source_review_registries' in verify_dataset([item],[],docs,sources,protocol)['errors']
    result=verify_dataset([item],[],docs,sources,protocol,document_holds={'documents':{}},document_families=families)
    assert result['passed']
    blocked=verify_dataset([item],[],docs,sources,protocol,document_holds={'documents':{'doc':{'reason':'Synthetic unresolved source defect'}}},document_families=families)
    assert 'held_evidence_document' in blocked['item_errors']['test:one']
    item['document_family']='invented'
    result=verify_dataset([item],[],docs,sources,protocol,document_holds={'documents':{}},document_families=families)
    assert 'document_family_assignment_mismatch' in result['item_errors']['test:one']
    assert result['accepted_test_items']==0


def test_secondary_evidence_cannot_hide_a_held_source_or_alias_family():
    from src.insurerag_vlm.holdout_quality import reviewed_families
    item,docs,sources=fixture();docs['alias']={**docs['doc'],'id':'alias'}
    item['evidence']=[{**item['evidence'][0],'document_id':'alias'}]
    assert 'held_evidence_document' in verify_item(item,docs,sources,{'alias'})
    assert 'document_group_not_cited' in verify_item(item,docs,sources)
    mapping=reviewed_families(docs,{'groups':[],'document_text_sha256':{}})
    assert mapping['doc']==mapping['alias']
    with pytest.raises(ValueError):reviewed_families(docs,{'groups':[{'document_ids':['doc','alias']}],'document_text_sha256':{'doc':'wrong','alias':'wrong'}})
