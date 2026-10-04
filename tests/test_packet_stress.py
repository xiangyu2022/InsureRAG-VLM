from pathlib import Path
from scripts.build_packet_stress import build_spec,materialize_pdfs
from scripts.eval_packet_stress import canonical_source,raw_sources,score_case,verify_fixture,validate_pdf_ingestion

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_synthetic_cases_are_reproducible_and_diverse():
    _,spec,cases=verify_fixture(ROOT/'data/benchmarks/packet_stress_v1')
    assert (spec,cases)==build_spec()
    assert len(cases)==24 and sum(c['answerable'] for c in cases)==16
    assert len(spec['packets'])==6
    assert len({c['category'] for c in cases})>=12


def test_actual_generated_pdf_ingestion_preserves_pages_and_provenance(tmp_path):
    spec,_=build_spec()
    hashes1=materialize_pdfs(spec,tmp_path/'first')
    hashes2=materialize_pdfs(spec,tmp_path/'second')
    assert hashes1==hashes2
    detail=validate_pdf_ingestion(spec,tmp_path/'first')
    assert sum(p['loaded_pages'] for p in detail.values())==13
    assert len(hashes1)==10


def test_sources_allow_directory_aliases_but_never_wrong_physical_pages():
    assert canonical_source(r'C:\temporary\SCENARIO-A.pdf#page=01')=='scenario-a.pdf#page=1'
    assert raw_sources('Source: SCENARIO-A.pdf#page=2.')==['scenario-a.pdf#page=2']
    assert canonical_source('SCENARIO-A.pdf#page=2')!=canonical_source('SCENARIO-A.pdf#page=1')


def supported_fixture():
    case={'answerable':True,'answer_keys':[['2500']],'gold_sources':['SCENARIO-A.pdf#page=1'],'forbidden_keys':[]}
    result={'raw_answer':'$2500. SOURCE: SCENARIO-A.pdf#page=1','answer':'$2500','abstain':False,
        'citations':[{'source':'SCENARIO-A.pdf#page=1'}],'generation_used':True,
        'source_ranking':[{'source':'SCENARIO-A.pdf#page=1','text_snippet':'The limit is $2500.'}],
        'backend_metadata':{'last_generation':{'done_reason':'stop','truncated':False}}}
    return case,result


def test_pretrained_correct_key_outside_actual_prompt_is_not_context_grounded():
    case,result=supported_fixture()
    scores=score_case(case,result,'Context:\nSOURCE: SCENARIO-B.pdf#page=1\nThe limit is $2500.\n\nQuestion: test')
    assert scores['raw']['supported_key_source_pass']
    assert not scores['raw']['context_key_source_pass']
    assert scores['served']['retrieved_evidence_key_source_pass']
    good='Context:\nSOURCE: SCENARIO-A.pdf#page=1\nThe limit is $2500.\n\nQuestion:\ntest'
    assert score_case(case,result,good)['raw']['context_key_source_pass']
    result['backend_metadata']['last_generation']['truncated']=True
    assert not score_case(case,result,good)['raw']['context_key_source_pass']


def test_repair_and_abstention_are_separate_from_raw_model_success():
    case,result=supported_fixture()
    result['raw_answer']='The limit is $25000. SOURCE: SCENARIO-A.pdf#page=1'
    scores=score_case(case,result)
    assert not scores['raw']['answer_key_match']
    assert scores['served']['answer_key_match']
    missing={'answerable':False,'answer_keys':[],'gold_sources':[],'forbidden_keys':[]}
    abstention={'raw_answer':'Insufficient evidence.','answer':'','abstain':True,'citations':[]}
    assert score_case(missing,abstention)['served']['empty_served_abstention']
    assert not score_case(missing,{})['served']['unsupported_abstention']
