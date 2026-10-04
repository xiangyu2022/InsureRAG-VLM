import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.eval_research_benchmark import (
    answer_key_match,key_present,verify_manifest,scoped_records,score_response,wilson,
    LocalOllamaClient,context_from_prompt,summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_annotations_and_document_disjoint_questions():
    lock,rows,pages = verify_manifest(ROOT/'data/benchmarks/research_v1',ROOT/'data/04_curated')
    assert len(rows)==36
    assert sum(row['answerable'] for row in rows)==24
    assert len([row for row in rows if row['split']=='dev'])==12
    assert len([row for row in rows if row['split']=='test'])==24
    assert all('archived' in row['question'] for row in rows)
    assert lock['frozen_before_model_inference']


def test_currency_boundaries_and_decimal_equivalence():
    assert key_present('The amount is $2,500.', '2500')
    assert key_present('The amount is $2,500.00.', '2500')
    assert not key_present('The amount is $25,000.', '2500')
    assert not key_present('The amount is $2,500.50.', '2500')
    assert not key_present('The amount is 12500.', '2500')
    assert not key_present('The rate is 12%.', '2%')
    assert answer_key_match('HO-4',[['ho-4']])


def test_scope_keeps_distractor_pages_without_reading_gold():
    pages=[{'doc_id':'A','page':1},{'doc_id':'A','page':2},{'doc_id':'B','page':3}]
    snippets=[{'doc_id':'A','text':'gold'},{'doc_id':'A','text':'distractor'},{'doc_id':'B','text':'other'}]
    selected,chosen=scoped_records(pages,snippets,['A'])
    assert [row['page'] for row in selected]==[1,2]
    assert [row['text'] for row in chosen]==['gold','distractor']


def test_a_correct_key_with_wrong_source_or_fabricated_quote_is_not_grounded():
    item={'answerable':True,'answer_keys':[['2500']], 'gold':[{'source':'policy.pdf#page=2','evidence_span':'The limit is $2,500.'}]}
    pages={'policy.pdf#page=2':{'text':'The limit is $2,500.'}}
    valid={'answer':'$2,500','source':'policy.pdf#page=2','evidence':'The limit is $2,500.','abstain':False}
    assert score_response(json.dumps(valid),item,pages,['policy.pdf#page=2'])['grounded_key_pass']
    wrong=dict(valid,source='policy.pdf#page=1')
    scores=score_response(json.dumps(wrong),item,pages,['policy.pdf#page=2'])
    assert scores['answer_key_match'] and not scores['grounded_key_pass']
    fabricated=dict(valid,evidence='The approved limit is $2,500.')
    assert not score_response(json.dumps(fabricated),item,pages,[])['grounded_key_pass']


def test_unsupported_must_abstain_without_supplying_an_answer():
    item={'answerable':False,'answer_keys':[],'gold':[]}
    valid={'answer':'','source':'','evidence':'','abstain':True}
    assert score_response(json.dumps(valid),item,{},[])['strict_abstention']
    invalid=dict(valid,answer='Probably $500')
    assert not score_response(json.dumps(invalid),item,{},[])['strict_abstention']
    assert not score_response('malformed',item,{},[])['valid_json']
    assert not score_response(json.dumps(dict(valid,extra='unrequested')),item,{},[])['valid_json']


@pytest.mark.parametrize('context,passes',[
    ('SOURCE: policy.pdf#page=2\nThe limit is $2,500.',True),
    ('SOURCE: policy.pdf#page=2\nThe limit\n is $2,500.',True),
    ('SOURCE: policy.pdf#page=1\nThe limit is $2,500.',False),
    ('SOURCE: policy.pdf#page=2\nAn unrelated sentence.',False),
    ('SOURCE: policy.pdf#page=2\nThe limit is $2,',False),
    ('SOURCE: policy.pdf#page=2\nNo amount.\n---\nSOURCE: policy.pdf#page=1\nThe limit is $2,500.',False),
    (None,False),
])
def test_context_audit_requires_full_quote_under_the_actual_supplied_source(context,passes):
    item={'answerable':True,'answer_keys':[['2500']], 'gold':[{'source':'policy.pdf#page=2','evidence_span':'The limit is $2,500.'}]}
    pages={'policy.pdf#page=2':{'text':'The limit is $2,500.'}}
    raw=json.dumps({'answer':'$2,500','source':'policy.pdf#page=2','evidence':'The limit is $2,500.','abstain':False})
    result=score_response(raw,item,pages,['policy.pdf#page=2'],observed_context=context,generation_complete=True)
    assert result['grounded_key_pass']  # Historical metric remains comparable.
    assert result['context_grounded_key_pass'] is passes
    assert not score_response(raw,item,pages,[],observed_context=context,generation_complete=False)['context_grounded_key_pass']


def test_prompt_extraction_excludes_question_and_completion_is_required_for_new_abstention_metric():
    prompt='Context:\nSOURCE: policy.pdf#page=1\nNo amount.\n\nQuestion:\nSOURCE: policy.pdf#page=2\nThe limit is $2,500.\n\nAnswer:'
    assert context_from_prompt(prompt)=='SOURCE: policy.pdf#page=1\nNo amount.'
    assert context_from_prompt('Unknown prompt contract') is None
    item={'answerable':False,'answer_keys':[],'gold':[]}
    raw=json.dumps({'answer':'','source':'','evidence':'','abstain':True})
    assert score_response(raw,item,{},[])['strict_abstention']
    assert not score_response(raw,item,{},[])['completed_strict_abstention']
    assert score_response(raw,item,{},[],generation_complete=True)['completed_strict_abstention']


def test_historical_rescore_preserves_raw_records_and_fails_missing_or_truncated_context_closed():
    from scripts.rescore_research_benchmark import rescore_rows
    item={'id':'case','split':'dev','question':'Limit?','document_scope':['policy'],'answerable':True,
          'answer_keys':[['2500']],'gold':[{'source':'policy.pdf#page=2','evidence_span':'The limit is $2,500.'}]}
    raw=json.dumps({'answer':'$2,500','source':'policy.pdf#page=2','evidence':'The limit is $2,500.','abstain':False})
    row={key:item[key] for key in ['id','split','question','document_scope','answerable']}
    row.update({'mode':'retrieved','raw_response':raw,'scores':{'grounded_key_pass':True},
                'generation':{'done':True,'done_reason':'stop'},'ranked_sources':['policy.pdf#page=2']})
    pages=[{'citation':'policy.pdf#page=2','text':'The limit is $2,500.'}]
    before=json.dumps(row,sort_keys=True)
    audited=rescore_rows([row],[item],pages)[0]
    assert json.dumps(row,sort_keys=True)==before
    assert audited['raw_response']==raw and audited['previous_scores']==row['scores']
    assert not audited['scores']['context_grounded_key_pass']
    row['prompt']='Context:\nSOURCE: policy.pdf#page=2\nThe limit is $2,500.\n\nQuestion:\nLimit?\n\nAnswer:'
    assert rescore_rows([row],[item],pages)[0]['scores']['context_grounded_key_pass']
    row['generation']['done_reason']='length'
    assert not rescore_rows([row],[item],pages)[0]['scores']['context_grounded_key_pass']


def test_wilson_handles_small_samples_and_empty_denominator():
    assert wilson(0,0)['rate'] is None
    assert wilson(4,4)['ci95_low']<0.6
    assert wilson(0,4)['ci95_high']>0.4


def test_summary_cannot_silently_label_historical_scores_as_v2():
    with pytest.raises(ValueError,match='mixed scores'):
        summarize([{'scores':{'grounded_key_pass':True}}])


@pytest.mark.parametrize('change', ['wrong_model','incomplete','empty','digest_changed'])
def test_local_runtime_rejects_unreliable_response_identity(monkeypatch,change):
    class Reply:
        def __init__(self,value): self.value=value
        def raise_for_status(self): pass
        def json(self): return self.value
    state={'generated':False}
    def get(url,**kwargs):
        if url.endswith('/api/version'):return Reply({'version':'test'})
        digest='changed' if change=='digest_changed' and state['generated'] else 'fixed'
        return Reply({'models':[{'name':'qwen3.5:4b','digest':digest}]})
    def post(url,**kwargs):
        state['generated']=True
        options=kwargs['json']['options']
        assert options['top_k']==40 and options['top_p']==1
        assert options['repeat_penalty']==1 and options['presence_penalty']==0
        return Reply({'model':'other:tag' if change=='wrong_model' else 'qwen3.5:4b',
            'done':change!='incomplete','message':{'content':'' if change=='empty' else '{}'}})
    monkeypatch.setattr('scripts.eval_research_benchmark.requests.get',get)
    monkeypatch.setattr('scripts.eval_research_benchmark.requests.post',post)
    client=LocalOllamaClient(SimpleNamespace(endpoint='http://localhost:11435',model='qwen3.5:4b',
        timeout=600,num_ctx=8192,num_predict=192))
    with pytest.raises(RuntimeError):client.generate('context')
