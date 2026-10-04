"""A comparison must reject silently mixed questions or experimental protocols."""
from copy import deepcopy
import pytest
from scripts.compare_research_runs import validate_matched


def _protocol():
    return {'benchmark_lock_sha256':'fixture', 'prompt_contract_sha256':'prompt',
            'code_sha256':{'pipeline':'same'}, 'score_version':2,
            'primary_answer_metric':'context_grounded_key_pass',
            'request_schedule':[{'id':'q1','mode':'retrieved'}],
            'run_config':{'split':'test','mode':'retrieved','top_k':3},
            'model':{'generation_options':{'seed':42},'thinking':False}}


@pytest.mark.parametrize('mutation', ['prompt','code','top_k','decoding','question','scope'])
def test_rejects_mixed_protocol_or_question(mutation):
    left,right = _protocol(),_protocol()
    a = {'id':'q1','mode':'retrieved','question':'Which deductible?',
         'answerable':True,'document_scope':['policy-a']}
    b = deepcopy(a)
    if mutation == 'prompt': right['prompt_contract_sha256']='different'
    if mutation == 'code': right['code_sha256']['pipeline']='different'
    if mutation == 'top_k': right['run_config']['top_k']=5
    if mutation == 'decoding': right['model']['generation_options']['seed']=7
    if mutation == 'question': b['question']='Which limit?'
    if mutation == 'scope': b['document_scope']=['policy-b']
    with pytest.raises(ValueError,match='Unmatched'):
        validate_matched(left,right,[a],[b])


def test_model_digest_and_size_may_differ_in_matched_deployment_comparison():
    left,right = _protocol(),_protocol()
    left['model'].update({'model_digest':'old','parameter_size':'3B'})
    right['model'].update({'model_digest':'new','parameter_size':'4.7B'})
    validate_matched(left,right,[],[])
