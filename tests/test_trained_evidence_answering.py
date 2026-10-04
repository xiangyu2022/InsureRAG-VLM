from scripts.answer_evidence_reranker import answer_retrieval
import pytest


class FakeClient:
    def __init__(self, text, truncated=False):
        self.text = text
        self.truncated = truncated
        self.inputs = []

    def generate(self, prompt):
        self.inputs.append((None, prompt))
        return self.text

    def generate_chat(self, system, prompt):
        self.inputs.append((system, prompt))
        return self.text

    def backend_metadata(self):
        return {'last_generation': {'truncated': self.truncated}}

    def answer_trace(self, **kwargs):
        return {'generation_used': True, 'answer_backend': 'fake-model', 'backend_metadata': self.backend_metadata()}


def retrieval(text='A deductible is the amount paid by the insured for a covered loss.'):
    return {'question': 'What is a deductible?', 'corpus': 'insuranceqa', 'annual_report_scope': None,
            'results': [{'rank': 1, 'answer_id': '123', 'text': text, 'source_url': 'https://example.org/faq'}]}


def test_research_prompt_keeps_real_provenance_and_raw_served_separation():
    client = FakeClient('A deductible is the amount paid by the insured.\nSOURCE: insuranceqa:123')
    result = answer_retrieval(retrieval(), client, prompt_mode='research')
    assert 'SECTION: https://example.org/faq' in result['context']
    assert result['raw_answer'] == client.text == result['served']['raw_answer']
    assert result['served']['generation_used']
    assert 'public FAQ examples are insufficient' in client.inputs[0][0]
    assert 'gold_answer_ids' not in result['prompt']


def test_research_prompt_cannot_relabel_an_unknown_citation():
    result = answer_retrieval(retrieval(), FakeClient('The deductible is $900.\nSOURCE: invented:123'), prompt_mode='research')
    assert result['served']['abstain']
    assert result['served']['citations'] == []
    assert result['served']['citation_support_reason'] == 'missing_citation'


def test_truncated_correct_looking_answer_is_not_served():
    result = answer_retrieval(retrieval(), FakeClient('A deductible is the amount paid by the insured.\nSOURCE: insuranceqa:123', True))
    assert result['served']['abstain'] and result['served']['generation_truncated']


def test_public_evidence_cannot_establish_personal_policy_identifier():
    request = retrieval('An example collision deductible is $500.')
    request['question'] = 'What is the collision deductible on my policy ZX-14?'
    result = answer_retrieval(request, FakeClient('Your collision deductible is $500.\nSOURCE: insuranceqa:123'), prompt_mode='research')
    assert result['served']['abstain']


def test_research_packing_respects_budget_and_does_not_invent_pdf_page():
    result = answer_retrieval(retrieval('A deductible is a shared cost. ' * 500), FakeClient('I cannot answer.'), max_context_chars=300, prompt_mode='research')
    assert len(result['context']) <= 300
    assert '#page=' not in result['context']
    assert result['context'].startswith('SOURCE: insuranceqa:123\n')


def test_definition_is_not_replaced_with_an_arbitrary_amount_example():
    request = retrieval('A deductible is the amount paid by the insured for a covered loss. For example, a $500 deductible applies to a $2000 loss.')
    request['question'] = 'What is an insurance deductible?'
    raw = 'A deductible is the amount paid by the insured for a covered loss.\nSOURCE: insuranceqa:123'
    result = answer_retrieval(request, FakeClient(raw), prompt_mode='research')
    assert not result['served']['abstain']
    assert not result['served']['answer_repaired']
    assert '$500' not in result['served']['answer']
    assert result['served']['raw_answer'] == raw


@pytest.mark.parametrize('question', ['What is my deductible?', 'What is the collision deductible?',
    'What is a deductible and how much is mine?', 'Define deductible for policy ZX-14.',
    'How much is an insurance premium?', 'What is a coverage limit for my policy?'])
def test_numeric_or_personal_requests_keep_the_existing_amount_requirement(question):
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    assert DocumentRetrievalPipeline._question_requires_numeric_evidence(question)


def test_comparison_runner_preserves_utf8_evidence_before_generation(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import compare_qwen35_grounding as runner
    from src.insurerag_vlm import vlm
    data = tmp_path / 'inputs'; data.mkdir()
    case = {'id': 'unicode', 'cohort': 'insuranceqa'}
    evidence = {'unicode': {'question': 'Define deductible.', 'results': [{'text': 'café\u00a0€'}]}}
    (data / 'cases.json').write_text(json.dumps([case], ensure_ascii=False), encoding='utf8')
    (data / 'retrieval.json').write_text(json.dumps(evidence, ensure_ascii=False), encoding='utf8')
    protocol = {'cases_sha256': runner.digest(data / 'cases.json'), 'retrieval_sha256': runner.digest(data / 'retrieval.json'),
                'num_ctx': 4096, 'num_predict': 384, 'temperature': 0, 'seed': 42}
    (data / 'protocol.json').write_text(json.dumps(protocol), encoding='utf8')
    captured = []
    def fake_answer(record, client, **kwargs):
        captured.append(record)
        return {'raw_answer': 'unused', 'served': {'abstain': True}}
    monkeypatch.setattr(runner, 'resources', lambda url: {'ollama_ps': {'models': []}})
    monkeypatch.setattr(runner, 'answer_retrieval', fake_answer)
    monkeypatch.setattr(vlm, 'VLMClient', lambda *args, **kwargs: FakeClient('unused'))
    runner.run(SimpleNamespace(data=data, output=tmp_path / 'output', ids=None, model='test',
                               expected_digest='test', base_url='http://localhost:1', prompt_mode='main'))
    assert captured == [evidence['unicode']]
