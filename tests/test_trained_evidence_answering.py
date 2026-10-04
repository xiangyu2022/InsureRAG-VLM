from scripts.answer_evidence_reranker import answer_retrieval


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
