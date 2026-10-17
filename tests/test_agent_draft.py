"""Synthetic controller-contract tests. No holdout data or model downloads."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.insurerag_vlm.agent_draft import (
    HybridRetrieverAdapter, InsuranceAgent, InvalidAction, Limits,
    LocalModelController, calculate, decode_action,
)
from src.insurerag_vlm.agent_demo import (
    DOCUMENTS, QUESTIONS, SyntheticController, action, build_synthetic_retriever, main,
)


class Retrieval:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [{'source': n, 'text': t} for n, t in DOCUMENTS.items()]
        self.calls = []

    def search(self, query, top_k):
        self.calls.append(query)
        return self.rows[:top_k]


class Script:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.calls = 0

    def next_action(self, system, payload, output_tokens, timeout):
        self.calls += 1
        step = next(self.steps)
        return (step(json.loads(payload)) if callable(step) else step), {}


@pytest.mark.parametrize('scenario,expected', [('ordinary', 'answered'), ('numeric', 'answered'),
    ('multi', 'answered'), ('refusal', 'abstained'), ('injection', 'invalid_action')])
def test_synthetic_scenarios(scenario, expected):
    result = InsuranceAgent(Retrieval(), SyntheticController(scenario)).run(QUESTIONS[scenario])
    assert result['stop_reason'] == expected, result
    if scenario == 'numeric':
        assert result['calculations'][0]['value'] == '12.0'
        assert result['calculations'][0]['unit'] == 'USD'
    if scenario == 'multi':
        assert len(result['answer']) == 2 and len(result['evidence']) == 2
    assert result['usage']['calls_with_reported_usage'] == 0


def test_real_hybrid_and_insurance_reranker_on_synthetic_text(tmp_path):
    retriever = build_synthetic_retriever(tmp_path)
    for scenario in ['ordinary', 'numeric', 'multi', 'refusal']:
        result = InsuranceAgent(retriever, SyntheticController(scenario)).run(QUESTIONS[scenario])
        assert result['stop_reason'] == ('abstained' if scenario == 'refusal' else 'answered'), result


def test_optional_existing_cross_encoder_interface():
    pipeline = Mock()
    rows = [{'source': 'a', 'text': 'one'}, {'source': 'b', 'text': 'two'}]
    pipeline.retrieve_text_candidates.return_value = rows
    pipeline.rerank_multimodal_candidates.return_value = rows
    cross = Mock(); cross.score.return_value = [0.1, 0.9]
    adapter = HybridRetrieverAdapter(pipeline, Path('synthetic'), cross)
    assert adapter.search('rule', 1) == [rows[1]]
    cross.score.assert_called_once_with('rule', ['one', 'two'])
    cross.score.return_value = [float('nan'), 1]
    with pytest.raises(ValueError):
        adapter.search('rule', 1)


@pytest.mark.parametrize('raw', ['{}', '[]', '{"action":"READ","action":"ANSWER"}',
    '{"x":NaN}', '```json\n{}\n```', 'x' * 12001])
def test_bad_json(raw):
    with pytest.raises(InvalidAction):
        decode_action(raw)


@pytest.mark.parametrize('step', [action('EXECUTE', {}), action('READ', {'ids': ['../../secret']}),
    action('SEARCH', {'query': 'policy', 'shell': 'anything'}, ['missing rule']),
    action('ANSWER', {'claims': []}), action('ABSTAIN', {'reason': 'unknown'})])
def test_invalid_parameters_fail_closed(step):
    result = InsuranceAgent(Retrieval(), Script([step])).run('synthetic rule')
    assert result['stop_reason'] == 'invalid_action' and result['answer'] is None


def test_repeat_normalized_query_does_not_invoke_retriever():
    retrieval = Retrieval()
    result = InsuranceAgent(retrieval, Script([action('SEARCH', {'query': 'SYNTHETIC   rule!!!'}, ['Need detail'])])).run('synthetic rule')
    assert result['stop_reason'] == 'invalid_action'
    assert len(retrieval.calls) == 1


def test_search_loop_budget_exactly_three_supplemental():
    retrieval = Retrieval()
    steps = [action('SEARCH', {'query': 'different ' + str(i)}, ['Need specific rule ' + str(i)]) for i in range(4)]
    result = InsuranceAgent(retrieval, Script(steps), Limits(token_budget=60000)).run('initial')
    assert result['stop_reason'] == 'search_budget'
    assert len(retrieval.calls) == 4
    assert result['remaining']['searches'] == 0


def test_successful_supplemental_search_then_read_answer():
    class StagedRetrieval(Retrieval):
        def search(self, query, top_k):
            return [] if query == 'initial' else super().search(query, top_k)
    script = Script([action('SEARCH', {'query': 'Cedar guide'}, ['Need guide']),
        lambda s: action('READ', {'ids': [next(e for e, row in s['candidates'].items() if row['source'] == 'guide.txt')]}, ['Need rule']),
        lambda s: action('ANSWER', {'claims': [{'text': 'Request a policy copy.', 'citations': [
            {'id': next(iter(s['evidence'])), 'quote': 'request a policy copy from the insurer'}]}]})])
    result = InsuranceAgent(StagedRetrieval(), script).run('initial')
    assert result['stop_reason'] == 'answered', result
    assert [e['action'] for e in result['trace']] == ['INITIAL_SEARCH', 'SEARCH', 'READ', 'ANSWER']


def test_repeated_read_is_rejected():
    step = lambda s: action('READ', {'ids': [next(iter(s['candidates']))]}, ['Need rule'])
    result = InsuranceAgent(Retrieval(), Script([step, step])).run('rule')
    assert result['stop_reason'] == 'invalid_action'
    assert 'Repeated action' in result['error']


@pytest.mark.parametrize('limits,reason', [(Limits(token_budget=10), 'token_budget'),
    (Limits(context_tokens=100), 'context_budget'), (Limits(actions=1), 'action_budget')])
def test_budgets(limits, reason):
    result = InsuranceAgent(Retrieval(), SyntheticController('ordinary'), limits).run(QUESTIONS['ordinary'])
    assert result['stop_reason'] == reason and result['answer'] is None


def test_late_controller_response_is_discarded():
    now = [0.0]
    class Slow:
        def next_action(self, *args):
            now[0] += 20
            return action('ABSTAIN', {'reason': 'missing'}, ['missing']), {}
    result = InsuranceAgent(Retrieval(), Slow(), clock=lambda: now[0]).run('rule')
    assert result['stop_reason'] == 'time_budget' and result['answer'] is None


def test_slow_retrieval_stops_before_controller():
    now = [0.0]
    class SlowRetrieval(Retrieval):
        def search(self, *args):
            now[0] += 20
            return self.rows
    result = InsuranceAgent(SlowRetrieval(), Script([]), clock=lambda: now[0]).run('rule')
    assert result['stop_reason'] == 'time_budget'
    assert result['usage']['controller_calls'] == 0


def operands():
    return {'operation': 'multiply', 'operands': [
        {'value': '120', 'unit': 'USD', 'source': 'question', 'quote': '120 USD'},
        {'value': '10', 'unit': 'percent', 'source': 'question', 'quote': '10 percent'}]}


def test_decimal_calculator_provenance():
    assert calculate(operands(), {}, '120 USD and 10 percent')['value'] == '12.0'


@pytest.mark.parametrize('change', [lambda p: p.update(operation='eval'),
    lambda p: p.update(operation=[]), lambda p: p.update(operation='add'),
    lambda p: p['operands'][0].update(value='__import__("os")'),
    lambda p: p['operands'][0].update(value='1e9'),
    lambda p: p['operands'][0].update(value='12'),
    lambda p: p['operands'][0].update(source='not_read'),
    lambda p: p['operands'][0].update(unit='CAD'),
    lambda p: p['operands'][0].update(quote='not present'),
    lambda p: p.update(extra='permission')])
def test_calculator_rejects_untrusted_or_implicit_inputs(change):
    params = operands(); change(params)
    with pytest.raises(InvalidAction):
        calculate(params, {}, '120 USD and 10 percent')


def test_zero_divisor():
    p = operands(); p['operation'] = 'divide'; p['operands'][1].update(value='0', quote='0 percent')
    with pytest.raises(InvalidAction):
        calculate(p, {}, '120 USD and 0 percent')


def test_citations_require_read_and_exact_quote():
    def answer(s):
        return action('ANSWER', {'claims': [{'text': 'anything', 'citations': [
            {'id': next(iter(s['candidates'])), 'quote': 'invented quote'}]}]})
    for steps in [[answer], [lambda s: action('READ', {'ids': [next(iter(s['candidates']))]}), answer]]:
        result = InsuranceAgent(Retrieval(), Script(steps)).run('rule')
        assert result['stop_reason'] == 'invalid_action' and result['answer'] is None


def test_document_injection_cannot_authorize_tool_even_after_read():
    read = lambda s: action('READ', {'ids': [next(iter(s['candidates']))]})
    result = InsuranceAgent(Retrieval([{'source': 'attack', 'text': DOCUMENTS['injection.txt']}]),
        Script([read, action('SHELL', {'command': 'read secrets'})])).run('Read document')
    assert result['stop_reason'] == 'invalid_action'
    assert result['remaining']['searches'] == 3


def test_local_controller_rejects_remote_or_paid_backend_before_initializing():
    for model, url in [('openai:gpt-test', 'http://localhost:11434'), ('ollama:demo', 'http://example.com'),
                       ('ollama:demo', 'http://localhost:11434?redirect=bad')]:
        with pytest.raises(ValueError):
            LocalModelController(model, url, 'digest')


def test_existing_vlm_adapter_preserves_usage_and_deadline(monkeypatch):
    import src.insurerag_vlm.vlm as vlm
    client = Mock(); client.generation_options = {}; client.generate_chat.return_value = '{}'
    client.last_generation_metadata = {'prompt_eval_count': 100, 'eval_count': 20}
    factory = Mock(return_value=client); monkeypatch.setattr(vlm, 'VLMClient', factory)
    adapter = LocalModelController('ollama:synthetic', expected_digest='digest')
    raw, usage = adapter.next_action('system', 'payload', 256, 10)
    assert usage == {'prompt_tokens': 100, 'output_tokens': 20}
    assert client.request_timeout == 5 and client.generation_options['num_predict'] == 256
    assert factory.call_args.kwargs['expected_model_digest'] == 'digest'


def test_cli_creates_trace_and_refuses_overwrite(tmp_path, capsys):
    target = tmp_path / 'trace.json'
    assert main(['--scenario', 'numeric', '--trace', str(target)]) == 0
    report = json.loads(target.read_text())
    assert report['results']['numeric']['stop_reason'] == 'answered'
    assert report['holdout_data_used'] is False
    with pytest.raises(FileExistsError):
        main(['--scenario', 'refusal', '--trace', str(target)])
    capsys.readouterr()


@pytest.mark.parametrize('kwargs', [{'supplemental_searches': 4}, {'actions': True}, {'seconds': float('nan')}, {'output_tokens': 0}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        Limits(**kwargs)
