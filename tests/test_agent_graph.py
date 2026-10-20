"""Control-flow tests with scripted fixtures; no model quality claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.insurerag_vlm.agent_demo import QUESTIONS, SyntheticController, build_synthetic_retriever
from src.insurerag_vlm.agent_draft import Limits
from src.insurerag_vlm.agent_graph import ReviewAgent, ReviewError, database_lock
from src.insurerag_vlm.agent_graph_demo import FixtureRetriever

REPO = Path(__file__).resolve().parents[1]


class CountingRetriever(FixtureRetriever):
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        return super().search(query, top_k)


class CountingController(SyntheticController):
    def __init__(self, scenario='ordinary'):
        super().__init__(scenario)
        self.calls = 0

    def next_action(self, *args):
        self.calls += 1
        return super().next_action(*args)


def create(tmp_path, scenario='ordinary', **kwargs):
    r, c = CountingRetriever(), CountingController(scenario)
    return ReviewAgent(tmp_path / 'review.sqlite', r, c, runtime_id='synthetic-v1', **kwargs), r, c


def decision(state, action='approve', ident='d1', **kwargs):
    return {'decision_id': ident, 'reviewer': 'test-operator', 'action': action,
            'version': state['version'], 'approval_id': state['approval']['approval_id'], **kwargs}


@pytest.mark.parametrize('scenario', ['ordinary', 'numeric', 'multi'])
def test_every_answer_pauses_and_restarts_without_approval(tmp_path, scenario):
    agent, r, c = create(tmp_path, scenario)
    s = agent.start('one', QUESTIONS[scenario])
    assert s['status'] == 'awaiting_review' and s['answer'] is None
    assert s['approval']['draft'] and s['approval']['evidence']
    if scenario == 'numeric':
        assert s['calculations'][0]['value'] == '12.0'
    counts = (r.calls, c.calls)
    for _ in range(2):
        assert agent.start('one', QUESTIONS[scenario])['answer'] is None
    assert (r.calls, c.calls) == counts
    restarted, r2, c2 = create(tmp_path, scenario)
    assert restarted.status('one')['approval'] == s['approval']
    final = restarted.resume('one', decision(s))
    assert final['status'] == 'approved' and final['answer'] == s['draft']
    assert (r2.calls, c2.calls) == (0, 0)
    assert restarted.resume('one', decision(s)) == final
    assert len(final['reviews']) == 1


def test_edit_requires_fresh_review_rejects_stale_and_conflicting_commands(tmp_path):
    a, r, c = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    old = decision(s)
    claims = copy.deepcopy(s['draft'])
    claims[0]['text'] = 'For this fictional example only, ask the insurer for a policy copy.'
    edit = decision(s, 'edit', 'edit1', claims=claims)
    changed = a.resume('one', edit)
    assert changed['status'] == 'awaiting_review' and changed['answer'] is None
    assert changed['version'] == 2 and changed['approval']['approval_id'] != old['approval_id']
    counts = (r.calls, c.calls)
    assert a.resume('one', edit) == changed
    with pytest.raises(ReviewError, match='Stale'):
        a.resume('one', old)
    conflict = dict(edit, action='approve')
    conflict.pop('claims')
    with pytest.raises(ReviewError, match='different request'):
        a.resume('one', conflict)
    final = a.resume('one', decision(changed, ident='approve2'))
    assert final['answer'] == claims and len(final['reviews']) == 2
    assert (r.calls, c.calls) == counts


def test_return_to_retrieval_uses_budget_and_new_approval(tmp_path):
    a, r, c = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    again = a.resume('one', decision(s, 'retrieve', query='Cedar policy copy conditions', gap='Recheck policy copy scope.'))
    assert again['status'] == 'awaiting_review' and again['answer'] is None
    assert again['version'] == 2
    assert again['remaining']['searches'] == s['remaining']['searches'] - 1
    assert again['remaining']['actions'] < s['remaining']['actions']
    assert r.calls == 2
    with pytest.raises(ReviewError, match='Stale'):
        a.resume('one', decision(s, ident='old'))
    assert a.resume('one', decision(again, ident='new'))['status'] == 'approved'


def test_reject_never_releases_answer(tmp_path):
    a, r, c = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    d = decision(s, 'reject')
    rejected = a.resume('one', d)
    assert rejected['status'] == 'rejected' and rejected['answer'] is None
    assert a.resume('one', d) == rejected
    with pytest.raises(ReviewError):
        a.resume('one', decision(s, ident='late'))


@pytest.mark.parametrize('mutation', [
    lambda d: {}, lambda d: dict(d, action=True), lambda d: dict(d, version=True),
    lambda d: dict(d, approval_id='wrong'), lambda d: dict(d, reviewer=''),
    lambda d: dict(d, extra=True), lambda d: dict(d, action='auto'),
])
def test_invalid_decision_preserves_pending_state(tmp_path, mutation):
    a, _, _ = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    with pytest.raises((ReviewError, ValueError)):
        a.resume('one', mutation(decision(s)))
    assert a.status('one') == s


def test_edit_bad_citation_and_noop_fail_before_resume(tmp_path):
    a, _, _ = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    for claims in [s['draft'], [{'text': 'Fake', 'citations': [{'id': 'fake', 'quote': 'missing'}]}]]:
        with pytest.raises(ReviewError):
            a.resume('one', decision(s, 'edit', claims=claims))
    assert a.status('one') == s


def test_thread_and_runtime_binding(tmp_path):
    a, _, _ = create(tmp_path)
    one = a.start('one', QUESTIONS['ordinary'])
    two = a.start('two', QUESTIONS['ordinary'])
    assert one['approval']['approval_id'] != two['approval']['approval_id']
    with pytest.raises(ReviewError):
        a.resume('two', decision(one))
    with pytest.raises(ReviewError):
        a.start('one', QUESTIONS['numeric'])
    changed = ReviewAgent(a.database, FixtureRetriever(), SyntheticController('ordinary'), runtime_id='changed')
    with pytest.raises(ReviewError, match='Runtime'):
        changed.start('one', QUESTIONS['ordinary'])
    with pytest.raises(ReviewError):
        a.status('unknown')


@pytest.mark.parametrize('scenario,reason', [('refusal', 'abstained'), ('injection', 'invalid_action')])
def test_refusal_and_injection_cannot_reach_approval(tmp_path, scenario, reason):
    a, _, _ = create(tmp_path, scenario)
    s = a.start('one', QUESTIONS[scenario])
    assert s['stop_reason'] == reason and s['answer'] is None and s['approval'] is None
    if scenario == 'refusal':
        assert 'fictional guide' in s['abstention']


def test_budget_survives_restart_and_human_retrieval_does_not_reset_it(tmp_path):
    a, _, _ = create(tmp_path, limits=Limits(supplemental_searches=0))
    s = a.start('one', QUESTIONS['ordinary'])
    b, _, _ = create(tmp_path, limits=Limits(supplemental_searches=0))
    with pytest.raises(ReviewError, match='budget'):
        b.resume('one', decision(s, 'retrieve', query='new query', gap='missing'))
    assert b.status('one')['remaining'] == s['remaining']


def test_actual_hybrid_retriever_can_supply_graph(tmp_path):
    root = tmp_path / 'hybrid'
    root.mkdir()
    retriever = build_synthetic_retriever(root)
    a = ReviewAgent(tmp_path / 'review.sqlite', retriever, SyntheticController('multi'), runtime_id='hybrid-fixture')
    s = a.start('hybrid', QUESTIONS['multi'])
    assert s['status'] == 'awaiting_review'
    assert len(s['draft']) == 2


def test_separate_process_restart_and_repeat_command(tmp_path):
    db = tmp_path / 'review.sqlite'
    args = [sys.executable, '-m', 'src.insurerag_vlm.agent_graph_demo']
    common = ['--db', str(db), '--thread-id', 'restart']
    def run(command, *extra):
        return json.loads(subprocess.check_output(args + [command] + common + list(extra), text=True, cwd=REPO))['result']
    s = run('start')
    assert run('status')['answer'] is None
    file = tmp_path / 'decision.json'
    file.write_text(json.dumps(decision(s)), encoding='utf8')
    final = run('resume', '--decision', str(file))
    assert final['status'] == 'approved'
    assert run('resume', '--decision', str(file)) == final
    assert run('start') == final


def test_lock_rejects_simultaneous_process(tmp_path):
    a, _, _ = create(tmp_path)
    with database_lock(a.database):
        code = ('from src.insurerag_vlm.agent_graph import database_lock,BusyError\n'
                'import sys\ntry:\n with database_lock(sys.argv[1]): pass\n'
                'except BusyError:\n sys.exit(23)\n')
        result = subprocess.run([sys.executable, '-c', code, str(a.database)], capture_output=True, cwd=REPO)
    assert result.returncode == 23


def test_crash_inside_external_call_does_not_repeat_it(tmp_path):
    # A real process exits after the durable started marker but before a result.
    db = tmp_path / 'review.sqlite'
    marker = tmp_path / 'calls.txt'
    code = """import os,sys
from pathlib import Path
from src.insurerag_vlm.agent_graph import ReviewAgent
from src.insurerag_vlm.agent_demo import SyntheticController
class CrashRetriever:
 def search(self,*args):
  Path(sys.argv[2]).write_text('called')
  os._exit(17)
ReviewAgent(sys.argv[1],CrashRetriever(),SyntheticController('ordinary'),runtime_id='synthetic-v1').start('crash','question')
"""
    p = subprocess.run([sys.executable, '-c', code, str(db), str(marker)], capture_output=True, cwd=REPO)
    assert p.returncode == 17 and marker.read_text() == 'called'
    a, r, c = create(tmp_path)
    s = a.start('crash', 'question')
    assert s['stop_reason'] == 'uncertain_operation' and s['answer'] is None
    assert r.calls == c.calls == 0


def test_completed_external_call_is_reused_after_checkpoint_gap(tmp_path):
    a, r, _ = create(tmp_path)
    original = a._operation
    def fail_after_result(conn, state, name, call):
        result = original(conn, state, name, call)
        if name == 'search':
            raise SystemExit('simulated process failure')
        return result
    a._operation = fail_after_result
    with pytest.raises(SystemExit):
        a.start('one', QUESTIONS['ordinary'])
    assert r.calls == 1
    b, r2, c2 = create(tmp_path)
    s = b.start('one', QUESTIONS['ordinary'])
    assert s['status'] == 'awaiting_review' and r2.calls == 0
    assert c2.calls == 2


def test_crash_after_interrupt_checkpoint_recovers_same_decision(tmp_path):
    a, _, _ = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    d = decision(s)
    # Exercise actual persisted review-node output, then restart before apply.
    from langgraph.types import Command
    with a._session() as (_, graph):
        graph.invoke(Command(resume=d), config=a._config('one'), interrupt_after=['review'], durability='sync')
    b, r, c = create(tmp_path)
    final = b.resume('one', d)
    assert final['status'] == 'approved' and len(final['reviews']) == 1
    assert r.calls == c.calls == 0
    assert b.resume('one', d) == final


@pytest.mark.parametrize('limits,reason', [
    (Limits(actions=1), 'action_budget'),
    (Limits(token_budget=1), 'token_budget'),
    (Limits(context_tokens=1), 'context_budget'),
])
def test_graph_budget_failures_do_not_offer_approval(tmp_path, limits, reason):
    a, _, _ = create(tmp_path, limits=limits)
    s = a.start('budget', QUESTIONS['ordinary'])
    assert s['stop_reason'] == reason and s['answer'] is None and s['approval'] is None


@pytest.mark.parametrize('response', [
    '{"action":"ANSWER","action":"READ","parameters":{},"gaps":[]}',
    '{"action":"APPROVE","parameters":{},"gaps":[]}',
    '{"action":"READ","parameters":{"ids":["unknown"]},"gaps":[]}',
    '{"action":"ANSWER","parameters":{"claims":[]},"gaps":[]}',
    '{"action":"ANSWER","parameters":{"claims":[{"text":"Fake","citations":[{"id":"unknown","quote":"fake"}]}]},"gaps":[]}',
    '{"action":"SEARCH","parameters":{"query":"new query"},"gaps":[]}',
])
def test_graph_rejects_untrusted_controller_actions(tmp_path, response):
    class Controller:
        def next_action(self, *args):
            return response, {}
    a = ReviewAgent(tmp_path/'review.sqlite', FixtureRetriever(), Controller(), runtime_id='malformed-fixture')
    s = a.start('one', 'Synthetic question')
    assert s['stop_reason'] == 'invalid_action' and s['answer'] is None
    assert s['approval'] is None


def test_controller_completed_before_crash_is_not_called_again(tmp_path):
    a, r, c = create(tmp_path)
    original = a._operation
    def fail(conn, state, name, call):
        result = original(conn, state, name, call)
        if name == 'model':
            raise SystemExit('crash after model result commit')
        return result
    a._operation = fail
    with pytest.raises(SystemExit):
        a.start('one', QUESTIONS['ordinary'])
    assert c.calls == 1
    b, r2, c2 = create(tmp_path)
    s = b.start('one', QUESTIONS['ordinary'])
    assert s['status'] == 'awaiting_review' and c2.calls == 1 and r2.calls == 0
    assert s['usage']['controller_calls'] == 2


def test_edit_recovery_after_decision_checkpoint_does_not_increment_twice(tmp_path):
    from langgraph.types import Command
    a, _, _ = create(tmp_path)
    s = a.start('one', QUESTIONS['ordinary'])
    claims = copy.deepcopy(s['draft'])
    claims[0]['text'] = 'In the fictional guide, a policy copy is requested from the insurer.'
    d = decision(s, 'edit', claims=claims)
    with a._session() as (_, graph):
        graph.invoke(Command(resume=d), config=a._config('one'), interrupt_after=['apply_review'], durability='sync')
    b, r, c = create(tmp_path)
    recovered = b.resume('one', d)
    assert recovered['version'] == 2 and recovered['answer'] is None
    assert recovered['status'] == 'awaiting_review' and len(recovered['reviews']) == 1
    assert b.resume('one', d) == recovered
    assert r.calls == c.calls == 0


def test_saved_local_model_review_works_without_model_service(tmp_path, monkeypatch, capsys):
    from src.insurerag_vlm import agent_graph_demo as demo
    from src.insurerag_vlm.agent_demo import DOCUMENTS
    from src.insurerag_vlm.agent_graph import digest
    def unavailable(*args):
        pytest.fail('Review/status must not initialize or contact the model')
    monkeypatch.setattr(demo, 'LocalModelController', unavailable)
    runtime = digest({'fixtures': DOCUMENTS, 'scenario': 'ordinary',
                      'controller': 'ollama', 'model': 'ollama:qwen3.5:4b',
                      'digest': 'test-digest', 'endpoint': 'http://localhost:11434'})
    # Scripted setup only: represents the persisted state, not an actual model run.
    db = tmp_path / 'review.sqlite'
    a = ReviewAgent(db, FixtureRetriever(), SyntheticController('ordinary'), runtime_id=runtime)
    s = a.start('offline-review', QUESTIONS['ordinary'])
    common = ['--db', str(db), '--thread-id', 'offline-review', '--controller', 'ollama',
              '--expected-digest', 'test-digest']
    assert demo.main(['status', *common]) == 0
    assert json.loads(capsys.readouterr().out)['result']['answer'] is None
    file = tmp_path / 'decision.json'
    file.write_text(json.dumps(decision(s)), encoding='utf8')
    assert demo.main(['resume', *common, '--decision', str(file)]) == 0
    assert json.loads(capsys.readouterr().out)['result']['status'] == 'approved'


def test_lazy_model_initializes_once_on_generation_only(monkeypatch):
    from src.insurerag_vlm import agent_graph_demo as demo
    initializations = []
    class Controller:
        def __init__(self, *settings):
            initializations.append(settings)
        def next_action(self, *args):
            return 'scripted-response', {}
    monkeypatch.setattr(demo, 'LocalModelController', Controller)
    lazy = demo.LazyLocalController('ollama:test', 'http://localhost:11434', 'test-digest')
    assert initializations == []
    for _ in range(2):
        assert lazy.next_action('system', '{}', 10, 1) == ('scripted-response', {})
    assert len(initializations) == 1
