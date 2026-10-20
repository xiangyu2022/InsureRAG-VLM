"""Durable, bounded LangGraph workflow. Every answer requires explicit review.

This local API assumes a trusted operator and private SQLite database. It is not
an authentication service. No holdout data or evaluation code is imported.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import copy
import json
from pathlib import Path
import re
import sqlite3
import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context

from .agent_draft import (SYSTEM, InvalidAction, Limits, calculate, canonical,
                          decode_action, keys, sequence, string)


SCHEMA = 1


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha256(encoded(value).encode('utf8')).hexdigest()


class ReviewError(ValueError):
    """Invalid, stale, conflicting, or absent review; no release is authorized."""


class BusyError(RuntimeError):
    pass


class UncertainOperation(RuntimeError):
    """A prior process may have performed the call. Never retry automatically."""


@contextmanager
def database_lock(path):
    """Process-wide serialization, released by the OS even after process death."""
    with open(str(path) + '.lock', 'a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        if __import__('os').name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BusyError('Another operation is using this database') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BusyError('Another operation is using this database') from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_claims(claims, evidence):
    for claim in sequence(claims, 5):
        keys(claim, ['text', 'citations'])
        string(claim['text'], 1000)
        for citation in sequence(claim['citations'], 5):
            keys(citation, ['id', 'quote'])
            eid, quote = string(citation['id'], 64), string(citation['quote'], 600)
            if eid not in evidence or quote not in evidence[eid]['text']:
                raise InvalidAction('Citation must quote read evidence exactly')
    return copy.deepcopy(claims)


class ReviewAgent:
    """SQLite-backed graph with idempotent local review commands.

    runtime_id must identify the immutable corpus/controller configuration.
    All methods use the same database lock, including status reads. The graph
    itself is internal; callers use start/status/resume, never update_state.
    """

    def __init__(self, database, retriever, controller, *, runtime_id, limits=None):
        self.database = Path(database).resolve()
        if not self.database.parent.is_dir():
            raise ValueError('Create a private database directory first')
        self.retriever, self.controller = retriever, controller
        self.limits = limits or Limits()
        self.profile = {'schema': SCHEMA, 'runtime_id': string(runtime_id, 300),
                        'limits': asdict(self.limits)}

    @contextmanager
    def _session(self):
        with database_lock(self.database), tracing_context(enabled=False):
            conn = sqlite3.connect(self.database, check_same_thread=False)
            try:
                conn.execute('PRAGMA synchronous=FULL')
                conn.execute('''CREATE TABLE IF NOT EXISTS agent_operations
                    (thread TEXT, operation TEXT, status TEXT, result TEXT,
                     PRIMARY KEY(thread, operation))''')
                conn.commit()
                saver = SqliteSaver(conn)
                saver.setup()
                yield conn, self._compile(conn, saver)
            finally:
                conn.close()

    def _config(self, thread_id):
        if type(thread_id) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', thread_id):
            raise ReviewError('thread_id must contain 1..100 letters, digits, underscores or hyphens')
        return {'configurable': {'thread_id': thread_id}, 'recursion_limit': 200}

    def _check_profile(self, snapshot):
        if snapshot.values and snapshot.values['profile'] != self.profile:
            raise ReviewError('Runtime/corpus/limits changed; use the original configuration or a new thread')

    def _invoke(self, graph, value, config):
        graph.invoke(value, config=config, durability='sync')

    def start(self, thread_id, question):
        config = self._config(thread_id)
        string(question, 1500)
        with self._session() as (_, graph):
            snapshot = graph.get_state(config)
            self._check_profile(snapshot)
            if snapshot.values:
                if snapshot.values['question'] != question:
                    raise ReviewError('thread_id is already bound to a different question')
                if snapshot.next:
                    self._invoke(graph, None, config)
            else:
                self._invoke(graph, {
                    'thread_id': thread_id, 'profile': self.profile, 'question': question,
                    'scope': 'consumer_rules_only', 'known': {}, 'candidates': {},
                    'evidence': {}, 'gaps': ['Evidence has not been read.'],
                    'calculations': [], 'trace': [], 'queries': [], 'calls': [],
                    'remaining': {'searches': self.limits.supplemental_searches,
                                  'actions': self.limits.actions,
                                  'charged_tokens': self.limits.token_budget},
                    'usage': {'controller_calls': 0, 'reported_prompt_tokens': 0,
                              'reported_output_tokens': 0, 'calls_with_reported_usage': 0},
                    'active_seconds': 0.0, 'answer': None, 'draft': None, 'version': 0,
                    'approval': None, 'decision': None, 'reviews': [],
                    'status': 'running', 'stop_reason': None,
                    'review_feedback': None, 'retrieval_round': 0,
                }, config)
            return self._view(graph.get_state(config))

    def status(self, thread_id):
        with self._session() as (_, graph):
            snapshot = graph.get_state(self._config(thread_id))
            self._check_profile(snapshot)
            if not snapshot.values:
                raise ReviewError('Unknown thread_id')
            return self._view(snapshot)

    def resume(self, thread_id, decision):
        config = self._config(thread_id)
        if type(decision) is not dict:
            raise ReviewError('An explicit review object is required')
        request = copy.deepcopy(decision)
        fingerprint = digest(request)
        with self._session() as (_, graph):
            snapshot = graph.get_state(config)
            self._check_profile(snapshot)
            if not snapshot.values:
                raise ReviewError('Unknown thread_id')
            s = snapshot.values
            # Recover a crash after the interrupt checkpoint or decision node.
            recorded = [r['request'] for r in s['reviews']]
            if s.get('decision'):
                recorded.append(s['decision'])
            for previous in recorded:
                if previous.get('decision_id') == request.get('decision_id'):
                    if digest(previous) != fingerprint:
                        raise ReviewError('decision_id was already used for a different request')
                    if snapshot.next:
                        self._invoke(graph, None, config)
                    return self._view(graph.get_state(config))
            if s.get('decision'):
                raise ReviewError('An earlier decision is pending; recover it before submitting another')
            self._validate_review(s, request)
            if snapshot.next != ('review',):
                raise ReviewError('Thread is not paused for human review')
            self._invoke(graph, Command(resume=request), config)
            return self._view(graph.get_state(config))

    def _validate_review(self, s, d):
        try:
            action = d.get('action')
            extras = {'approve': [], 'reject': [], 'edit': ['claims'], 'retrieve': ['query', 'gap']}
            if action not in extras:
                raise ReviewError('Choose approve, edit, retrieve or reject explicitly')
            keys(d, ['decision_id', 'reviewer', 'version', 'approval_id', 'action', *extras[action]])
            string(d['decision_id'], 100)
            string(d['reviewer'], 100)
            if s['status'] != 'awaiting_review' or not s['approval']:
                raise ReviewError('No answer is awaiting review')
            if type(d['version']) is not int or d['version'] != s['version'] or d['approval_id'] != s['approval']['approval_id']:
                raise ReviewError('Stale or wrong-thread approval')
            if action == 'edit':
                validate_claims(d['claims'], s['evidence'])
                if d['claims'] == s['draft']:
                    raise ReviewError('Edit must change the draft; it cannot approve it')
            if action == 'retrieve':
                query = string(d['query'], 400)
                string(d['gap'], 300)
                if not canonical(query) or canonical(query) in s['queries']:
                    raise ReviewError('Use a new concrete retrieval query')
                if s['remaining']['searches'] <= 0 or s['remaining']['actions'] <= 0:
                    raise ReviewError('Retrieval/action budget exhausted; reject or edit the draft')
        except (InvalidAction, KeyError, TypeError) as exc:
            raise ReviewError(str(exc)) from exc

    def _view(self, snapshot):
        s = snapshot.values
        result = copy.deepcopy({k: s.get(k) for k in (
            'thread_id', 'question', 'status', 'stop_reason', 'answer', 'draft', 'version',
            'approval', 'reviews', 'evidence', 'calculations', 'remaining', 'usage',
            'active_seconds', 'trace', 'error', 'abstention')})
        result['remaining']['seconds'] = round(max(0, self.limits.seconds - s['active_seconds']), 6)
        return result

    def _operation(self, conn, s, name, call):
        """Cache completed calls; a crash mid-call blocks instead of reissuing it."""
        key = digest([name, len(s['trace']), s['retrieval_round']])
        row = conn.execute('SELECT status,result FROM agent_operations WHERE thread=? AND operation=?',
                           (s['thread_id'], key)).fetchone()
        if row:
            if row[0] != 'complete':
                raise UncertainOperation('Prior ' + name + ' has no durable result; automatic retry disabled')
            return json.loads(row[1])
        conn.execute('INSERT INTO agent_operations VALUES (?, ?, ?, ?)',
                     (s['thread_id'], key, 'started', None))
        conn.commit()
        started = time.monotonic()
        value = call()
        result = {'value': value, 'seconds': time.monotonic() - started}
        conn.execute('UPDATE agent_operations SET status=?,result=? WHERE thread=? AND operation=?',
                     ('complete', encoded(result), s['thread_id'], key))
        conn.commit()
        return result

    def _stop(self, s, reason, error=None):
        s.update(status='stopped', stop_reason=reason, answer=None, approval=None)
        if error:
            s['error'] = error
        return s

    def _compile(self, conn, saver):
        lim = self.limits

        def guarded(fn):
            def run(state):
                s = copy.deepcopy(state)
                try:
                    return fn(s)
                except UncertainOperation as exc:
                    return self._stop(s, 'uncertain_operation', str(exc))
                except TimeoutError:
                    return self._stop(s, 'time_budget')
                except (InvalidAction, ReviewError, ValueError, TypeError, KeyError) as exc:
                    return self._stop(s, 'invalid_action', str(exc))
                except Exception as exc:
                    return self._stop(s, 'dependency_error', type(exc).__name__)
            return run

        def search(s, query):
            query = string(query, 1500)
            normalized = canonical(query)
            if not normalized or normalized in s['queries']:
                raise InvalidAction('Empty or repeated query')
            if s['active_seconds'] >= lim.seconds:
                raise TimeoutError()
            result = self._operation(conn, s, 'search', lambda: self.retriever.search(query, lim.top_k))
            s['active_seconds'] += result['seconds']
            if s['active_seconds'] > lim.seconds or result['seconds'] > lim.call_seconds:
                raise TimeoutError()
            s['queries'].append(normalized)
            ids = []
            for row in sequence(result['value'], lim.top_k, 0):
                source = string(row['source'], 2000)
                raw = string(row['text'], 100000)
                text = raw[:lim.evidence_chars]
                eid = sha256((source + '\0' + text).encode()).hexdigest()[:16]
                s['known'][eid] = {'id': eid, 'source': source, 'text': text,
                                   'truncated': len(raw) > len(text)}
                s['candidates'][eid] = {'id': eid, 'source': source, 'preview': text[:200]}
                ids.append(eid)
            s['trace'].append({'action': 'SEARCH', 'parameters': {'query': query}, 'result_ids': ids})
            return s

        @guarded
        def initial(s):
            return search(s, s['question'])

        @guarded
        def model(s):
            if s['remaining']['actions'] <= 0:
                return self._stop(s, 'action_budget')
            seconds = lim.seconds - s['active_seconds']
            if seconds <= 0:
                return self._stop(s, 'time_budget')
            s['remaining']['seconds'] = round(seconds, 6)
            view = {k: s[k] for k in ('question', 'scope', 'candidates', 'evidence',
                                      'gaps', 'calculations', 'remaining', 'review_feedback')}
            view['called_parameters'] = [{'action': e['action'], 'parameters': e['parameters']}
                                         for e in s['trace']]
            payload = encoded(view)
            prompt_charge = len((SYSTEM + payload).encode('utf8')) + 256
            charge = prompt_charge + lim.output_tokens
            if charge > s['remaining']['charged_tokens']:
                return self._stop(s, 'token_budget')
            if charge > min(lim.context_tokens, getattr(self.controller, 'context_tokens', lim.context_tokens)):
                return self._stop(s, 'context_budget')
            timeout = min(seconds, lim.call_seconds)
            result = self._operation(conn, s, 'model', lambda: self.controller.next_action(
                SYSTEM, payload, lim.output_tokens, timeout))
            s['active_seconds'] += result['seconds']
            s['remaining']['actions'] -= 1
            s['remaining']['charged_tokens'] -= charge
            s['usage']['controller_calls'] += 1
            raw, usage = result['value']
            if type(usage) is not dict:
                raise InvalidAction('Invalid usage metadata')
            if 'prompt_tokens' in usage and 'output_tokens' in usage:
                s['usage']['calls_with_reported_usage'] += 1
            for key in ('prompt_tokens', 'output_tokens'):
                value = usage.get(key, 0)
                if type(value) is not int or value < 0:
                    raise InvalidAction('Invalid usage metadata')
                s['usage']['reported_' + key] += value
            if usage.get('output_tokens', 0) > lim.output_tokens or usage.get('prompt_tokens', 0) > prompt_charge:
                return self._stop(s, 'token_budget')
            if result['seconds'] > timeout or s['active_seconds'] > lim.seconds:
                return self._stop(s, 'time_budget')
            a = decode_action(raw)
            signature = digest([a['action'], a['parameters'], s['retrieval_round']])
            if signature in s['calls']:
                raise InvalidAction('Repeated action')
            s['calls'].append(signature)
            s['action'] = a
            s['gaps'] = a['gaps']
            return s

        @guarded
        def execute(s):
            a = s.pop('action')
            kind, p = a['action'], a['parameters']
            s['trace'].append(copy.deepcopy(a))
            if kind == 'SEARCH':
                keys(p, ['query'])
                string(p['query'], 400)
                if not s['gaps']:
                    raise InvalidAction('SEARCH requires a concrete gap')
                if not s['remaining']['searches']:
                    return self._stop(s, 'search_budget')
                s['remaining']['searches'] -= 1
                return search(s, p['query'])
            if kind == 'READ':
                keys(p, ['ids'])
                ids = sequence(p['ids'], 3)
                if any(type(i) is not str for i in ids) or len(set(ids)) != len(ids):
                    raise InvalidAction('READ requires distinct string IDs')
                for eid in ids:
                    if eid not in s['known'] or eid in s['evidence']:
                        raise InvalidAction('READ requires an unread retrieved ID')
                for eid in ids:
                    s['evidence'][eid] = copy.deepcopy(s['known'][eid])
            elif kind == 'CALCULATE':
                s['calculations'].append(calculate(p, s['evidence'], s['question']))
            elif kind == 'ANSWER':
                keys(p, ['claims'])
                if s['gaps']:
                    raise InvalidAction('Cannot answer with unresolved gaps')
                s['draft'] = validate_claims(p['claims'], s['evidence'])
                s['status'] = 'draft'
            elif kind == 'ABSTAIN':
                keys(p, ['reason'])
                string(p['reason'], 600)
                if not s['gaps']:
                    raise InvalidAction('ABSTAIN requires a concrete gap')
                s['abstention'] = p['reason']
                return self._stop(s, 'abstained')
            else:
                raise InvalidAction('Unknown action')
            return s

        @guarded
        def prepare(s):
            s['version'] += 1
            s['status'] = 'awaiting_review'
            binding = {k: s[k] for k in ('thread_id', 'question', 'profile', 'version',
                                        'draft', 'evidence', 'calculations')}
            s['approval'] = {'approval_id': digest(binding), 'version': s['version'],
                             'trigger': 'Every ANSWER draft and every human edit requires review',
                             'draft': copy.deepcopy(s['draft']),
                             'evidence': copy.deepcopy(s['evidence']),
                             'calculations': copy.deepcopy(s['calculations']),
                             'actions': ['approve', 'edit', 'retrieve', 'reject'],
                             'warning': 'Exact quotes are checked mechanically; entailment, jurisdiction, dates and missing conditions require review.'}
            return s

        def review(s):
            # No side effects before interrupt: LangGraph restarts this node.
            decision = interrupt(copy.deepcopy(s['approval']))
            return {**s, 'decision': decision}

        @guarded
        def apply_review(s):
            d = s['decision']
            self._validate_review(s, d)
            s['reviews'].append({'request': copy.deepcopy(d), 'draft_sha256': digest(s['draft'])})
            s['decision'] = None
            s['approval'] = None
            if d['action'] == 'approve':
                s.update(status='approved', stop_reason='human_approved', answer=copy.deepcopy(s['draft']))
            elif d['action'] == 'reject':
                s.update(status='rejected', stop_reason='human_rejected', answer=None)
            elif d['action'] == 'edit':
                s.update(status='draft', draft=copy.deepcopy(d['claims']), answer=None)
            else:
                s.update(status='retrieving', draft=None, answer=None,
                         review_feedback={'query': d['query'], 'gap': d['gap']},
                         gaps=[d['gap']], retrieval_round=s['retrieval_round'] + 1)
            return s

        @guarded
        def retrieve_again(s):
            s['remaining']['searches'] -= 1
            s['status'] = 'running'
            return search(s, s['review_feedback']['query'])

        def after_execute(s):
            return END if s['status'] == 'stopped' else 'prepare' if s['status'] == 'draft' else 'model'

        def after_review(s):
            return {'draft': 'prepare', 'retrieving': 'retrieve_again'}.get(s['status'], END)

        graph = StateGraph(dict)
        for name, fn in [('initial', initial), ('model', model), ('execute', execute),
                         ('prepare', prepare), ('review', review), ('apply_review', apply_review),
                         ('retrieve_again', retrieve_again)]:
            graph.add_node(name, fn)
        graph.add_edge(START, 'initial')
        for name in ('initial', 'retrieve_again'):
            graph.add_conditional_edges(name, lambda s: END if s['status'] == 'stopped' else 'model')
        graph.add_conditional_edges('model', lambda s: END if s['status'] == 'stopped' else 'execute')
        graph.add_conditional_edges('execute', after_execute)
        graph.add_conditional_edges('prepare', lambda s: END if s['status'] == 'stopped' else 'review')
        graph.add_edge('review', 'apply_review')
        graph.add_conditional_edges('apply_review', after_review)
        return graph.compile(checkpointer=saver)
