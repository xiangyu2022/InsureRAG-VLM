from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2
from scripts import prepare_research_benchmark_v2 as prepare
from scripts import eval_research_benchmark_v2 as evaluate
from scripts import compare_research_runs_v2 as compare


def write_rows(path, rows):
    path.write_text(''.join(v2.canonical(row) + '\n' for row in rows), encoding='utf-8')


def fixture(tmp_path):
    root = tmp_path / 'fixture'
    corpus = root / 'corpus'
    corpus.mkdir(parents=True)
    documents, pages, snippets, rows = [], [], [], []
    for index, doc in enumerate(('dev_a', 'dev_b', 'test_a', 'test_b')):
        split = doc.split('_')[0]
        source = doc + '.pdf#page=1'
        text = f'Archived document {doc} states a collision deductible of $125. This particular deductible applies to the named collision coverage, not to any unrelated coverage or general premium.'
        documents.append({'doc_id': doc, 'document_family_id': doc, 'split': split, 'title': doc})
        page = {'record_id': 'page::' + doc, 'record_type': 'page', 'doc_id': doc, 'page': 1,
                'source_file': doc + '.pdf', 'citation': source, 'text': text}
        pages.append(page)
        snippets.append({**page, 'record_id': 'snippet::' + doc, 'record_type': 'snippet'})
        rows.append({'id': 'case_' + doc, 'fact_id': 'fact_' + doc, 'document_family_id': doc,
            'category': 'deductible', 'split': split, 'document_scope': [doc],
            'question': f'According to archived document {doc}, what collision deductible is stated?',
            'answerable': True, 'reference_answer': '$125', 'answer_keys': [['125']],
            'gold': [{'source': source, 'page': 1, 'evidence_span': f'Archived document {doc} states a collision deductible of $125.'}]})
    for split in ('dev', 'test'):
        write_rows(root / (split + '.jsonl'), [row for row in rows if row['split'] == split])
    v1.write_json(documents, root / 'documents.json')
    write_rows(corpus / 'rag_pages.jsonl', pages)
    write_rows(corpus / 'rag_snippets.jsonl', snippets)
    freeze(root)
    return root, corpus


def freeze(root):
    v1.write_json({'counts': {'dev': 2, 'test': 2},
        'files': {name: v1.sha(root / name) for name in ('dev.jsonl', 'test.jsonl', 'documents.json')},
        'corpus_files': {'corpus/' + name: v1.sha(root / 'corpus' / name) for name in ('rag_pages.jsonl', 'rag_snippets.jsonl')}},
        root / 'manifest.lock.json')


def verify(root, corpus):
    return v2.verify_fixture(root, corpus, expected_counts={'dev': 2, 'test': 2})


@pytest.mark.parametrize('problem', ['family', 'fact', 'question', 'duplicate_page', 'source', 'span', 'category', 'count'])
def test_fixture_rejects_leakage_duplicates_and_invalid_annotations(tmp_path, problem):
    root, corpus = fixture(tmp_path)
    rows = v1.read_jsonl(root / 'test.jsonl')
    if problem == 'family':
        docs = json.loads((root / 'documents.json').read_text())
        docs[2]['document_family_id'] = 'dev_a'
        v1.write_json(docs, root / 'documents.json')
        rows[0]['document_family_id'] = 'dev_a'
    elif problem == 'fact':
        rows[0]['fact_id'] = 'fact_dev_a'
    elif problem == 'question':
        rows[0]['question'] = v1.read_jsonl(root / 'dev.jsonl')[0]['question'].upper()
    elif problem == 'duplicate_page':
        pages = v1.read_jsonl(corpus / 'rag_pages.jsonl')
        pages[2]['text'] = pages[0]['text']
        write_rows(corpus / 'rag_pages.jsonl', pages)
    elif problem == 'source':
        rows[0]['gold'][0]['source'] = 'dev_a.pdf#page=1'
    elif problem == 'span':
        rows[0]['gold'][0]['evidence_span'] = 'An absent supported statement.'
    elif problem == 'category':
        rows[0]['category'] = ''
    else:
        rows.pop()
    write_rows(root / 'test.jsonl', rows)
    freeze(root)
    with pytest.raises(ValueError):
        verify(root, corpus)


def test_manifest_rejects_changed_bytes_without_relabeling(tmp_path):
    root, corpus = fixture(tmp_path)
    verify(root, corpus)
    path = corpus / 'rag_pages.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Frozen corpus'):
        verify(root, corpus)


def test_gold_span_must_reach_its_actual_source_section_and_generation_must_finish(tmp_path):
    root, corpus = fixture(tmp_path)
    _, items, pages, _, _ = verify(root, corpus)
    item = items[0]
    quote = item['gold'][0]['evidence_span']
    source = item['gold'][0]['source']
    raw = json.dumps({'answer': '$125', 'evidence': quote, 'source': source, 'abstain': False})
    prompt = v1.PROMPT_TEMPLATE.format(context='SOURCE: ' + source + '\n' + quote, question=item['question'])
    kwargs = {'prompt': prompt, 'generation': {'done': True, 'done_reason': 'stop'}}
    score = v2.score_response(raw, item, {p['citation']: p for p in pages}, [source], **kwargs)
    assert score['answered'] and score['context_grounded_key_pass'] and score['gold_evidence_in_context']
    wrong_context = prompt.replace('SOURCE: ' + source, 'SOURCE: unrelated.pdf#page=1')
    score = v2.score_response(raw, item, {p['citation']: p for p in pages}, [source], prompt=wrong_context, generation=kwargs['generation'])
    assert score['retrieval_hit'] and not score['gold_evidence_in_context'] and not score['context_grounded_key_pass']
    score = v2.score_response(raw, item, {p['citation']: p for p in pages}, [source], prompt=prompt, generation={'done': True, 'done_reason': 'length'})
    assert score['gold_evidence_in_context'] and not score['answered'] and not score['context_grounded_key_pass']


def row(identifier, doc, *, supported=True, answered=False, correct=False, decline=False, error=False):
    return {'id': identifier, 'mode': 'retrieved', 'document_id': doc, 'document_family_id': doc,
        'category': 'example', 'answerable': supported, 'error': 'Timeout' if error else None,
        'scores': {'score_version': v2.SCORE_VERSION, 'answered': answered,
                   'context_grounded_key_pass': correct, 'completed_strict_abstention': decline,
                   'generation_complete': not error, 'valid_json': not error,
                   'gold_evidence_in_context': supported, 'retrieval_hit': supported}}


def test_coverage_conditional_contract_and_unsupported_denominators_are_distinct():
    rows = [row('a', 'a', answered=True, correct=True), row('b', 'b', decline=True),
            row('c', 'c', supported=False, answered=True), row('d', 'd', supported=False, decline=True),
            row('e', 'e', supported=False, error=True)]
    table = v2.metric_table(rows, bootstrap=100)
    assert table['supported_context_contract_success']['rate'] == 0.5
    assert table['supported_answer_coverage']['rate'] == 0.5
    assert table['conditional_supported_contract_accuracy']['rate'] == 1
    assert table['answer_coverage']['rate'] == 2 / 5
    assert table['conditional_answer_contract_precision']['rate'] == 0.5
    assert table['unsupported_strict_abstention']['rate'] == 1 / 3
    assert table['unsupported_failure_to_strictly_abstain']['rate'] == 2 / 3
    assert table['unsupported_answer_rate']['rate'] == 1 / 3
    assert table['request_error_rate']['rate'] == 1 / 5


def test_bootstrap_samples_documents_not_individual_questions_and_is_seeded():
    rows = [row('a1', 'a', answered=True, correct=True), row('a2', 'a', answered=True, correct=True), row('b1', 'b')]
    result = v2.ratio_summary(rows, 'supported_context_contract_success', bootstrap=500, seed=19)
    assert result == v2.ratio_summary(rows, 'supported_context_contract_success', bootstrap=500, seed=19)
    assert result['rate'] == 2 / 3
    assert result['clusters'] == 2
    assert (result['ci95_low'], result['ci95_high']) == (0, 1)
    empty = v2.ratio_summary(rows, 'unsupported_strict_abstention')
    assert empty['rate'] is None and empty['ci95_low'] is None
    one = v2.ratio_summary(rows[:2], 'supported_context_contract_success')
    assert one['ci95_low'] is None


def test_paired_bootstrap_preserves_pairing_and_variable_conditional_denominators():
    left = [row('a', 'a', answered=True, correct=True), row('b', 'b', answered=True)]
    same = v2.paired_metric(left, deepcopy(left), 'supported_context_contract_success', bootstrap=100)
    assert same['right_minus_left_percentage_points'] == 0
    assert same['ci95_low_pp'] == same['ci95_high_pp'] == 0
    right = [row('a', 'a', answered=True, correct=True), row('b', 'b', decline=True)]
    conditional = v2.paired_metric(left, right, 'conditional_supported_contract_accuracy', bootstrap=100)
    assert conditional['left_total'] == 2 and conditional['right_total'] == 1
    assert conditional['right_minus_left_percentage_points'] == 50
    assert conditional['ci95_low_pp'] is None


def prepared_fixture(tmp_path, monkeypatch):
    root, corpus = fixture(tmp_path)
    original = v2.verify_fixture
    monkeypatch.setattr(v2, 'verify_fixture', lambda folder, source: original(folder, source, expected_counts={'dev': 2, 'test': 2}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation must not call a model or hosted API')
    monkeypatch.setattr(v1.requests, 'post', forbidden)
    monkeypatch.setattr(v1.requests, 'get', forbidden)
    output = tmp_path / 'prepared'
    args = prepare.parser().parse_args(['--benchmark', str(root), '--split', 'test', '--output', str(output), '--bootstrap', '100'])
    prepare.prepare(args)
    return output


def test_cpu_preparation_freezes_actual_prompts_and_rejects_mutation(tmp_path, monkeypatch):
    output = prepared_fixture(tmp_path, monkeypatch)
    protocol, inputs, _ = v2.load_prepared(output, root=prepare.ROOT)
    assert protocol['preparation_generation_calls'] == 0
    assert len(inputs) == 2 and all(row['mode'] == 'retrieved' for row in inputs)
    assert all(row['gold_evidence_in_context'] for row in inputs)
    assert all(row['prompt'].startswith('Context:\nSOURCE: ') for row in inputs)
    path = output / 'inputs.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Frozen prepared input'):
        v2.load_prepared(output, root=prepare.ROOT)


def fake_client(inputs, *, digest='digest', fail_second=True):
    class Client:
        def __init__(self, args):
            self.args = args
            self.index = 0
            self.model_info = {'digest': digest}
            self.options = {'temperature': 0, 'seed': 42, 'num_ctx': args.num_ctx, 'num_predict': args.num_predict,
                            'top_k': 40, 'top_p': 1.0, 'repeat_penalty': 1.0, 'presence_penalty': 0.0}
            self.last_generation_metadata = {}
            self.last_prompt = None

        def backend_metadata(self):
            return {'model_digest': digest, 'resolved_model': self.args.model, 'generation_options': self.options, 'thinking': False}

        def generate(self, prompt):
            item = inputs[self.index]['case']
            self.index += 1
            self.last_prompt = prompt
            if fail_second and self.index == 2:
                raise TimeoutError('constructed transport failure')
            self.last_generation_metadata = {'done': True, 'done_reason': 'stop'}
            return json.dumps({'answer': '$125', 'evidence': item['gold'][0]['evidence_span'],
                               'source': item['gold'][0]['source'], 'abstain': False})
    return Client


def test_runner_keeps_failed_calls_and_comparator_requires_matching_failed_prompts(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    _, inputs, _ = v2.load_prepared(prepared, root=prepare.ROOT)
    monkeypatch.setattr(evaluate.v1, 'LocalOllamaClient', fake_client(inputs))
    run = tmp_path / 'run'
    args = evaluate.parser().parse_args(['--prepared', str(prepared), '--model', 'mock:model',
        '--expected-digest', 'digest', '--output', str(run)])
    metadata = evaluate.run(args)
    assert metadata['status'] == 'complete' and metadata['request_errors'] == 1
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['modes']['retrieved']['overall']['supported_context_contract_success']['rate'] == 0.5
    left, rows = compare.read_run(run)
    right = deepcopy(left)
    right['model']['model_digest'] = 'other-model-digest'
    compare.validate_matched(left, right, rows, deepcopy(rows))
    changed = deepcopy(rows)
    changed[1]['prompt'] += ' changed even though this call failed'
    with pytest.raises(ValueError, match='Unmatched actual'):
        compare.validate_matched(left, right, rows, changed)
    with pytest.raises(ValueError, match='complete frozen run'):
        args.resume = True
        evaluate.run(args)


def test_runner_rejects_wrong_exact_digest_without_creating_run(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    _, inputs, _ = v2.load_prepared(prepared, root=prepare.ROOT)
    monkeypatch.setattr(evaluate.v1, 'LocalOllamaClient', fake_client(inputs, digest='installed-other'))
    output = tmp_path / 'wrong-model'
    args = evaluate.parser().parse_args(['--prepared', str(prepared), '--model', 'mock:model',
        '--expected-digest', 'wanted-digest', '--output', str(output)])
    with pytest.raises(RuntimeError, match='explicitly requested artifact'):
        evaluate.run(args)
    assert not output.exists()


def test_partial_test_preparation_is_prohibited_before_inference(tmp_path):
    args = prepare.parser().parse_args(['--benchmark', str(tmp_path), '--split', 'test', '--limit', '1'])
    with pytest.raises(ValueError, match='only for a development'):
        prepare.prepare(args)


def test_original_source_hashes_are_checked_even_when_curated_text_is_unchanged(tmp_path):
    root, corpus = fixture(tmp_path)
    (root / 'sources').mkdir()
    original = root / 'sources/example.pdf'
    original.write_bytes(b'An opaque test source; this is a hash check, not a PDF parser test.')
    lock = json.loads((root / 'manifest.lock.json').read_text())
    lock['source_files'] = {'sources/example.pdf': v1.sha(original)}
    v1.write_json(lock, root / 'manifest.lock.json')
    verify(root, corpus)
    original.write_bytes(b'Changed source bytes')
    with pytest.raises(ValueError, match='original source file'):
        verify(root, corpus)
