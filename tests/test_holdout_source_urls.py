import copy
import hashlib
import json

import pytest

from scripts.preflight_holdout_source_urls import check_report_binding, screen_source_urls, url_requests
from scripts.prepare_source_faq import norm


def source(ident, url):
    return {'id': ident, 'publisher': 'example_agency', 'source_url': url}


def history(tmp_path, texts):
    path = tmp_path / 'history.jsonl'
    path.write_text(''.join(json.dumps({'text': norm(text), 'origin': 'fixture'}) + '\n' for text in texts), encoding='utf8')
    return path


def test_shared_url_aliases_and_overlapping_paths_all_receive_exposure(tmp_path):
    documents = [source('parent', 'https://example.gov/guide'),
                 source('child', 'https://example.gov/guide/claims'),
                 source('alias', 'http://www.example.gov/guide/claims'),
                 source('unseen', 'https://example.gov/other')]
    path = history(tmp_path, ['Only a reference: https://www.example.gov/guide/claims'])
    report = screen_source_urls(documents, path)
    hits = {r['id']: r['excluded_from_independent_holdout'] for r in report['rows']}
    assert hits == {'alias': True, 'child': True, 'parent': True, 'unseen': False}
    assert report['excluded_sources'] == 3 and report['accepted_test_items'] == 0


def test_first_hit_offsets_recover_the_bound_record_without_opening_origin(tmp_path):
    path = history(tmp_path, ['unrelated', 'Reference https://example.gov/policy', 'https://example.gov/policy again'])
    report = screen_source_urls([source('s', 'https://example.gov/policy')], path)
    hit = report['rows'][0]['historical_url_hit']
    with path.open('rb') as handle:
        handle.seek(hit['byte_offset'])
        record = json.loads(handle.read(hit['byte_length']))
    assert hit['history_row_index'] == 1
    assert hashlib.sha256(record['text'].encode()).hexdigest() == hit['historical_text_sha256']
    assert record['text'][hit['match_start']:hit['match_end']] == 'example gov policy'


def test_cache_binding_rejects_changed_url_history_or_code_but_not_source_body(tmp_path):
    documents = [source('s', 'https://example.gov/policy')]
    path = history(tmp_path, ['no source reference'])
    report = screen_source_urls(documents, path)
    history_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    check_report_binding(report, [{**documents[0], 'text': 'changed source body'}], history_sha)
    with pytest.raises(ValueError, match='Stale'):
        check_report_binding(report, [source('s', 'https://example.gov/different')], history_sha)
    with pytest.raises(ValueError, match='Stale'):
        check_report_binding(report, documents, '0' * 64)
    changed = copy.deepcopy(report)
    changed['configuration']['normalization_code_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Stale'):
        check_report_binding(changed, documents, history_sha)


def test_cache_binding_rejects_missing_rows_or_acceptance_claim(tmp_path):
    documents = [source('s', 'https://example.gov/policy')]
    report = screen_source_urls(documents, history(tmp_path, ['no hit']))
    changed = copy.deepcopy(report)
    changed['rows'] = []
    with pytest.raises(ValueError, match='rows'):
        check_report_binding(changed, documents, report['historical_sha256'])
    changed = copy.deepcopy(report)
    changed['accepted_test_items'] = 1
    with pytest.raises(ValueError, match='cannot accept'):
        check_report_binding(changed, documents, report['historical_sha256'])


@pytest.mark.parametrize('url', ['file:///private', 'https://user:pass@example.gov/policy', 'not a URL'])
def test_nonpublic_or_invalid_urls_fail_closed(url):
    with pytest.raises(ValueError, match='public HTTP'):
        url_requests([source('s', url)])


def test_duplicate_ids_fail_instead_of_overwriting():
    with pytest.raises(ValueError, match='Duplicate'):
        url_requests([source('s', 'https://example.gov/a'), source('s', 'https://example.gov/b')])
