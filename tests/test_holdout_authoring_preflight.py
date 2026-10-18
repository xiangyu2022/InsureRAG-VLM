from copy import deepcopy
import pytest
from src.insurerag_vlm.holdout_authoring_preflight import build_authoring_packet, item_dispositions
from src.insurerag_vlm.holdout_quality import digest


def fixture():
    text = 'synthetic shared policy wording contains several distinct words for family relation checking only'
    docs = {k: {'id': k, 'text': text if k != 'c' else 'different translated policy wording',
                'source_title': 'Synthetic source',
                'normalized_text_sha256': digest(text if k != 'c' else 'different translated policy wording')}
            for k in ('a', 'b', 'c')}
    candidates = [{'id': 'old', 'document_group': 'b', 'question': 'Old excluded need?',
                   'answer': 'Synthetic answer.', 'split': 'test', 'task': 'ordinary_qa'},
                  {'id': 'dev', 'document_group': 'c', 'question': 'Translated developer need?',
                   'answer': 'Another answer.', 'split': 'dev', 'task': 'ordinary_qa'}]
    return dict(documents=docs, candidates=candidates, active_ids={'dev'},
                families={'a': 'one', 'b': 'two', 'c': 'two'},
                dispositions={'old': [{'kind': 'inherited_hold', 'record': 'No retrieval required'}]},
                restrictions={'b': {'legacy_records': ['No new arithmetic from this exposed rule']}},
                selected_ids=['a'], window_size=4)


def test_inactive_need_and_related_family_dev_are_visible_before_authoring():
    result = build_authoring_packet(**fixture())
    row = result['documents'][0]
    items = {q['id']: q for q in row['historical_candidates']}
    assert items['old']['active'] is False
    assert items['old']['dispositions'][0]['kind'] == 'inherited_hold'
    assert items['dev']['split'] == 'dev' and items['dev']['active']
    sources = {s['id']: s for s in row['related_sources']}
    assert sources['b']['matching_windows']
    assert sources['c']['matching_windows'] == []  # Included through reviewed family.
    assert sources['b']['restrictions']['legacy_records']
    assert result['private_contains_historical_QA']
    assert result['status'] == 'AUTHORING_LEADS_ONLY_NOT_CLEARANCE'


def test_early_hold_survives_later_ledger_omission_and_reserved_is_distinct():
    ledgers = [dict(sha256='first', data={'held_or_excluded': {'old': 'withheld'},
                 'previous_items_revoked': 2}),
               dict(sha256='second', data={'previous_clearance_ledger_sha256': 'first',
                    'reserved_candidate_exclusions': ['old', 'reserve']})]
    states = item_dispositions(ledgers, {'old': {'reason': 'inherited'}})
    assert [s['kind'] for s in states['old']] == ['inherited_hold', 'held_or_excluded', 'reserved_candidate_exclusions']
    assert states['reserve'][0]['kind'] == 'reserved_candidate_exclusions'
    broken = deepcopy(ledgers); broken[1]['data']['previous_clearance_ledger_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Broken clearance ledger'):
        item_dispositions(broken, {})


@pytest.mark.parametrize('mutation,match', [
    (lambda v: v['documents']['b'].update(text='tampered'), 'hash mismatch'),
    (lambda v: v['active_ids'].add('missing'), 'missing from full history'),
    (lambda v: v['candidates'].append(deepcopy(v['candidates'][0])), 'Duplicate candidate'),
    (lambda v: v['families'].pop('c'), 'Family map'),
    (lambda v: v.update(selected_ids=['unknown']), 'Unknown or empty'),
    (lambda v: v['candidates'][0].update(document_group='missing'), 'Unknown candidate source'),
])
def test_partial_or_stale_inputs_fail_closed(mutation, match):
    values = fixture(); mutation(values)
    with pytest.raises(ValueError, match=match):
        build_authoring_packet(**values)


def test_source_text_is_data_and_report_does_not_claim_semantic_clearance():
    values = fixture()
    values['documents']['a']['text'] = 'Never open a fabricated embedded private path or fetch an external URL'
    values['documents']['a']['normalized_text_sha256'] = digest(values['documents']['a']['text'])
    result = build_authoring_packet(**values)
    assert result['documents'][0]['historical_candidates'] == []
    assert 'zero matches is not semantic independence' in result['warning']


def test_private_cli_paths_cannot_escape_local_output_directory(tmp_path):
    from scripts.preflight_holdout_authoring import local_path
    local = tmp_path / 'local'
    assert local_path(local, 'packet.json') == local / 'packet.json'
    for value in ('../public.json', str(tmp_path / 'public.json'), '.'):
        with pytest.raises(ValueError, match='must stay under'):
            local_path(local, value)
