from copy import deepcopy
import pytest
from src.insurerag_vlm.evidence_arithmetic import verify_calculation, decimal_value

SOURCES = {'E1': {'answer_id': 'a1', 'source_group': 'ACME/2020', 'text': 'Revenue for 2020 is 120; revenue for 2019 is 100.'}}


def payload():
    return {'abstain': False, 'operation': 'percent_change', 'proposed_result': '20',
            'operands': [{'source': 'E1', 'value': '120', 'quote': 'Revenue for 2020 is 120'},
                         {'source': 'E1', 'value': '100', 'quote': 'revenue for 2019 is 100'}]}


def test_checks_arithmetic_without_claiming_semantic_verification():
    result = verify_calculation(payload(), SOURCES, 'ACME/2020')
    assert result['computed_result'] == '20.0'
    assert result['proposed_within_0_02'] and not result['semantics_verified']


def test_reports_model_math_error_without_rewriting_its_proposal():
    p = payload(); p['proposed_result'] = '35'
    result = verify_calculation(p, SOURCES, 'ACME/2020')
    assert result['computed_result'] == '20.0' and result['proposed_result'] == '35'
    assert not result['proposed_within_0_02']


@pytest.mark.parametrize('value', ['nan', 'inf', '1e3', '__import__("os")', '10000000000000000000'])
def test_rejects_nonliteral_or_unbounded_values(value):
    with pytest.raises(ValueError): decimal_value(value)


@pytest.mark.parametrize('change', ['unknown_source', 'fake_quote', 'unprinted_value', 'wrong_report', 'wrong_count', 'bad_operation'])
def test_rejects_ungrounded_or_invalid_calculations(change):
    p = payload(); scope = 'ACME/2020'
    if change == 'unknown_source': p['operands'][0]['source'] = 'E2'
    elif change == 'fake_quote': p['operands'][0]['quote'] = 'Invented revenue is 120'
    elif change == 'unprinted_value': p['operands'][0]['value'] = '12'
    elif change == 'wrong_report': scope = 'ACME/2021'
    elif change == 'wrong_count': p['operands'].pop()
    else: p['operation'] = 'execute'
    with pytest.raises(ValueError): verify_calculation(p, SOURCES, scope)


def test_mean_supplies_its_own_count_and_rejects_zero_denominator():
    p = payload(); p['operation'] = 'mean'; p['proposed_result'] = '110'
    assert verify_calculation(p, SOURCES, 'ACME/2020')['computed_result'] == '110'
    p['operation'] = 'ratio'; p['operands'][1] = {'source': 'E1', 'value': '0', 'quote': 'Revenue for 2019 is 0'}
    sources = deepcopy(SOURCES); sources['E1']['text'] += ' Revenue for 2019 is 0'
    with pytest.raises(ValueError, match='Zero denominator'): verify_calculation(p, sources, 'ACME/2020')
