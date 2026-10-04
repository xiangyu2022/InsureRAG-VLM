from scripts.extract_government_qa_boundaries_v2 import extract


ANSWER = (
    'No, provided that the dispute does not affect a claimant right to benefits. '
    'This paragraph explains the applicable scope and preserves all conditions '
    'needed to understand which parties and contractual arrangements are covered.'
)


def test_wrapped_reference_does_not_end_an_answer():
    source = (
        'Q-A8: Do these requirements apply to contractual payment disputes?\n'
        'A: ' + ANSWER + ' (See Q-A3, Q-A4, and\nQ-A5.) '
        'The provider must have no recourse against the claimant.\n'
        'Q-A9: How do the rules apply to a later claim?\nA: ' + ANSWER
    )
    pairs, _ = extract(source)
    assert len(pairs) == 2
    assert '(See Q-A3, Q-A4, and Q-A5.)' in pairs[0]['text']
    assert pairs[0]['text'].endswith('no recourse against the claimant.')
    assert 'later claim' not in pairs[0]['text']
    for pair in pairs:
        for field in ['question', 'answer']:
            start, end = pair[field + '_span']
            assert ' '.join(source[start:end].split()) == pair['question' if field == 'question' else 'text']


def test_numbered_reference_without_answer_marker_is_not_a_boundary():
    source = 'Q1: When do the described requirements take effect?\nA1: ' + ANSWER
    source += '\nQ5. This is a cross reference, not a question. Additional explanation follows.'
    pairs, rejected = extract(source)
    assert len(pairs) == 1
    assert pairs[0]['text'].endswith('Additional explanation follows.')
    assert rejected['no_explicit_answer'] == 1


def test_mismatched_question_and_answer_numbers_are_rejected():
    pairs, rejected = extract('Question 12: When do these rules take effect for participants?\nAnswer 13: ' + ANSWER)
    assert pairs == []
    assert rejected['number_mismatch'] == 1


def test_unfinished_cross_reference_is_rejected():
    pairs, rejected = extract('Q1: When do these rules take effect for participants?\nA: ' + ANSWER + ' (See Q-A3 and')
    assert pairs == []
    assert rejected['unfinished_reference_or_clause'] == 1


def test_unmarked_next_question_does_not_get_absorbed_as_valid_answer():
    pairs, rejected = extract('Q1: When do these rules take effect for participants?\nA: ' + ANSWER + '\nWhat happens under a different plan?')
    assert pairs == []
    assert rejected['ambiguous_answer'] == 1


def test_parser_requires_explicit_answer_marker():
    pairs, rejected = extract('Q1: When do these rules take effect for participants?\n' + ANSWER)
    assert pairs == []
    assert rejected['no_explicit_answer'] == 1
