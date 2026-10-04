import pytest
from hashlib import sha256
from copy import deepcopy
from src.insurerag_vlm.holdout_containment import evidence_windows, scan_history, verify_evidence_projection


def words(count, prefix='token'):
    return ' '.join(f'{prefix}{n}' for n in range(count))


def test_source_projection_must_match_document_offsets_and_hashes():
    text='Prefix. The synthetic policy rule applies. Footer.'
    start,end=8,41
    digest=lambda value:sha256(value.encode()).hexdigest()
    docs={'doc':{'text':text,'normalized_text_sha256':digest(text)}}
    row={'id':'evidence','parent_id':'case','answer':text[start:end],
         'evidence':{'document_id':'doc','start':start,'end':end,'sha256':digest(text[start:end])}}
    assert verify_evidence_projection([row],docs)==1
    edited=deepcopy(row);edited['answer']='An authored paraphrase that is not the source evidence.'
    with pytest.raises(ValueError,match='source span'):verify_evidence_projection([edited],docs)
    bad=deepcopy(row);bad['evidence']['start']=-1
    with pytest.raises(ValueError,match='offsets'):verify_evidence_projection([bad],docs)
    with pytest.raises(ValueError,match='requires parent_id'):verify_evidence_projection([{'id':'x','answer':text}],docs)
    docs['doc']['text']='Source changed'
    with pytest.raises(ValueError,match='document hash'):verify_evidence_projection([row],docs)


def test_finds_evidence_inside_long_record_without_cosine_dilution():
    evidence = words(24)
    index = evidence_windows([{'id': 'case', 'answer': evidence}])
    report = scan_history(index, [{'text': words(300, 'before') + ' ' + evidence + ' ' + words(300, 'after'), 'origin': 'synthetic-long-record'}])
    assert report['flagged_candidate_ids'] == ['case']
    assert report['matched_windows'] == 9
    assert report['coverage'][0]['token_coverage_across_history'] == 1
    assert {m['origin'] for m in report['matches']} == {'synthetic-long-record'}


def test_does_not_join_different_historical_records_or_partial_tokens():
    evidence = words(16)
    index = evidence_windows([{'id': 'case', 'answer': evidence}])
    pieces = evidence.split()
    report = scan_history(index, [{'text': ' '.join(pieces[:8])}, {'text': ' '.join(pieces[8:])}, {'text': evidence.replace('token0 ', 'xtoken0 ')}])
    assert report['matched_windows'] == 0
    assert report['historical_strings'] == 3


def test_repaired_format_variant_finds_joined_word_but_keeps_raw_source_intact():
    raw = 'al\u200bpha bravo charlie delta echo foxtrot golf hotel india juliet'
    row = {'id': 'format', 'answer': raw}
    index = evidence_windows([row])
    report = scan_history(index, [{'text': raw.replace('\u200b', ''), 'origin': 'synthetic-clean'}])
    assert report['flagged_candidate_ids'] == ['format']
    assert any(c['token_coverage_across_history'] == 1 for c in report['coverage'])
    assert row['answer'] == raw


def test_short_evidence_requires_entire_text_and_first_origin_is_preserved():
    evidence = words(9)
    index = evidence_windows([{'id': 'short', 'answer': evidence}, {'id': 'too-short', 'answer': words(7)}])
    report = scan_history(index, [{'text': evidence, 'origin': 'first'}, {'text': evidence, 'origin': 'second'}])
    assert report['unique_windows'] == report['matched_windows'] == 1
    assert report['flagged_candidate_ids'] == ['short']
    assert report['matches'][0]['origin'] == 'first'


def test_overlap_coverage_is_not_double_counted():
    index = evidence_windows([{'id': 'partial', 'answer': words(30)}])
    report = scan_history(index, [{'text': words(20)}])
    assert report['matched_windows'] == 5
    assert report['coverage'][0]['matched_tokens_across_history'] == 20
    assert report['coverage'][0]['token_coverage_across_history'] == pytest.approx(2 / 3)


def test_duplicate_candidate_ids_fail_closed():
    with pytest.raises(ValueError, match='Duplicate candidate'):
        evidence_windows([{'id': 'same', 'answer': words(8)}] * 2)
