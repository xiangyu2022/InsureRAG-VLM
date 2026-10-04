"""Document a post-test source extraction repair without changing the benchmark."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.extract_government_qa_boundaries_v2 import extract


def main():
    fixture = ROOT / 'data/benchmarks/condition_v1'
    before = {p.name: sha(p) for p in fixture.glob('*.json*')}
    cases = read_jsonl(fixture / 'test.jsonl')
    case = next(c for c in cases if c['id'] == 'condgov_q_72f9a4e449c7fa4f')
    records = ROOT / 'data/research_corpus/hicric_public_v1/records.jsonl'
    record = next(r for r in read_jsonl(records) if r['id'] == case['source_record_id'])
    answers = {a['id']: a for a in read_jsonl(fixture / 'answers.jsonl')}
    old = answers[case['gold_answer_ids'][0]]
    pairs, rejections = extract(record['text'])
    candidates = [p for p in pairs if p['question'] == case['question']]
    assert len(candidates) == 1
    repaired = candidates[0]
    assert repaired['question_span'] == case['question_span']
    assert repaired['answer_span'][0] == case['answer_span'][0]
    assert repaired['answer_span'][1] > case['answer_span'][1]
    assert repaired['text'].startswith(old['text'])
    assert 'where the provider has no recourse' in repaired['text']
    assert '(See Q-A3, Q-A4, and Q-A5.)' in repaired['text']
    screening = {}
    for split in ['train_additions', 'test']:
        government = [c for c in read_jsonl(fixture / f'{split}.jsonl') if c['domain'] == 'government']
        flagged = []
        for c in government:
            text = answers[c['gold_answer_ids'][0]]['text']
            reasons = []
            if text.count('(') != text.count(')'):
                reasons.append('unbalanced_parentheses')
            if re.search(r'\b(?:and|or|see)\s*$', text, re.I):
                reasons.append('unfinished_clause')
            if reasons:
                flagged.append({'id': c['id'], 'source_record_id': c['source_record_id'],
                                'source_url': c['source_url'], 'reasons': reasons, 'answer_tail': text[-300:]})
                if c['source_record_id'] == record['id']:
                    matching = [p for p in pairs if p['question'] == c['question']]
                    flagged[-1]['v2_matching_pair_count'] = len(matching)
                    flagged[-1]['v2_answer_span'] = matching[0]['answer_span'] if matching else None
        screening[split] = {'government_questions': len(government), 'flagged': flagged,
                            'flagged_count': len(flagged)}
    assert before == {p.name: sha(p) for p in fixture.glob('*.json*')}
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Post-test diagnostic and future-parser repair only',
        'case_id': case['id'], 'source_record_id': record['id'],
        'source_url': case['source_url'], 'source_text_sha256': record['content_sha256'],
        'root_cause': 'The old line-anchored question regex misreads wrapped cross-reference Q-A5.) as a new question.',
        'old_answer_span': case['answer_span'], 'old_answer_words': len(old['text'].split()),
        'repaired_preview': repaired, 'repaired_answer_words': len(repaired['text'].split()),
        'source_parser_accepted_pairs': len(pairs), 'source_parser_rejections': rejections,
        'all_added_government_boundary_screen': screening,
        'screening_interpretation': 'Heuristic flags require source review; unflagged does not mean correct. Three flagged test cases have wrapped cross-reference boundaries in the same claims-procedure source. A fourth has unmatched parentheses; its cause is not adjudicated. The v2 parser repairs the demonstrated case and one other but conservatively rejects a case with a page-number-prefixed next heading. No cases are removed or rescored.',
        'repair_parser_sha256': sha(ROOT / 'scripts/extract_government_qa_boundaries_v2.py'),
        'diagnostic_code_sha256': sha(Path(__file__)),
        'frozen_fixture_hashes_before_and_after': before,
        'frozen_fixture_modified': False, 'rescored': False, 'used_in_training': False,
        'limitations': 'Repair was designed after inspecting this test case. It is a parser regression fixture, not a new blind evaluation. The new parser handles explicit Q/A only, has not been deployed to the frozen data, and still needs broader source and expert review.',
    }
    write_json(report, ROOT / 'reports/condition_listwise_v1/boundary_repair_diagnostic.json')
    print(json.dumps({k: report[k] for k in ['status', 'old_answer_words', 'repaired_answer_words', 'frozen_fixture_modified', 'rescored']}))


if __name__ == '__main__':
    main()
