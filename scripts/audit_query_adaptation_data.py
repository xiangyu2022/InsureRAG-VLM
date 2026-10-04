"""Verify publisher labels, exclusion masks, and prior-reranker question exposure."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.prepare_reranker_training import normalize

INSURANCE = re.compile(r'\b(?:insurance|insured|insurer|insurers|premium|premiums|deductible|deductibles|copay|medicare|medicaid|annuity|annuities|coverage)\b', re.I)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    data = ROOT / 'data/training/query_adaptation_v1'
    fixture = ROOT / 'data/benchmarks/fiqa_v1'
    raw = ROOT / '../fiqa_download'
    for d in [data, fixture]:
        m = json.loads((d / 'manifest.lock.json').read_text(encoding='utf8'))
        for name, digest in m['files'].items(): assert sha(d / name) == digest
    answers = read_jsonl(data / 'answers.jsonl')
    lookup = {a['id']: a for a in answers}
    groups = read_jsonl(data / 'train_groups.jsonl')
    exclusion = json.loads((data / 'isolation.json').read_text(encoding='utf8'))
    forbidden = set(exclusion['forbidden_answer_ids'])
    assert not any(set(g['positive_ids']) & forbidden for g in groups)
    assert len({normalize(g['question']) for g in groups}) == len(groups)
    queries = {r['_id']: r['text'] for r in read_jsonl(raw / 'queries.jsonl')}
    upstream = read_jsonl(raw / 'corpus.jsonl')
    assert all(lookup['fiqa_a_'+r['_id']]['text'] == (r['title']+' '+r['text']).strip() for r in upstream)
    cases_by_split = {}
    for split, name in [('train', 'train'), ('valid', 'dev'), ('test', 'test')]:
        expected = defaultdict(set)
        for row in csv.DictReader((raw / f'qrels/{name}.tsv').open(encoding='utf8'), delimiter='\t'):
            assert row['score'] == '1'
            expected[row['query-id']].add('fiqa_a_'+row['corpus-id'])
        cases = read_jsonl(fixture / f'{split}.jsonl')
        cases_by_split[split] = cases
        assert {c['upstream_query_id'] for c in cases} == set(expected)
        for c in cases:
            assert c['question'] == queries[c['upstream_query_id']]
            assert set(c['gold_answer_ids']) == expected[c['upstream_query_id']]
            if split != 'train': assert set(c['gold_answer_ids']) <= forbidden
    held_text = {normalize(lookup[a]['text']) for split in ['valid', 'test'] for c in cases_by_split[split] for a in c['gold_answer_ids']}
    assert all(a['id'] in forbidden for a in answers if normalize(a['text']) in held_text)
    previous = read_jsonl(ROOT / 'data/training/condition_listwise_v1/train_groups.jsonl')
    held = cases_by_split['valid'] + cases_by_split['test']
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform([g['question'] for g in previous]+[c['question'] for c in held])
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=4).fit(matrix[:len(previous)])
    flags = []
    for offset in range(0, len(held), 128):
        distances, neighbors = nn.kneighbors(matrix[len(previous)+offset:len(previous)+offset+128])
        for c, d, ix in zip(held[offset:offset+128], distances, neighbors):
            similarity = float(1-d[0])
            if similarity >= .92:
                g = previous[int(ix[0])]
                flags.append({'id': c['id'], 'split': c['split'], 'question': c['question'],
                              'previous_training_id': g['id'], 'previous_training_question': g['question'],
                              'similarity': similarity})
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified',
        'all_fiqa_original_questions_and_relevance_labels_match': True,
        'all_fiqa_corpus_texts_match_stripped_original': True,
        'new_training_questions': len(groups), 'domains': dict(Counter(g['source_domain'] for g in groups)),
        'positive_pairs': sum(len(g['positive_ids']) for g in groups),
        'finance_positive_pairs': sum(len(g['positive_ids']) for g in groups if g['source_domain'] == 'finance'),
        'forbidden_positive_or_candidate_count': 0,
        'eligible_training_candidate_count': len(answers)-len(forbidden),
        'exclusion_reasons': dict(Counter(e['reason'] for e in exclusion['exclusions'])),
        'insurance_keyword_slice': {'rule': INSURANCE.pattern, 'expert_labels': False,
                                    'finance_training_questions': sum(g['source_domain'] == 'finance' and bool(INSURANCE.search(g['question'])) for g in groups),
                                    **{split: [c['id'] for c in cases if INSURANCE.search(c['question'])] for split, cases in cases_by_split.items()}},
        'prior_reranker_near_question_flags': flags,
        'prior_exposure_threshold': .92, 'primary_tests_modified': False,
        'public_pretraining_exposure': 'Unknown; BEIR is a public benchmark.',
        'source_cluster_limitation': 'FiQA supplies query and document IDs, not reliable originating author/thread groups; use shared-label components and disclose remaining dependence.',
        'training_manifest_sha256': sha(data / 'manifest.lock.json'), 'fixture_sha256': sha(fixture / 'manifest.lock.json'),
        'code_sha256': sha(Path(__file__)),
    }
    write_json(result, ROOT / 'reports/query_adaptation_v1/data_verification.json')
    print(json.dumps({'status': 'verified', 'domains': result['domains'], 'finance_positive_pairs': result['finance_positive_pairs'],
                      'insurance_keyword_training': result['insurance_keyword_slice']['finance_training_questions'],
                      'insurance_keyword_test': len(result['insurance_keyword_slice']['test']), 'prior_exposure_flags': flags}))


if __name__ == '__main__':
    main()
