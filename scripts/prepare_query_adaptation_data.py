"""Add original FiQA labels and isolate held-out positives before query training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha
from scripts.prepare_reranker_training import normalize


def write_jsonl(rows, path):
    # Escape Unicode separators: older frozen readers use str.splitlines().
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=True, sort_keys=True)+'\n' for r in rows), encoding='utf8')


def main():
    raw = ROOT / '../fiqa_download'
    raw.mkdir(exist_ok=True)
    url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip'
    archive = raw / 'fiqa.zip'
    if not archive.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        archive.write_bytes(response.content)
    assert hashlib.md5(archive.read_bytes()).hexdigest() == '17918ed23cd04fb15047f73e6c3bd9d9'
    with zipfile.ZipFile(archive) as z:
        for name in ['corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/dev.tsv', 'qrels/test.tsv']:
            path = raw / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(z.read('fiqa/' + name))
    api = requests.get('https://huggingface.co/api/datasets/BeIR/fiqa', timeout=40)
    api.raise_for_status()
    revision = api.json()['sha']
    cardurl = f'https://huggingface.co/datasets/BeIR/fiqa/raw/{revision}/README.md'
    card = requests.get(cardurl, timeout=40)
    card.raise_for_status()
    (raw / 'UPSTREAM_README.md').write_bytes(card.content)
    out = ROOT / 'data/benchmarks/fiqa_v1'
    out.mkdir(parents=True, exist_ok=False)
    corpus = read_jsonl(raw / 'corpus.jsonl')
    queries = {r['_id']: r['text'] for r in read_jsonl(raw / 'queries.jsonl')}
    answers = [{'id': 'fiqa_a_' + r['_id'], 'text': (r['title'] + ' ' + r['text']).strip(),
                'domain': 'finance', 'source_group': 'fiqa_document:' + r['_id'],
                'upstream_document_id': r['_id']} for r in corpus]
    assert len(answers) == 57638
    ids = {a['id'] for a in answers}
    by_split = {}
    for split, source in [('train', 'train'), ('valid', 'dev'), ('test', 'test')]:
        qrels = defaultdict(list)
        for row in csv.DictReader((raw / f'qrels/{source}.tsv').open(encoding='utf8'), delimiter='\t'):
            if int(row['score']) > 0:
                qrels[row['query-id']].append('fiqa_a_' + row['corpus-id'])
        by_split[split] = [{'id': 'fiqa_q_' + q, 'question': queries[q],
                            'gold_answer_ids': sorted(set(labels)), 'domain': 'finance', 'split': split,
                            'annotation_origin': 'Original FiQA/BEIR relevance labels; unchanged IDs',
                            'upstream_query_id': q} for q, labels in sorted(qrels.items(), key=lambda kv: int(kv[0]))]
        assert all(set(c['gold_answer_ids']) <= ids for c in by_split[split])
        write_jsonl(by_split[split], out / f'{split}.jsonl')
    assert {s: len(c) for s, c in by_split.items()} == {'train': 5500, 'valid': 500, 'test': 648}
    write_jsonl(answers, out / 'answers.jsonl')
    (out / 'UPSTREAM_README.md').write_bytes(card.content)
    write_json({'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'url': url,
                'archive_sha256': sha(archive), 'archive_md5': hashlib.md5(archive.read_bytes()).hexdigest(),
                'expected_md5_source': 'https://github.com/beir-cellar/beir',
                'dataset_card_url': cardurl, 'dataset_card_sha256': sha(out / 'UPSTREAM_README.md'),
                'license_as_declared_by_dataset_card': 'CC-BY-SA-4.0',
                'limitations': ['Public historical financial forum answers; not verified insurance or legal guidance.',
                                'Public BGE/MS MARCO pretraining exposure unknown.',
                                'Answer IDs are available; originating thread/author clusters are not supplied.'],
                'upstream_files_sha256': {str(p.relative_to(raw)): sha(p) for p in raw.rglob('*') if p.is_file()}}, out / 'provenance.json')
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': {s: len(c) for s, c in by_split.items()},
                'answers': len(answers), 'code_sha256': sha(Path(__file__)),
                'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}, out / 'manifest.lock.json')
    print(json.dumps({'fiqa_imported': True, 'counts': {s: len(c) for s, c in by_split.items()}}), flush=True)

    previous = ROOT / 'data/training/condition_listwise_v1'
    oldgroups = read_jsonl(previous / 'train_groups.jsonl')
    allanswers = read_jsonl(previous / 'answers.jsonl') + answers
    answer_lookup = {a['id']: a for a in allanswers}
    assert len(answer_lookup) == len(allanswers)
    textkeys = {a['id']: normalize(a['text']) for a in allanswers}
    held_fiqa = by_split['valid'] + by_split['test']
    held_legacy = [c for split in ['valid', 'test'] for c in read_jsonl(ROOT / f'data/benchmarks/insuranceqa_v2/{split}.jsonl')]
    held_other = [c for split in ['valid', 'test'] for c in read_jsonl(ROOT / f'data/benchmarks/multidomain_v1/{split}.jsonl')]
    held_other += read_jsonl(ROOT / 'data/benchmarks/condition_v1/test.jsonl')
    oldgov_answers = read_jsonl(ROOT / 'data/benchmarks/hicric_government_qa_v1/answers.jsonl')
    held_other += read_jsonl(ROOT / 'data/benchmarks/hicric_government_qa_v1/questions.jsonl')
    held = held_fiqa + held_legacy + held_other
    forbidden_sources = {c.get('source_group') or c.get('source_url') for c in held_other}
    forbidden_texts = {normalize(a['text']) for a in oldgov_answers}
    forbidden_texts |= {textkeys[a] for c in held_fiqa for a in c['gold_answer_ids']}
    forbidden_texts |= {textkeys[a['id']] for a in allanswers if a.get('source_group') in forbidden_sources and a.get('source_group')}
    forbidden = {a for a, t in textkeys.items() if t in forbidden_texts or not t}
    groups = [{'id': g['id'], 'question': g['question'], 'positive_ids': g['positive_ids'],
               'source_domain': g['source_domain'], 'source_group': g.get('source_group'),
               'original_question': True} for g in oldgroups]
    groups += [{'id': c['id'], 'question': c['question'], 'positive_ids': c['gold_answer_ids'],
                'source_domain': 'finance', 'original_question': True} for c in by_split['train']]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    # Held-out text is used only to reject overlaps, never as a training label.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform([c['question'] for c in held] + [g['question'] for g in groups])
    nearest = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=4).fit(matrix[:len(held)])
    similarities = []
    for offset in range(len(held), matrix.shape[0], 128):
        distances, indices = nearest.kneighbors(matrix[offset:offset+128])
        similarities.extend((float(1-d[0]), held[int(i[0])]['id']) for d, i in zip(distances, indices))
    accepted, exclusions, seen = [], [], set()
    for g, (similarity, nearest_id) in zip(groups, similarities):
        key = normalize(g['question'])
        reason = None
        if any(a in forbidden for a in g['positive_ids']): reason = 'heldout_positive_text_or_source'
        elif not key or key in seen: reason = 'duplicate_or_empty_question'
        elif similarity >= .92: reason = 'near_heldout_question'
        if reason:
            exclusions.append({'id': g['id'], 'reason': reason, 'similarity': similarity, 'nearest_heldout': nearest_id})
            continue
        seen.add(key)
        accepted.append(g)
    training = ROOT / 'data/training/query_adaptation_v1'
    training.mkdir(parents=True, exist_ok=False)
    write_jsonl(allanswers, training / 'answers.jsonl')
    write_jsonl(accepted, training / 'train_groups.jsonl')
    write_json({'forbidden_answer_ids': sorted(forbidden), 'forbidden_sources': sorted(x for x in forbidden_sources if x),
                'exclusions': exclusions, 'near_question_threshold': .92,
                'excluded_heldout_fiqa_gold_documents_and_text_aliases': True,
                'legacy_insuranceqa_shared_labels': 'Original historical FAQ source split retained; shared answer IDs remain.'}, training / 'isolation.json')
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'training_questions': len(accepted),
                'source_domains': dict(Counter(g['source_domain'] for g in accepted)),
                'additional_finance_questions': sum(g['source_domain'] == 'finance' for g in accepted),
                'answer_candidates': len(allanswers), 'eligible_training_answers': len(allanswers)-len(forbidden),
                'removed_previous_training_questions': sum(g['id'] in {e['id'] for e in exclusions} for g in oldgroups),
                'parent_training_sha256': sha(previous / 'manifest.lock.json'),
                'fiqa_fixture_sha256': sha(out / 'manifest.lock.json'), 'code_sha256': sha(Path(__file__)),
                'files': {p.name: sha(p) for p in training.iterdir() if p.is_file()}}, training / 'manifest.lock.json')
    print(json.dumps({'training_questions': len(accepted), 'domains': dict(Counter(g['source_domain'] for g in accepted)),
                      'excluded': len(exclusions), 'eligible_training_answers': len(allanswers)-len(forbidden)}), flush=True)


if __name__ == '__main__':
    main()
