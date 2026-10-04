"""Import the authors' InsuranceQA V2 without inventing or injecting positives."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '5c380dd086067adacc2fec3bfa920a78e57bdcf6'
BASE = f'https://raw.githubusercontent.com/shuzi/insuranceQA/{REVISION}/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def write_jsonl(rows, path):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows), encoding='utf8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def decode(text, vocabulary):
    return ' '.join(vocabulary[token] if token.startswith('idx_') else token for token in text.split())


def parse_question(line, vocabulary, split, position):
    domain, question, gold, pool = line.rstrip('\n').split('\t')
    return {'id': f'iqa_v2_{split}_{position:05d}', 'split': split, 'domain': domain,
            'question': decode(question, vocabulary), 'gold_answer_ids': sorted(set(gold.split())),
            'official_pool_ids': list(dict.fromkeys(pool.split())),
            'upstream_row': position, 'annotation_origin': 'InsuranceQA V2 authors; unchanged answer IDs'}


def audit_splits(cases, answers):
    answer_ids = {a['id'] for a in answers}
    by_split = {split: [r for r in cases if r['split'] == split] for split in ['train', 'valid', 'test']}
    seen = {split: {normalized(r['question']) for r in rows} for split, rows in by_split.items()}
    for row in cases:
        if not row['question'].strip() or not row['gold_answer_ids']:
            raise ValueError('Empty question or missing positive labels')
        if not set(row['gold_answer_ids'] + row['official_pool_ids']) <= answer_ids:
            raise ValueError('Unknown answer ID')
        row['exact_question_overlap_with_train_or_valid'] = (
            row['split'] == 'test' and normalized(row['question']) in seen['train'] | seen['valid'])
    text_groups = defaultdict(list)
    for answer in answers:
        text_groups[normalized(answer['text'])].append(answer['id'])
    return {'counts': {s: len(rs) for s, rs in by_split.items()}, 'answers': len(answers),
            'domains': {s: dict(Counter(r['domain'] for r in rs)) for s, rs in by_split.items()},
            'unique_normalized_questions': {s: len(seen[s]) for s in seen},
            'test_exact_overlap_with_train_or_valid': sum(r['exact_question_overlap_with_train_or_valid'] for r in by_split['test']),
            'within_test_duplicate_rows': len(by_split['test'])-len(seen['test']),
            'official_pool_contains_positive': {s: sum(bool(set(r['gold_answer_ids']) & set(r['official_pool_ids'])) for r in rs) for s, rs in by_split.items()},
            'answer_text_duplicate_groups': sum(len(ids)>1 for ids in text_groups.values()),
            'answer_text_duplicate_extra_ids': sum(len(ids)-1 for ids in text_groups.values()),
            'near_duplicate_questions_not_removed': True,
            'split_contract': 'Original author splits, not document-family holdout. Train is imported for audit only; no fitting in baseline.'}


def verify_fixture(folder):
    lock = json.loads((folder/'manifest.lock.json').read_text(encoding='utf8'))
    for name, digest in lock['files'].items():
        if sha(folder/name) != digest:
            raise ValueError(f'Fixture checksum mismatch: {name}')
    return lock


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    raw = output/'upstream'
    raw.mkdir()
    paths = ['README.md', 'V2/README.md', 'V2/vocabulary', 'V2/InsuranceQA.label2answer.raw.encoded.gz']
    paths += [f'V2/InsuranceQA.question.anslabel.raw.{100 if s == "train" else 1000}.pool.solr.{s}.encoded.gz' for s in ['train','valid','test']]
    provenance = []
    for name in paths:
        target = raw/name
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(BASE+name, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
        provenance.append({'url': BASE+name, 'file': str(target.relative_to(output)).replace('\\','/'),
                           'sha256': sha(target), 'bytes': target.stat().st_size, 'retrieved_utc': datetime.now(timezone.utc).isoformat()})
    vocabulary = dict(line.split('\t',1) for line in (raw/'V2/vocabulary').read_text(encoding='utf8').splitlines())
    answers = []
    with gzip.open(raw/'V2/InsuranceQA.label2answer.raw.encoded.gz', 'rt', encoding='utf8') as handle:
        for line in handle:
            label, encoded = line.rstrip('\n').split('\t',1)
            answers.append({'id': label, 'text': decode(encoded, vocabulary)})
    if len({a['id'] for a in answers}) != len(answers):
        raise ValueError('Duplicate answer IDs')
    cases = []
    for split in ['train','valid','test']:
        filename = f'InsuranceQA.question.anslabel.raw.{100 if split == "train" else 1000}.pool.solr.{split}.encoded.gz'
        with gzip.open(raw/'V2'/filename, 'rt', encoding='utf8') as handle:
            cases.extend(parse_question(line, vocabulary, split, i) for i,line in enumerate(handle,1))
    audit = audit_splits(cases, answers)
    if audit['counts'] != {'train':12889,'valid':2000,'test':2000} or len(answers) != 27413:
        raise ValueError(f'Unexpected pinned upstream counts: {audit["counts"]}')
    write_jsonl(answers, output/'answers.jsonl')
    for split in ['train','valid','test']:
        write_jsonl([r for r in cases if r['split']==split], output/f'{split}.jsonl')
    write_json(audit, output/'audit.json')
    write_json(provenance, output/'provenance.json')
    lock = {'version':'insuranceqa_v2_retrieval_v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
            'upstream_revision':REVISION, 'upstream_repository':'https://github.com/shuzi/insuranceQA',
            'use_terms':'Upstream: provided as is and for research purpose only. Cite Feng et al., ASRU 2015. Not relicensed by this repository.',
            'corpus_type':'Historical professional FAQ answers, not policy pages or legal authority',
            'files':{str(p.relative_to(output)).replace('\\','/'):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
            'counts':audit['counts'], 'answers':len(answers)}
    write_json(lock, output/'manifest.lock.json')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/benchmarks/insuranceqa_v2')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_fixture(args.output)['counts'])) if args.verify_only else build(args.output)
