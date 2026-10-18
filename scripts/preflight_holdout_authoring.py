#!/usr/bin/env python3
"""Write an ignored private packet before drafting from cached holdout sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.holdout_authoring_preflight import build_authoring_packet, item_dispositions
from src.insurerag_vlm.holdout_quality import reviewed_families


def local_path(local, value):
    path = (local / value).resolve()
    if not path.is_relative_to(local.resolve()) or path == local.resolve():
        raise ValueError('All private inputs and output must stay under the explicit local directory')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool-number', type=int, required=True)
    for name in ('documents', 'candidates', 'manifest', 'legacy', 'output'):
        parser.add_argument('--' + name, required=True, help='Path relative to root/local')
    parser.add_argument('--document', action='append', required=True)
    args = parser.parse_args()
    if args.pool_number < 1:
        parser.error('--pool-number must be positive')
    root = args.root.resolve(); local = root / 'local'; bindings = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        bindings[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()] if lines else json.loads(raw)

    # No globs, origin paths, URLs or embedded review pointers are followed.
    paths = {name: local_path(local, getattr(args, name)) for name in
             ('documents', 'candidates', 'manifest', 'legacy', 'output')}
    if paths['output'].exists():
        raise FileExistsError('Private packet output must be new')
    docs = read(paths['documents'], True)
    documents = {d['id']: d for d in docs}
    if len(docs) != len(documents):
        raise ValueError('Duplicate document identity')
    candidates = read(paths['candidates'], True)
    active = read(local / f'provisional_item_pool_{args.pool_number:02d}.jsonl', True)
    families = reviewed_families(documents, read(root / 'document_families.json'))
    source_holds = read(root / 'document_holds.json')['documents']
    inherited = read(local / 'stage10_inherited_holds_reconciliation.json')['effective_item_holds']
    ledgers = []
    for number in range(1, args.pool_number + 1):
        path = local / f'review_pool_{number:02d}_clearance_ledger.json'
        data = read(path)
        ledgers.append({'sha256': bindings[str(path.relative_to(root))], 'data': data})
    manifest = read(paths['manifest']); legacy = read(paths['legacy'])
    decisions = {e['document_id']: e['review'] for e in manifest['entries']}
    restrictions = {key: {'source_hold': source_holds.get(key),
                          'source_review': decisions.get(key, {'decision': 'missing'}),
                          'legacy_records': [r for r in legacy['records'] if r.get('document') == key]}
                    for key in documents}
    result = build_authoring_packet(documents, candidates, {q['id'] for q in active}, families,
                                     item_dispositions(ledgers, inherited), restrictions, args.document)
    result['input_sha256'] = bindings
    with paths['output'].open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'documents': len(result['documents']),
                      'output': str(paths['output']), 'sha256': hashlib.sha256(paths['output'].read_bytes()).hexdigest(),
                      'models_called': 0, 'private_packet_written': True}))


if __name__ == '__main__':
    main()
