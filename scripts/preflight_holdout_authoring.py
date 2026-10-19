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
from scripts.preflight_holdout_source_urls import check_report_binding


def local_path(local, value):
    path = (local / value).resolve()
    if not path.is_relative_to(local.resolve()) or path == local.resolve():
        raise ValueError('All private inputs and output must stay under the explicit local directory')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool-number', type=int, required=True)
    for name in ('documents', 'candidates', 'manifest', 'legacy', 'issues', 'output'):
        parser.add_argument('--' + name, required=True, help='Path relative to root/local')
    parser.add_argument('--document', action='append', required=True)
    parser.add_argument('--source-url-audit', help='Hash-bound URL-only report relative to root/local')
    parser.add_argument('--history-sha256', help='Digest of the verified unchanged historical-text cache')
    args = parser.parse_args()
    if bool(args.source_url_audit) != bool(args.history_sha256):
        parser.error('--source-url-audit and --history-sha256 must be supplied together')
    if args.pool_number < 1:
        parser.error('--pool-number must be positive')
    root = args.root.resolve(); local = root / 'local'; bindings = {}

    def read(path, lines=False):
        raw = path.read_bytes()
        bindings[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()] if lines else json.loads(raw)

    # No globs, origin paths, URLs or embedded review pointers are followed.
    paths = {name: local_path(local, getattr(args, name)) for name in
             ('documents', 'candidates', 'manifest', 'legacy', 'issues', 'output')}
    if paths['output'].exists():
        raise FileExistsError('Private packet output must be new')
    docs = read(paths['documents'], True)
    documents = {d['id']: d for d in docs}
    if len(docs) != len(documents):
        raise ValueError('Duplicate document identity')
    url_rows = {}
    if args.source_url_audit:
        url_audit = read(local_path(local, args.source_url_audit))
        check_report_binding(url_audit, docs, args.history_sha256)
        url_rows = {r['id']: r for r in url_audit['rows']}
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
    issues = read(paths['issues'])['issues']
    if any(d['id'] not in documents for issue in issues for d in issue['documents']):
        raise ValueError('Issue register references an unknown document')
    decisions = {e['document_id']: e['review'] for e in manifest['entries']}
    restrictions = {key: {'source_hold': source_holds.get(key),
                          'source_review': decisions.get(key, {'decision': 'missing'}),
                          'source_url_gate_status': ('URL_EXPOSED' if url_rows[key]['excluded_from_independent_holdout'] else 'NO_URL_HIT_NOT_CLEARANCE') if url_rows else 'NOT_CHECKED',
                          'historical_source_url_hit': url_rows[key]['historical_url_hit'] if url_rows else None,
                          'legacy_records': [r for r in legacy['records'] if r.get('document') == key],
                          'source_quality_issues': [r for r in issues
                              if any(d['id'] == key for d in r['documents'])]}
                    for key in documents}
    result = build_authoring_packet(documents, candidates, {q['id'] for q in active}, families,
                                     item_dispositions(ledgers, inherited), restrictions, args.document)
    result['input_sha256'] = bindings
    result['source_url_gate'] = {
        'status': 'CHECKED_URL_ONLY' if url_rows else 'NOT_CHECKED',
        'historical_sha256': args.history_sha256,
        'selected_sources_exposed': [key for key in args.document if url_rows and url_rows[key]['excluded_from_independent_holdout']],
        'not_global_clearance': True,
    }
    with paths['output'].open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'documents': len(result['documents']),
                      'output': str(paths['output']), 'sha256': hashlib.sha256(paths['output'].read_bytes()).hexdigest(),
                      'models_called': 0, 'private_packet_written': True}))


if __name__ == '__main__':
    main()
