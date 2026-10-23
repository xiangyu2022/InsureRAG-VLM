"""Cheap source-URL exposure gate before authoring or lexical/model audits.

This preserves screen_holdout1000_history's conservative normalized substring
rule, including mere references. Unlike its per-host URL dictionary, all source
IDs sharing a normalized URL, and overlapping parent/child URLs, receive hits.
No source text, question, answer or model inference is needed.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_source_faq import norm

SCHEMA_VERSION = 1


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def url_requests(documents):
    requests = []
    seen = set()
    for document in documents:
        ident, publisher, url = (document[k] for k in ('id', 'publisher', 'source_url'))
        if not all(isinstance(v, str) and v.strip() for v in (ident, publisher, url)):
            raise ValueError('Source identifiers, publishers and URLs must be nonempty strings')
        if ident in seen:
            raise ValueError('Duplicate source ID')
        seen.add(ident)
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Expected a public HTTP(S) source URL')
        requests.append({
            'id': ident, 'publisher': publisher,
            'host': norm(parsed.netloc).removeprefix('www '),
            'url_key': norm(url).removeprefix('https ').removeprefix('http ').removeprefix('www '),
        })
    if not requests:
        raise ValueError('At least one source required')
    return sorted(requests, key=lambda r: r['id'])


def configuration(documents):
    requests = url_requests(documents)
    encoded = json.dumps(requests, sort_keys=True, separators=(',', ':')).encode('utf8')
    return requests, {
        'schema_version': SCHEMA_VERSION,
        'source_url_records_sha256': digest_bytes(encoded),
        'normalization_code_sha256': digest_bytes((ROOT / 'scripts/prepare_source_faq.py').read_bytes()),
        'screening_code_sha256': digest_bytes(Path(__file__).read_bytes()),
        'normalization': 'prepare_source_faq.norm: casefold and word tokens; same as exact-history gate',
    }


def check_report_binding(report, documents, historical_sha256):
    """Check cached URL-only provenance; caller also binds the report file hash.

The supplied history digest must come from a verified unchanged history cache.
Document text may change without invalidating this URL-only check. Such reuse
provides no content, exact-QA, near-duplicate or semantic clearance.
    """
    requests, expected = configuration(documents)
    if report.get('configuration') != expected or report.get('historical_sha256') != historical_sha256:
        raise ValueError('Stale source-URL report binding')
    if [row['id'] for row in report.get('rows', [])] != [r['id'] for r in requests]:
        raise ValueError('Source-URL report rows do not match requested sources')
    if report.get('accepted_test_items') != 0:
        raise ValueError('A URL-only report cannot accept test items')


def screen_source_urls(documents, history_path):
    requests, config = configuration(documents)
    by_host = defaultdict(dict)
    for request in requests:
        by_host[request['host']][request['url_key']] = None
    first_hits = {}
    digest = hashlib.sha256()
    total = offset = 0
    with Path(history_path).open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            row_offset = offset
            offset += len(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            text, origin = record['text'], record['origin']
            if not isinstance(text, str) or not isinstance(origin, str):
                raise ValueError('History text and origin must be strings')
            # Existing historical_texts.jsonl stores exact-gate-normalized text.
            for host, keys in by_host.items():
                if host not in text:
                    continue
                for key in keys:
                    if key in first_hits:
                        continue
                    start = text.find(key)
                    if start >= 0:
                        first_hits[key] = {
                            'origin': origin,
                            'historical_text_sha256': digest_bytes(text.encode('utf8')),
                            'history_row_index': total,
                            'byte_offset': row_offset,
                            'byte_length': len(raw),
                            'match_start': start,
                            'match_end': start + len(key),
                        }
            total += 1
    rows = [{**r, 'historical_url_hit': first_hits.get(r['url_key']),
             'excluded_from_independent_holdout': r['url_key'] in first_hits} for r in requests]
    return {
        'status': 'SOURCE_URL_GATE_ONLY_NOT_CONTAMINATION_CLEARANCE',
        'configuration': config,
        'historical_sha256': digest.hexdigest(),
        'historical_strings': total,
        'source_documents': len(requests),
        'excluded_sources': sum(r['excluded_from_independent_holdout'] for r in rows),
        'rows': rows,
        'accepted_test_items': 0,
        'model_calls': 0,
        'limitations': [
            'A URL occurrence, including a reference, excludes the source under the existing rule.',
            'No URL hit is not proof of source or question independence.',
            'QA exact/containment/lexical/semantic, aliases, family, rights and temporal gates remain separate.',
            'History origin labels are recorded, never opened as paths.',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--documents', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous source-URL audit')
    document_bytes = args.documents.read_bytes()
    documents = [json.loads(line) for line in document_bytes.decode('utf8').splitlines() if line.strip()]
    report = screen_source_urls(documents, args.history)
    report['documents_file_sha256'] = digest_bytes(document_bytes)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
