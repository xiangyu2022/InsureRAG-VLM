"""Materialize the original corpus plus the separately archived v2 official PDFs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    original = ROOT/'data/04_curated'
    benchmark = ROOT/'data/benchmarks/research_v2'
    lock = json.loads((benchmark/'manifest.lock.json').read_text())
    for name, digest in lock['source_files'].items():
        if sha(benchmark/name) != digest:
            raise ValueError('Changed original PDF: '+name)
    for name, digest in lock['corpus_files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed frozen benchmark corpus: '+name)
    documents = json.loads((benchmark/'documents.json').read_text(encoding='utf8'))
    added = {d['doc_id']: d for d in documents if d['source_kind'] == 'official_pdf'}
    output.mkdir(parents=True, exist_ok=False)
    counts, files = {}, {}
    for kind in ['pages', 'snippets']:
        rows = read(original/f'rag_{kind}.jsonl')
        previous = len(rows)
        rows += [{**r, 'source_url': added[r['doc_id']]['source_url'], 'source_origin': 'archived_official_pdf'}
                 for r in read(benchmark/f'corpus/rag_{kind}.jsonl') if r['doc_id'] in added]
        if len({r['record_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate record ID')
        rows.sort(key=lambda r: (r['doc_id'], r['page'], r['record_id']))
        path = output/f'rag_{kind}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf8')
        files[path.name] = sha(path)
        counts[kind] = {'original': previous, 'added': len(rows)-previous, 'total': len(rows)}
        if kind == 'pages':
            source_kinds = Counter('html_record' if r['source_file'].lower().endswith('.html') else 'pdf_page_record' for r in rows)
            counts.update(documents=len({r['doc_id'] for r in rows}), page_record_types=dict(source_kinds),
                          added_pdf_physical_pages=sum(d['physical_pages'] for d in added.values()))
    manifest = {'version': 'expanded_public_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'counts': counts, 'files': files, 'new_source_documents': list(added.values()),
                'inputs': {f'data/04_curated/{n}': sha(original/n) for n in ['rag_pages.jsonl', 'rag_snippets.jsonl']},
                'benchmark_lock_sha256': sha(benchmark/'manifest.lock.json'),
                'limitations': ['Original 56-source snapshot preserved; acquisition dates not recovered for all old sources.',
                    'PDF page records exclude empty text pages; HTML pseudo-pages are not physical PDF pages.',
                    'Inherited snippets keep historical chunking; new PDF snippets use 180 words / 120 stride.',
                    'This corpus is for retrieval; adding text does not add independent evaluation questions.',
                    'Archived consumer guidance is not an issued policy or a current legal-advice dataset.']}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/research_corpus/expanded_public_v1')
    build(p.parse_args().output)
