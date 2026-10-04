"""Re-extract every new PDF and compare its physical pages with the frozen text."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1


def run(output):
    fixture = ROOT/'data/benchmarks/research_v2'
    lock = json.loads((fixture/'manifest.lock.json').read_text())
    documents = json.loads((fixture/'documents.json').read_text(encoding='utf8'))
    pages = v1.read_jsonl(fixture/'corpus/rag_pages.jsonl')
    by_source = {p['citation']: p for p in pages}
    normalize = lambda t: re.sub(r'\s+', ' ', t).strip()
    results = []
    for doc in documents:
        if doc['source_kind'] != 'official_pdf':
            continue
        name = 'sources/'+doc['doc_id']+'.pdf'
        path = fixture/name
        if v1.sha(path) != lock['source_files'][name]:
            raise ValueError('Changed PDF: '+name)
        with fitz.open(path) as pdf:
            checked, empty = 0, 0
            for i, page in enumerate(pdf, 1):
                text = normalize(page.get_text('text', sort=True))
                if not text:
                    empty += 1
                    continue
                source = doc['doc_id']+f'.pdf#page={i}'
                if source not in by_source or normalize(by_source[source]['text']) != text:
                    raise ValueError('Extracted text mismatch: '+source)
                checked += 1
            results.append({'doc_id': doc['doc_id'], 'source_url': doc['source_url'], 'sha256': v1.sha(path),
                            'physical_pages': len(pdf), 'exact_text_matches': checked, 'empty_pages': empty})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'pymupdf': fitz.VersionBind,
              'verified_pdfs': len(results), 'verified_text_pages': sum(r['exact_text_matches'] for r in results),
              'physical_pages': sum(r['physical_pages'] for r in results), 'documents': results,
              'meaning': 'Byte identity and native text/physical-page correspondence only; not expert semantic review.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'documents'}))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'reports/graph_research_v1/source_audit.json')
    run(p.parse_args().output)
