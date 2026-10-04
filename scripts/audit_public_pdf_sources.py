"""Re-fetch six official PDFs and audit frozen evidence against physical pages.

This creates a separate provenance audit. It never changes corpus text, labels,
or scores, and does not authenticate an unavailable historical download.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().lower()


def fetch(document, cache):
    url = document['public_origin_reference']
    if document['doc_id']=='de_auto_insurance_guide':
        url = 'https://insurance.delaware.gov/wp-content/uploads/sites/15/2022/09/Auto-Insurance-Guide.pdf'
    result = {'doc_id':document['doc_id'],'requested_url':url,
              'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url,timeout=45)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b'%PDF'):
            raise ValueError('Official response is not a PDF')
        path = cache/(document['doc_id']+'.pdf')
        path.write_bytes(content)
        doc = pymupdf.open(stream=content,filetype='pdf')
        result.update({'status':'downloaded','resolved_url':response.url,
            'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),
            'physical_pages':len(doc),'local_file':str(path),
            'extracted_pages':{str(i+1):page.get_text() for i,page in enumerate(doc)}})
    except Exception as exc:
        result.update({'status':'error','error':type(exc).__name__+': '+str(exc)})
    return result


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    args.cache.mkdir(parents=True,exist_ok=True)
    documents = json.loads((ROOT/'data/benchmarks/research_v1/documents.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item:fetch(item,args.cache), documents))
    by_id = {item['doc_id']:item for item in results}
    questions = []
    for split in ['dev','test']:
        questions.extend(json.loads(line) for line in (ROOT/f'data/benchmarks/research_v1/{split}.jsonl').read_text().splitlines())
    evidence = []
    for question in questions:
        if not question['answerable']:
            continue
        doc = by_id[question['document_scope'][0]]
        for gold in question['gold']:
            page_text = doc.get('extracted_pages',{}).get(str(gold['page']))
            evidence.append({'id':question['id'],'source':gold['source'],
                'physical_page':gold['page'],'evidence_span':gold['evidence_span'],
                'page_available':page_text is not None,
                'normalized_span_present':normalize(gold['evidence_span']) in normalize(page_text) if page_text else False})
    for item in results:
        item.pop('extracted_pages',None)
    report = {'role':'current_official_source_provenance_audit_not_a_new_benchmark',
        'created_utc':datetime.now(timezone.utc).isoformat(),'documents':results,
        'frozen_evidence_checks':evidence,
        'summary':{'documents_downloaded':sum(item['status']=='downloaded' for item in results),
            'documents_attempted':len(results),'spans_checked':len(evidence),
            'spans_present_on_declared_physical_page':sum(item['normalized_span_present'] for item in evidence)},
        'limitations':['A fresh PDF hash does not prove the bytes originally used to curate the historical corpus.',
            'Only six benchmark guides are audited, not all56 sources.',
            'Whitespace/NFKC substring matching can fail because of extraction order or typography.',
            'The original benchmark corpus and labels are immutable; current downloads do not replace them.',
            'These guides are evidence documents for a research test, not a check of current insurance law.']}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
