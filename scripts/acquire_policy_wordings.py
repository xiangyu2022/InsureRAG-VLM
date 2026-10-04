"""Download a fresh or hash-pinned snapshot of eight OPM plan brochures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import requests
import fitz

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('71-004','APWU FEHB','apwu','dev'),('71-006','GEHA High and Standard','geha','dev'),
 ('71-014','GEHA HDHP','geha','dev'),('71-015','SAMBA','samba','test'),('71-024','NALC PSHB','nalc','test'),
 ('71-027','MHBP Consumer Option','mhbp','test'),('71-028','Compass Rose','compass_rose','test'),
 ('72-001','Foreign Service Benefit Plan','foreign_service','test')]


def run(output,expected_registry=None):
    expected={r['doc_id']:r for r in json.loads(expected_registry.read_text(encoding='utf8'))} if expected_registry else {}
    output.mkdir(parents=True,exist_ok=False)
    def acquire(source):
        code,title,family,split=source
        url=f'https://www.opm.gov/healthcare-insurance/healthcare/plan-information/plans/pdf/2026/brochures/{code}.pdf'
        row=dict(doc_id=f'opm_2026_{code}',title=title,document_family_id=family,split=split,url=url,year=2026)
        try:
            response=requests.get(url,timeout=120);response.raise_for_status()
            if not response.content.startswith(b'%PDF-'):raise ValueError('Not a PDF')
            digest=hashlib.sha256(response.content).hexdigest()
            if expected and digest!=expected[row['doc_id']]['sha256']:
                raise ValueError('Upstream bytes changed; use archived source, do not silently replace benchmark input')
            dest=output/'sources'/f'{row["doc_id"]}.pdf';dest.parent.mkdir(exist_ok=True);dest.write_bytes(response.content)
            row.update(sha256=digest,bytes=len(response.content),file='sources/'+dest.name,
                       retrieved_utc=datetime.now(timezone.utc).isoformat(),final_url=response.url)
            with fitz.open(dest) as pdf:
                pages=[dict(doc_id=row['doc_id'],physical_page=i+1,text=page.get_text('text',sort=True)) for i,page in enumerate(pdf)]
                row.update(physical_pages=len(pdf),text_pages=sum(bool(p['text'].strip()) for p in pages),pdf_metadata=pdf.metadata)
            textdir=output/'extracted';textdir.mkdir(exist_ok=True)
            (textdir/f'{row["doc_id"]}.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pages),encoding='utf8')
            row['status']='available'
        except Exception as exc:row.update(status='failed',error=str(exc))
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(acquire,SOURCES))
    (output/'acquisition.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    if any(r['status']!='available' for r in rows):raise RuntimeError('Acquisition incomplete; see acquisition.json')
    print(json.dumps({'documents':len(rows),'physical_pages':sum(r['physical_pages'] for r in rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-registry',type=Path)
    args=parser.parse_args();run(args.output,args.expected_registry)
