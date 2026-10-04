"""Extract source-authored FAQs with exact normalized-text evidence offsets.

Every output remains a candidate. No automated extraction is content review.
"""
import argparse,hashlib,json,re,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_source_faq import clean,norm
from src.insurerag_vlm.source_faq_extraction import extract_original_faq,content_root
LOCAL=ROOT/'reports/holdout1000_v1/local'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def html_rejection_reason(raw,metadata):
    """Reject non-HTML downloads before parsing; never guess spreadsheet text."""
    if raw.lstrip().startswith(b'%PDF'):return 'pdf_not_html'
    if raw.startswith((b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08',b'\xd0\xcf\x11\xe0',b'\x89PNG',b'GIF87a',b'GIF89a',b'\xff\xd8\xff')):
        return 'binary_signature_not_html'
    suffix=Path(urlsplit(metadata.get('final_url') or metadata.get('url','')).path).suffix.lower()
    if suffix in {'.xlsx','.xls','.xlsm','.doc','.docx','.ppt','.pptx','.zip','.csv','.json','.png','.jpg','.jpeg','.gif','.pdf'}:
        return 'non_html_download_extension'
    media=(metadata.get('content_type') or '').split(';',1)[0].strip().lower()
    if media and media not in {'text/html','application/xhtml+xml'}:return 'non_html_content_type'
    if b'\x00' in raw:return 'binary_nul_requires_dedicated_decoder'
    if not re.search(br'<(?:!doctype\s+html|html\b|head\b|body\b|main\b|article\b|div\b|h[1-6]\b|p\b)',raw[:65536],re.I):
        return 'html_markup_not_detected'
    return None

def extract(root=LOCAL):
    registry={r['id']:r for r in json.loads((ROOT/'reports/holdout1000_v1/source_candidates.json').read_text(encoding='utf8'))}
    records=[];documents=[];errors=[];canonical_documents={}
    for path in sorted((root/'documents').glob('*/*.json')):
        m=json.loads(path.read_text(encoding='utf8'));publisher=path.parent.name
        if m.get('status')!=200:continue
        raw=(path.parent/m['artifact']).read_bytes()
        if sha(raw)!=m['sha256']:raise ValueError('Source bytes changed: '+str(path))
        rejection=html_rejection_reason(raw,m)
        if rejection:errors.append({'url':m['url'],'final_url':m.get('final_url'),'source_sha256':m['sha256'],'reason':rejection});continue
        soup=BeautifulSoup(raw,'html.parser');main=content_root(soup)
        text=clean(main.get_text(' ',strip=True));url=m.get('final_url',m['url']);doc_id=sha(url.encode())[:24]
        title=m.get('title','');h1=main.find('h1')
        if h1:title=clean(h1.get_text(' ',strip=True))
        doc={'id':doc_id,'publisher':publisher,'jurisdiction':registry[publisher]['jurisdiction'],'source_url':url,
             'source_title':title,'source_sha256':m['sha256'],'acquired_utc':m['acquired_utc'],
             'normalized_text_sha256':sha(text.encode()),'text':text,'source_artifact':str(path.relative_to(root)).replace('\\','/')}
        previous=canonical_documents.get(doc_id)
        if previous:
            if previous['normalized_text_sha256']!=doc['normalized_text_sha256']:
                raise ValueError('Canonical URL has conflicting snapshots; explicit version resolution required: '+url)
            previous.setdefault('source_aliases',[]).append({'requested_url':m['url'],'source_artifact':doc['source_artifact'],'source_sha256':doc['source_sha256']})
            continue
        canonical_documents[doc_id]=doc;documents.append(doc)
        seen=set()
        for pair in extract_original_faq(raw):
            q=pair['question'];answer=pair['answer'];key=norm(q)+'|'+norm(answer)
            if key in seen:continue
            seen.add(key);offset=text.find(answer);flags=list(pair.get('extraction_flags',[]))
            if offset<0:flags.append('evidence_alignment_failed')
            if '\ufffd' in q+answer:flags.append('encoding_replacement_character')
            if len(q.split())<5:flags.append('question_context_incomplete')
            if len(answer.split())>1000:flags.append('answer_boundary_or_excess_length')
            if re.search(r'(was this|is this page|feedback|what can we help|how can we help)',q,re.I):flags.append('website_navigation_question')
            if answer.count('?')>2:flags.append('possible_multiple_faq_boundary')
            if re.search(r'\b(click here|see above|below|following link)\b',answer,re.I):flags.append('external_or_context_dependency')
            record={'id':sha((doc_id+'|'+key).encode())[:24],'status':'candidate_unreviewed','publisher':publisher,
                    'jurisdiction':registry[publisher]['jurisdiction'],'insurance_type':None,'task':'ordinary_qa',
                    'origin':'source_authored_faq','extraction_method':pair['extraction_method'],'document_group':doc_id,'source_title':title,'source_url':url,
                    'source_sha256':m['sha256'],'question':q,'answer':answer,
                    'evidence':[{'document_id':doc_id,'start':offset,'end':offset+len(answer),'sha256':sha(answer.encode())}],
                    'automatic_flags':flags,'content_review':None}
            records.append(record)
    duplicates=defaultdict(list)
    for r in records:duplicates[norm(r['question'])].append(r['id'])
    groups=[v for v in duplicates.values() if len(v)>1]
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'documents':len(documents),'candidates':len(records),
            'accepted_test_items':0,'by_publisher':dict(Counter(r['publisher'] for r in records)),
            'flag_counts':dict(Counter(f for r in records for f in r['automatic_flags'])),
            'exact_question_duplicate_groups':groups,'source_errors':errors}
    return documents,records,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve extraction snapshot; use fresh output')
    documents,records,report=extract();a.output.mkdir(parents=True)
    for name,values in [('documents',documents),('candidates',records)]:
        with (a.output/(name+'.jsonl')).open('x',encoding='utf8') as handle:
            for value in values:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'exact_question_duplicate_groups','source_errors'}},indent=2))
if __name__=='__main__':main()
