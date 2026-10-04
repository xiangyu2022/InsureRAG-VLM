"""Build source-linked public plan wording corpus from archived OPM PDF bytes.

Numeric page references only. No QA labels, guessed cross-plan links, or claim
of legal precedence. Printed labels are read from PDF footer geometry.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,re,sys,shutil
from pathlib import Path
import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,read_jsonl,write_json,write_jsonl


def printed_footer(page):
    candidates=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            text=''.join(s['text'] for s in line['spans']).strip()
            x0,y0,x1,y1=line['bbox']
            if y0>=page.rect.height*.93 and re.fullmatch(r'\d{1,3}',text):
                candidates.append((text,[x0,y0,x1,y1]))
    return candidates[0] if len(candidates)==1 else (None,None)


def literal_references(text,labels,source_page):
    """Conservatively accept singular page N, excluding lists/ranges/self-links."""
    accepted=[];rejected=[]
    for match in re.finditer(r'\bpage\s+(\d{1,3})\b',text,re.I):
        label=match.group(1)
        tail=text[match.end():match.end()+30]
        if re.match(r'\s*(?:[-–—]\s*\d|,\s*\d|and\s+\d)',tail,re.I):
            rejected.append({'label':label,'reason':'multi_target_list_or_range'});continue
        targets=labels.get(label,[])
        if len(targets)!=1 or targets[0]==source_page:
            rejected.append({'label':label,'reason':'missing_ambiguous_or_self_target'});continue
        # Keep a literal, reproducible context window; it is not an answer label.
        quote=text[max(0,match.start()-140):min(len(text),match.end()+80)]
        accepted.append({'label':label,'target':targets[0],'quote':quote,'offset':match.start()})
    return accepted,rejected


def build(source,output,diagnostic):
    if (source/'manifest.json').exists():raise ValueError('Already frozen; use a new version for changes')
    docs=json.loads((source/'acquisition.json').read_text(encoding='utf8'))
    if any(d['status']!='available' for d in docs):raise ValueError('Incomplete acquisition')
    pages=[];references=[];rejections=[];page_maps=[]
    for doc in docs:
        pdfpath=source/doc['file']
        if sha(pdfpath)!=doc['sha256']:raise ValueError('Changed PDF')
        labels=defaultdict(list);docpages={}
        with fitz.open(pdfpath) as pdf:
            for number,page in enumerate(pdf,1):
                label,bbox=printed_footer(page)
                if label:labels[label].append(number)
                page_maps.append({'doc_id':doc['doc_id'],'physical_page':number,'printed_page_label':label,'footer_bbox':bbox})
                text=' '.join(page.get_text('text',sort=True).split())
                row={'record_id':f'rag_page::{doc["doc_id"]}::p{number:04d}','record_type':'page',
                     'doc_id':doc['doc_id'],'page':number,'source_file':pdfpath.name,'source_url':doc['url'],
                     'citation':f'{pdfpath.name}#page={number}','authority':'OPM-published plan brochure',
                     'content_type':'public_health_plan_wording','insurance_domain':['health'],
                     'source_origin':'archived_official_pdf','document_family_id':doc['document_family_id'],
                     'text':text,'printed_page_label':label,'plan_year':2026}
                if text:docpages[number]=row
            for number,row in docpages.items():
                accepted,rejected=literal_references(row['text'],labels,number)
                rejections.extend({'doc_id':doc['doc_id'],'source_page':number,**r} for r in rejected)
                used=set()
                for item in accepted:
                    if item['target'] in used:continue
                    used.add(item['target'])
                    target=docpages[item['target']]
                    ref={'target_doc_id':doc['doc_id'],'target_physical_page':item['target'],
                         'target_printed_page_label':item['label'],'evidence_span':item['quote'],
                         'target_evidence_span':target['text'][:180],'source_pdf_sha256':doc['sha256'],
                         'annotation_author':'Automated literal page reference + PDF footer geometry; no expert adjudication',
                         'reference_origin':'automated_pdf_footer_mapping'}
                    row.setdefault('source_references',[]).append(ref)
                    references.append({'id':f'opm_ref_{doc["doc_id"]}_{number:03d}_{item["target"]:03d}',
                        'source_doc_id':doc['doc_id'],'source_physical_page':number,
                        'split':doc['split'],'document_family_id':doc['document_family_id'],**ref})
        pages.extend(docpages.values())
    snippets=[]
    for row in pages:
        words=row['text'].split()
        for chunk,start in enumerate(range(0,len(words),120),1):
            wordslice=words[start:start+180]
            if not wordslice:continue
            snippets.append({k:v for k,v in {**row,'record_id':row['record_id'].replace('rag_page','rag_snippet')+f'::c{chunk:03d}',
                'record_type':'snippet','parent_page_id':row['record_id'],'chunk_id':chunk,'text':' '.join(wordslice)}.items()
                if k not in ['source_references','printed_page_label']})
    write_jsonl(pages,source/'rag_pages.jsonl');write_jsonl(snippets,source/'rag_snippets.jsonl')
    write_json(references,source/'reference_annotations.json');write_json(page_maps,source/'printed_page_maps.json')
    write_json(rejections,source/'rejected_references.json')
    counts={'documents':len(docs),'document_families':len({d['document_family_id'] for d in docs}),
            'physical_pdf_pages':sum(d['physical_pages'] for d in docs),'text_page_records':len(pages),
            'snippets':len(snippets),'unique_directed_page_references':len(references),
            'reference_cases_by_split':dict(Counter(r['split'] for r in references)),
            'mapped_printed_pages':sum(bool(r['printed_page_label']) for r in page_maps)}
    manifest={'version':'policy_wordings_v1','created_utc':datetime.now(timezone.utc).isoformat(),'counts':counts,
              'files':{str(p.relative_to(source)).replace('\\','/'):sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
              'qa_labels_used':False,'reference_extraction':'Singular page N; unique numeric footer below 93% page height; skip lists/ranges and self-links; deduplicate source-target',
              'limitations':['Eight plan brochures from seven plan families, all use an OPM format; not eight independent document genres.',
                 'Official archived benefit wordings; not personal issued policies, coverage decisions, or current advice.',
                 'Links validate published references and footer mappings, not legal applicability or reference correctness.',
                 'Native PDF text extraction can flatten tables; no VLM/table semantic accuracy is established.']}
    write_json(manifest,source/'manifest.json')
    old=ROOT/'data/research_corpus/source_linked_public_v1'
    oldmanifest=json.loads((old/'manifest.json').read_text(encoding='utf8'))
    for name,digest in oldmanifest['files'].items():
        if sha(old/name)!=digest:raise ValueError('Changed original corpus')
    output.mkdir(parents=True,exist_ok=False)
    allpages=read_jsonl(old/'rag_pages.jsonl')+pages
    allsnippets=read_jsonl(old/'rag_snippets.jsonl')+snippets
    write_jsonl(allpages,output/'rag_pages.jsonl');write_jsonl(allsnippets,output/'rag_snippets.jsonl')
    # Keep previews inside the explicit corpus root, using the existing safe
    # resolver rather than broadening access to arbitrary repository files.
    (output/'sources').mkdir()
    for source_folder in [source/'sources',ROOT/'data/benchmarks/research_v2/sources']:
        for path in source_folder.glob('*.pdf'):
            shutil.copyfile(path,output/'sources'/path.name)
    write_json({'version':'scaled_public_v2','created_utc':datetime.now(timezone.utc).isoformat(),
                'counts':{'documents':len({r['doc_id'] for r in allpages}),'page_records':len(allpages),
                   'pdf_text_pages':sum(not r['source_file'].endswith('.html') for r in allpages),
                   'html_records':sum(r['source_file'].endswith('.html') for r in allpages),'snippets':len(allsnippets),
                   'explicit_page_references':len(references)+6},
                'files':{str(p.relative_to(output)).replace('\\','/'):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
                'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [old/'manifest.json',source/'manifest.json']},
                'limitations':manifest['limitations']+['FAQ answers are evaluated separately, never counted as PDF pages.']},output/'manifest.json')
    diagnostic.mkdir(parents=True,exist_ok=False)
    cases=[]
    for ref in references:
        cases.append({'id':ref['id'],'split':ref['split'],'document_family_id':ref['document_family_id'],
                      'document_scope':[ref['source_doc_id']], 'anchor_page':ref['source_physical_page'],
                      'target_page':ref['target_physical_page'],'anchor_text':ref['evidence_span'],
                      'task':'Follow the literal page reference from the supplied anchor; graph wiring diagnostic, not QA.',
                      'answerable':True})
    write_jsonl(cases,diagnostic/'cases.jsonl')
    write_json({'version':'opm_reference_navigation_v1','created_utc':datetime.now(timezone.utc).isoformat(),
                'cases':len(cases),'counts_by_split':dict(Counter(c['split'] for c in cases)),
                'families_by_split':{s:sorted({c['document_family_id'] for c in cases if c['split']==s}) for s in ['dev','test']},
                'files':{'cases.jsonl':sha(diagnostic/'cases.jsonl')},'corpus_manifest_sha256':sha(source/'manifest.json'),
                'limitations':['Labels and edges share literal reference extraction; this is an integration diagnostic, not independent graph extraction accuracy.',
                  'Anchor is supplied. Does not measure end-to-end natural-question answering or support legal precedence.',
                  'No synthetic numeric variants. Multiple references from the same plan are correlated; do not use question-level confidence intervals.']},diagnostic/'manifest.lock.json')
    print(json.dumps({'new':counts,'combined':json.loads((output/'manifest.json').read_text())['counts']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/research_corpus/policy_wordings_v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/research_corpus/scaled_public_v2')
    p.add_argument('--diagnostic',type=Path,default=ROOT/'data/benchmarks/opm_reference_navigation_v1')
    args=p.parse_args();build(args.source,args.output,args.diagnostic)
