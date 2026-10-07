"""Conservative exact/history-document preflight; never a full contamination clearance."""
import argparse,hashlib,json,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_source_faq import norm
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior audit')
    rows=[json.loads(l) for l in a.candidates.read_text(encoding='utf8').splitlines() if l.strip()]
    wanted=defaultdict(list);urls=defaultdict(dict);found=defaultdict(dict);urlfound={}
    for r in rows:
        for key in ['question','answer']:wanted[norm(r[key])].append((r['id'],key))
        parsed=urlparse(r['source_url']);urlkey=norm(r['source_url']).removeprefix('https ').removeprefix('http ')
        host=norm(parsed.netloc).removeprefix('www ')
        urls[host][urlkey.removeprefix('www ')]=r['document_group']
    patterns={h:re.compile('|'.join(re.escape(u) for u in sorted(paths,key=len,reverse=True))) for h,paths in urls.items()}
    total=0;digest=hashlib.sha256();started=time.perf_counter()
    with a.history.open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():continue
            r=json.loads(raw);text=r['text'];total+=1
            for ident,key in wanted.get(text,[]):found[ident][key]={'origin':r['origin'],'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
            for host,pattern in patterns.items():
                if host not in text:continue
                for match in pattern.finditer(text):urlfound.setdefault(urls[host][match.group()],r['origin'])
    output=[]
    for r in rows:
        hits=found.get(r['id'],{})
        output.append({'id':r['id'],'publisher':r['publisher'],'document_group':r['document_group'],
                       'exact_fields':hits,'historical_document_url_origin':urlfound.get(r['document_group']),
                       'excluded_from_independent_holdout':bool(hits or r['document_group'] in urlfound)})
    report={'method':'Exact normalized question/answer match plus normalized document URL occurrence in all accessible historical strings; URL presence conservatively excludes the document, even a mere reference.',
            'not_complete':'No near-duplicate/semantic clearance; source-family aliases, text containment and historical access gaps need further checks.',
            'historical_sha256':digest.hexdigest(),'historical_strings':total,'candidate_file_sha256':hashlib.sha256(a.candidates.read_bytes()).hexdigest(),
            'candidates':len(rows),'excluded':sum(r['excluded_from_independent_holdout'] for r in output),
            'remaining_by_publisher':dict(Counter(r['publisher'] for r in output if not r['excluded_from_independent_holdout'])),
            'rows':output,'seconds':time.perf_counter()-started,'accepted_test_items':0}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
