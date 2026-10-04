"""Read only explicit project roots; record denied paths without retry/bypass.

The roots JSON maps portable aliases to local directories and an optional
exclude_relative list. All JSON string values are screened, not a field-name
allowlist. Original contents and full local paths remain untracked.
"""
import argparse,collections,hashlib,json,os,re
from pathlib import Path
from urllib.parse import urlparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    specs=json.loads(a.roots.read_text(encoding='utf8'))
    hosts=collections.Counter();files=[];failures=[];hashes=set();strings={};questions={};exclusions=[]
    def walk(obj,key='',origin=''):
        if isinstance(obj,dict):
            for k,v in obj.items():walk(v,k,origin)
        elif isinstance(obj,list):
            for v in obj:walk(v,key,origin)
        elif isinstance(obj,str):
            for url in re.findall(r'https?://[^\s<>"\\]+',obj):
                try:host=urlparse(url).hostname
                except ValueError:host='malformed_url'
                hosts[host or 'unknown']+=1
            if len(obj.strip())>=12:
                normalized=' '.join(re.findall(r'\w+',obj.casefold()))
                if normalized:strings.setdefault(normalized,origin)
                if key in {'question','query','instruction','question_text'} and normalized:questions.setdefault(normalized,origin)
    for spec in specs:
        alias=spec['alias'];root=Path(spec['path']);excluded=set(spec.get('exclude_relative',[]))
        exclusions.extend({'root':alias,'relative':s,'reason':'Explicitly excluded; known access restriction or demo upload scope'} for s in sorted(excluded))
        if not root.is_dir():failures.append({'path':alias,'error':'Root unavailable'});continue
        def denied(error):
            path=Path(error.filename)
            failures.append({'path':alias+'/'+str(path.relative_to(root)).replace('\\','/'),'error':type(error).__name__})
        for directory,dirs,names in os.walk(root,onerror=denied):
            directory=Path(directory)
            dirs[:]=sorted(d for d in dirs if str((directory/d).relative_to(root)).replace('\\','/') not in excluded)
            for name in sorted(names):
                path=directory/name
                if path.suffix not in {'.json','.jsonl'}:continue
                rel=alias+'/'+str(path.relative_to(root)).replace('\\','/')
                try:
                    h=hashlib.sha256()
                    with path.open('rb') as f:
                        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                    digest=h.hexdigest();files.append({'path':rel,'bytes':path.stat().st_size,'sha256':digest,'duplicate_file':digest in hashes})
                    if digest in hashes:continue
                    hashes.add(digest)
                    with path.open(encoding='utf-8-sig') as f:
                        if path.suffix=='.jsonl':
                            for line in f:
                                if line.strip():walk(json.loads(line),origin=rel)
                        else:walk(json.load(f),origin=rel)
                except (OSError,ValueError) as exc:failures.append({'path':rel,'error':type(exc).__name__})
                if len(files)%100==0:print(json.dumps({'files':len(files),'unique_texts':len(strings)}),flush=True)
    report={'files':files,'failures':failures,'exclusions':exclusions,'unique_file_count':len(hashes),'text_count':len(strings),
            'question_count':len(questions),'hosts':dict(hosts),'roots_aliases':[s['alias'] for s in specs],
            'scope':'All string values in readable JSON/JSONL in explicitly listed project data and report roots; no PDF bytes or inaccessible paths read.'}
    (a.output/'historical_inventory.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for name,values in [('historical_texts',strings),('historical_questions',questions)]:
        with (a.output/(name+'.jsonl')).open('w',encoding='utf8') as f:
            for value,origin in values.items():f.write(json.dumps({'text':value,'origin':origin},ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'files','hosts'}},indent=2),flush=True)
if __name__=='__main__':main()
