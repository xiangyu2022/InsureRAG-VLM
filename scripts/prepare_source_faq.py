"""Extract publisher question headings and adjacent answers without generating labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.source_snapshot_io import verified_metadata
LOCAL=ROOT/'reports/source_holdout_v1/local'
ALLOWED={'ccpc_ireland','fsra_ontario','oregon_dfr','hia_ireland','australia_privatehealth'}

def norm(s): return ' '.join(re.findall(r'\w+',s.casefold()))
def digest(s): return hashlib.sha256(s.encode('utf8')).hexdigest()
def clean(s): return ' '.join(s.split())

def extract_html(raw):
    soup=BeautifulSoup(raw,'html.parser')
    main=soup.find('main') or soup.select_one('#main-content') or soup.select_one('#main') or soup
    for tag in main.find_all(['script','style','nav','header','footer']): tag.decompose()
    headings=[]
    for n in main.find_all(['h1','h2','h3','h4','h5','h6','summary','strong','dt']):
        text=clean(n.get_text(' ',strip=True))
        if n.name=='strong' and (n.parent.name!='p' or clean(n.parent.get_text(' ',strip=True))!=text): continue
        if n.find_parent(['h1','h2','h3','h4','h5','h6','summary']): continue
        if text.endswith('?') and 12<=len(text)<=400 and not re.search(r'(more questions|was this|find what|helpful|help us|how can we help)',text,re.I):
            headings.append(n)
    output=[]
    for n in headings:
        if n.name=='dt':
            sibling=n.find_next_sibling()
            if sibling is not None and sibling.name=='dd':
                answer=clean(sibling.get_text(' ',strip=True))
                if len(answer.split())>=12:
                    output.append({'question':clean(n.get_text(' ',strip=True)),'answer':answer,'heading_tag':'dt'})
            continue
        parts=[]; started=False
        for x in n.next_elements:
            if isinstance(x,Tag) and x in n.descendants: continue
            if isinstance(x,NavigableString) and n in x.parents: continue
            if isinstance(x,Tag) and (x.name in ['h1','h2','h3','h4','h5','h6','summary'] or any(x is h for h in headings)): break
            if isinstance(x,Tag) and x.name in ['footer','nav']: break
            if isinstance(x,NavigableString) and main in x.parents and not x.find_parent(['script','style','button']):
                text=clean(str(x))
                if text: parts.append(text)
        answer=clean(' '.join(parts))
        if len(answer.split())>=12 and len(answer)<25000:
            output.append({'question':clean(n.get_text(' ',strip=True)),'answer':answer,'heading_tag':n.name})
    return output

def chunk_text(text,max_chars=2000):
    if type(max_chars) is not int or max_chars<1:raise ValueError('Chunk budget must be a positive integer')
    words=text.split(); chunks=[]; current=[]; size=0
    for word in words:
        if len(word)>max_chars:raise ValueError('A token exceeds the word-boundary chunk budget')
        if current and size+1+len(word)>max_chars:
            chunks.append(' '.join(current)); current=[]; size=0
        current.append(word); size+=len(word)+(1 if size else 0)
    if current:chunks.append(' '.join(current))
    return chunks

def collect_pairs(acquisition):
    out=[]; sources=[]; seen={};duplicates=0
    if not acquisition.is_dir():raise ValueError('Acquisition directory does not exist')
    for path in sorted(acquisition.glob('*.json')):
        m=json.loads(path.read_text(encoding='utf8'))
        if m['group'] not in ALLOWED or not m.get('artifact','').endswith('.html'):continue
        if any(t in m['url'] for t in ['terms','/legal','policies']): continue
        m=verified_metadata(path)
        if m.get('error'):raise ValueError('An eligible snapshot has a recorded acquisition error')
        pairs=extract_html((path.parent/m['artifact']).read_bytes())
        sources.append({'publisher':m['group'],'url':m['final_url'],'sha256':m['sha256'],'pairs':len(pairs)})
        for pair in pairs:
            key=m['group']+'|'+norm(pair['question'])
            if key in seen:
                if clean(seen[key])!=clean(pair['answer']):raise ValueError('Conflicting answers for duplicate publisher/question ID: '+digest(key)[:20])
                duplicates+=1;continue
            seen[key]=pair['answer']
            out.append({'id':digest(key)[:20],'publisher':m['group'],'source_url':m['final_url'],
                        'source_sha256':m['sha256'],'source_title':m['title'],**pair})
    return out,{'counts':dict(Counter(r['publisher'] for r in out)),'sources':sources,'identical_duplicate_questions':duplicates}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-dir',type=Path,default=LOCAL);args=parser.parse_args()
    paths=[args.local_dir/'extracted_pairs.json',args.local_dir/'extraction_summary.json']
    if any(p.exists() for p in paths):raise ValueError('Extraction artifacts already exist; preserve frozen data and use a fresh workspace')
    out,summary=collect_pairs(args.local_dir/'acquisition')
    for path,value in zip(paths,[out,summary]):
        with path.open('x',encoding='utf8') as handle:handle.write(json.dumps(value,indent=2,ensure_ascii=False))
    print(json.dumps({'pairs':len(out),'by_publisher':dict(Counter(r['publisher'] for r in out)),
                      'answer_words':sorted(len(r['answer'].split()) for r in out)[::max(1,len(out)//10)]},indent=2))
if __name__=='__main__':main()
