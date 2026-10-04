"""Archive publisher FAQ HTML discovered from the official DOL index."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html.parser import HTMLParser
import hashlib,json,re,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import write_json,sha

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']:self.skip+=1
        if not self.skip and tag in ['p','h1','h2','h3','h4','li','table','tr','br']:self.parts.append('\n\n' if tag!='br' else '\n')
    def handle_endtag(self,tag):
        if tag in ['script','style']:self.skip-=1
        if not self.skip and tag in ['p','h1','h2','h3','h4','li','table','tr']:self.parts.append('\n\n')
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

def run():
    cache=ROOT/'../condition_source_cache';links=json.loads((cache/'discovered_links.json').read_text(encoding='utf8'))
    def one(url):
        name=url.rsplit('/',1)[-1];path=cache/(name+'.html')
        if not path.exists():
            r=requests.get(url,timeout=40);r.raise_for_status();path.write_bytes(r.content)
        raw=path.read_text(encoding='utf8');main=re.search(r'<main\b.*?</main>',raw,re.S|re.I)
        parser=Text();parser.feed(main.group() if main else raw)
        text=''.join(parser.parts);text=re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',text).strip()
        return {'id':'dol_html_'+name,'text':text,'source_url':url,'source_group':'dol-aca:'+name,
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),'html_sha256':sha(path),'text_sha256':hashlib.sha256(text.encode('utf8')).hexdigest(),
                'category':'regulatory-guidance','rights':'US federal government publication; preserve source/date; not a determination of current legal status.'}
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,links))
    out=ROOT/'data/research_corpus/dol_additional_v1';out.mkdir(parents=True,exist_ok=False)
    with (out/'records.jsonl').open('w',encoding='utf8') as handle:
        for row in records:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'records':len(records),'index_sha256':sha(cache/'dol_index.html'),
                'records_sha256':sha(out/'records.jsonl'),'code_sha256':sha(Path(__file__))},out/'manifest.json')
    print(json.dumps({'records':len(records),'characters':sum(len(r['text']) for r in records)}))

if __name__=='__main__':run()
