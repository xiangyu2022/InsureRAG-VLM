"""Acquire explicitly listed public publisher pages; no recursive crawling."""
import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.source_snapshot_io import verified_metadata
LOCAL = ROOT / 'reports/source_holdout_v1/local'

def sha(data): return hashlib.sha256(data).hexdigest()

def fetch(row,output_dir=None,max_bytes=10_000_000):
    group, url = row['group'], row['url']
    parsed=urlparse(url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Registry requires an explicit public HTTP(S) URL without credentials')
    out = Path(output_dir) if output_dir is not None else LOCAL / 'acquisition'
    out.mkdir(parents=True, exist_ok=True)
    key = sha(url.encode())[:20]
    meta_path = out / (key + '.json')
    if meta_path.exists(): return verified_metadata(meta_path,group,url)
    if any((out/(key+ext)).exists() for ext in ['.html','.pdf']):
        raise ValueError('Orphan snapshot exists; preserve it and use a fresh reviewed acquisition directory')
    meta = {'group': group, 'url': url, 'acquired_utc': datetime.now(timezone.utc).isoformat()}
    try:
        with requests.get(url, timeout=25, stream=True, headers={'User-Agent': 'InsureRAG-Research/1.0 (local public-source evaluation)'}) as response:
            meta.update(status_code=response.status_code, final_url=response.url, content_type=response.headers.get('Content-Type',''))
            response.raise_for_status()
            chunks=[];size=0
            for chunk in response.iter_content(chunk_size=65536):
                size+=len(chunk)
                if size>max_bytes:raise ValueError('Page exceeds acquisition limit')
                chunks.append(chunk)
            raw=b''.join(chunks)
        extension='.pdf' if raw.startswith(b'%PDF') else '.html'
        with (out/(key+extension)).open('xb') as handle:handle.write(raw)
        meta.update(bytes=len(raw), sha256=sha(raw), artifact=key+extension)
        if extension=='.pdf':
            import fitz
            with fitz.open(stream=raw,filetype='pdf') as document:
                meta.update(title=document.metadata.get('title',''),pages=len(document),links=[])
            meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
            return meta
        soup = BeautifulSoup(raw, 'html.parser')
        meta['title']=soup.title.get_text(' ',strip=True) if soup.title else ''
        links={}
        for a in soup.find_all('a',href=True):
            href=urljoin(response.url,a['href']); text=a.get_text(' ',strip=True)
            if urlparse(href).scheme in {'http','https'} and re.search('faq|frequent|insurance|copyright|terms|privacy|disclaimer',text+' '+href,re.I):
                links[href]=text[:100]
        meta['links']=[{'url':u,'text':t} for u,t in links.items()]
        meta['question_heading_count']=sum('?' in x.get_text() for x in soup.find_all(['h2','h3','h4','summary','button']))
    except Exception as exc: meta['error']=str(exc)[:300]
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    return meta

def main():
    p=argparse.ArgumentParser(); p.add_argument('registry',type=Path); p.add_argument('--links',action='store_true')
    p.add_argument('--output-dir',type=Path,default=LOCAL/'acquisition');args=p.parse_args()
    rows=json.loads(args.registry.read_text(encoding='utf8'))
    if len({r['url'] for r in rows})!=len(rows):raise ValueError('Registry URLs must be unique; resolve publisher aliases before acquisition')
    errors=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda row:fetch(row,args.output_dir),rows):
            errors+=bool(result.get('error'))
            print(json.dumps({k:v for k,v in result.items() if k!='links'},ensure_ascii=True),flush=True)
            if args.links:
                print(json.dumps(result.get('links',[]),ensure_ascii=True),flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__': main()
