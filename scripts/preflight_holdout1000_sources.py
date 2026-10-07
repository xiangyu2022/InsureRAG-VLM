"""Bounded public navigation/terms preflight; does not accept any dataset item."""
import argparse,hashlib,json,time,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,urljoin,urldefrag
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.source_snapshot_io import verified_metadata
LOCAL=ROOT/'reports/holdout1000_v1/local'
AGENT='InsureRAG-Research/1.0 (noncommercial local source quality audit)'

class AccessGuardRejected(ValueError):
    pass


def fetch(url,folder,allowed_hosts=None,url_guard=None):
    allowed_hosts=set(allowed_hosts or [urlparse(url).hostname])
    def check_url(target):
        if url_guard:
            try:url_guard(target)
            except ValueError as exc:raise AccessGuardRejected(str(exc)) from exc
    if url_guard:
        try:check_url(url)
        except AccessGuardRejected as exc:return {'url':url,'error':str(exc),'status':None,'error_kind':'access_guard_rejected'}
    key=hashlib.sha256(url.encode()).hexdigest()[:20];meta_path=folder/(key+'.json')
    if meta_path.exists():
        meta=verified_metadata(meta_path,expected_url=url)
        if urlparse(meta.get('final_url',url)).hostname not in allowed_hosts:
            return {'url':url,'error':'cached_cross_host_redirect_requires_review','status':None,'error_kind':'access_guard_rejected'}
        if url_guard and meta.get('final_url',url)!=url:
            try:check_url(meta['final_url'])
            except AccessGuardRejected as exc:return {'url':url,'error':str(exc),'status':None,'error_kind':'access_guard_rejected'}
        return meta
    meta={'url':url,'acquired_utc':datetime.now(timezone.utc).isoformat()}
    try:
        current=url
        for attempt in range(6):
          with requests.get(current,headers={'User-Agent':AGENT},timeout=25,stream=True,allow_redirects=False) as response:
            meta.update(status=response.status_code,final_url=response.url)
            if response.status_code in {301,302,303,307,308}:
                target=urljoin(current,response.headers.get('Location',''));parsed=urlparse(target)
                if parsed.hostname not in allowed_hosts or parsed.scheme!='https' or parsed.username or parsed.password:
                    raise AccessGuardRejected('Redirect outside reviewed HTTPS host scope')
                if attempt==5:raise ValueError('Redirect limit exceeded')
                check_url(target)
                current=target;continue
            if response.status_code!=200:meta['error']='HTTP_'+str(response.status_code)
            else:
                chunks=[];total=0
                for chunk in response.iter_content(65536):
                    total+=len(chunk)
                    if total>5_000_000:raise ValueError('Preflight body exceeds 5 MB')
                    chunks.append(chunk)
                raw=b''.join(chunks);artifact=key+'.body'
                (folder/artifact).write_bytes(raw)
                soup=BeautifulSoup(raw,'html.parser')
                links=[]
                for a in soup.find_all(['a','va-link'],href=True):
                    href=urldefrag(urljoin(response.url,a['href']))[0]
                    if urlparse(href).scheme in {'http','https'}:links.append({'title':a.get_text(' ',strip=True) or a.get('text',''),'url':href})
                for element in soup(['script','style','noscript']):element.decompose()
                text=soup.get_text('\n',strip=True)
                (folder/(key+'.txt')).write_text(text,encoding='utf8')
                meta.update(artifact=artifact,sha256=hashlib.sha256(raw).hexdigest(),bytes=total,
                            title=soup.title.get_text(' ',strip=True) if soup.title else '',links=links)
            break
    except Exception as exc:
        meta['error']=type(exc).__name__+': '+str(exc)[:180]
        meta['error_kind']='access_guard_rejected' if isinstance(exc,AccessGuardRejected) else 'acquisition_error'
        # A denied redirect is not a reusable successful acquisition.
        meta['status']=None
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return meta

def preflight(row):
    folder=LOCAL/'preflight'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    base=urlparse(row['seed']);robot_url=base.scheme+'://'+base.netloc+'/robots.txt'
    robots=fetch(robot_url,folder);out={'publisher':row['id'],'jurisdiction':row['jurisdiction'],'robots':robots,'pages':[],'status':'pending_terms_review'}
    if robots.get('status') in {401,403,429}:
        out['status']='blocked_robots_no_retry';return out
    parser=None
    if robots.get('status')==200:
        parser=RobotFileParser();parser.parse((folder/robots['artifact']).read_text(encoding='utf8',errors='replace').splitlines())
    for url in dict.fromkeys([row.get('terms'),row['seed']]):
        if not url:continue
        if parser and not parser.can_fetch(AGENT,url):
            out['pages'].append({'url':url,'error':'robots_disallow'});continue
        time.sleep(1)
        result=fetch(url,folder);out['pages'].append(result)
        if result.get('status') in {401,403,429}:
            out['status']='blocked_http_no_retry';break
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--publishers',nargs='*');a=p.parse_args()
    rows=json.loads((ROOT/'reports/holdout1000_v1/source_candidates.json').read_text(encoding='utf8'))
    if a.publishers:rows=[r for r in rows if r['id'] in a.publishers]
    result=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(preflight,rows):
            result.append(row)
            print(json.dumps({'publisher':row['publisher'],'status':row['status'],
                              'pages':[{'url':p['url'],'status':p.get('status'),'error':p.get('error'),'links':len(p.get('links',[]))} for p in row['pages']]}),flush=True)
    LOCAL.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    (LOCAL/('preflight_'+stamp+'.json')).write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
