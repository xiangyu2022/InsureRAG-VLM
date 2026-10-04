"""Acquire bounded, licensed public consumer pages; candidates remain unaccepted."""
import argparse,json,hashlib,re,sys,time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,urldefrag
from urllib.robotparser import RobotFileParser
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.preflight_holdout1000_sources import fetch,LOCAL,AGENT
from scripts.holdout_access_guard import AccessGuard,ConservativeRobots,REVIEW_AGENTS,excluded_path

def relevant(url,title,policy):
    p=urlparse(url)
    allowed_hosts=policy.get('allowed_hosts',[urlparse(policy['seeds'][0]).hostname])
    if p.scheme!='https' or p.query or p.hostname not in allowed_hosts:return False
    if excluded_path(url,policy):return False
    if not any(p.path.lower().startswith(x.lower()) for x in policy['prefixes']):return False
    if re.search(r'\.(pdf|docx?|xlsx?|zip|jpg|png|mp4|xml)$',p.path,re.I):return False
    if re.search(r'(spanish|espanol|chinese|vietnamese|korean|tagalog|contact|subscribe|complaint.?form|find.?agent|license.?lookup)',url+' '+title,re.I):return False
    if policy['id']=='newyork_dfs' and re.search(r'banking|mortgage|foreclosure|student|credit|debt|bail|holocaust|virtual_currency',p.path,re.I):return False
    if policy['id']=='us_va' and p.path.startswith('/resources/') and not re.search(r'insurance|\b[sv]gli\b|\bfsgli\b|\btsgli\b|\bvalife\b',url+' '+title,re.I):return False
    return True

def acquire(policy):
    group=policy['id'];folder=LOCAL/'documents'/group;folder.mkdir(parents=True,exist_ok=True)
    preflight=LOCAL/'preflight'/group
    preflight.mkdir(parents=True,exist_ok=True)
    content_hosts=policy.get('allowed_hosts',[urlparse(policy['seeds'][0]).hostname])
    terms_host=urlparse(policy['terms']).hostname
    parsers={}
    for host in dict.fromkeys([*content_hosts,terms_host]):
        robot_url='https://'+host+'/robots.txt'
        def robots_scope(url):
            if url!=robot_url:raise ValueError('Robots redirect requires separate review')
        robot=fetch(robot_url,preflight,[host],url_guard=robots_scope)
        if robot.get('status')==200:
            raw=(preflight/robot['artifact']).read_text(encoding='utf8',errors='replace')
            if '<html' in raw.lower() or '<!doctype' in raw.lower():raise ValueError('Ambiguous HTML robots response: '+group)
            parser=ConservativeRobots(raw,[AGENT,*REVIEW_AGENTS,*policy.get('robots_agents',[])])
        elif robot.get('status')==404:parser=ConservativeRobots('',[])
        else:return {'publisher':group,'blocked':'robots_unavailable','pages':[]}
        parsers[host]=parser
    guard=AccessGuard(parsers,policy,lambda url:urldefrag(url)[0]==urldefrag(policy['terms'])[0])
    # Include the robots request in the first post-robots wait on every host.
    guard.last_access={host:guard.clock() for host in parsers}
    terms=fetch(policy['terms'],preflight,[terms_host],url_guard=guard)
    if terms.get('status')!=200:return {'publisher':group,'blocked':'terms_unavailable_or_disallowed','pages':[]}
    if policy.get('require_terms_sha256') and (not policy.get('terms_sha256') or terms.get('sha256')!=policy['terms_sha256']):
        return {'publisher':group,'blocked':'blocked_terms_changed','pages':[]}
    guard.content_check=lambda url:relevant(url,'',policy)
    queue=deque((u,0) for u in policy['seeds']);seen=set();pages=[];excluded=[];blocked=None;acquisition_error=None
    while queue and len(pages)<policy['max_pages']:
        url,depth=queue.popleft();url=urldefrag(url)[0]
        if url in seen:continue
        seen.add(url)
        try:m=fetch(url,folder,content_hosts,url_guard=guard)
        except Exception as exc:
            m={'url':url,'status':None,'error_kind':'acquisition_error','error':str(exc)}
        if m.get('status') is None:
            if m.get('error_kind')=='access_guard_rejected':
                excluded.append({'url':url,'reason':m['error'],'error_kind':m['error_kind']});continue
            blocked='acquisition_error';acquisition_error=m;break
        m['discovery_depth']=depth;pages.append(m)
        if m.get('status') in {401,403,429}:blocked='HTTP_'+str(m['status']);break
        if m.get('status')==200 and depth<3:
            links=[a for a in m.get('links',[]) if relevant(a['url'],a['title'],policy)]
            links.sort(key=lambda a:(0 if re.search('faq|frequent|question',a['url']+' '+a['title'],re.I) else 1,a['url']))
            queue.extend((a['url'],depth+1) for a in links)
        if len(pages)%10==0:print(json.dumps({'publisher':group,'pages':len(pages),'queued':len(queue),'status':'candidate_acquisition'}),flush=True)
    report={'publisher':group,'completed_utc':datetime.now(timezone.utc).isoformat(),'terms_sha256':terms['sha256'],
            'blocked':blocked,'acquisition_error':acquisition_error,'pages':[{'url':m['url'],'final_url':m.get('final_url'),'sha256':m.get('sha256'),'status':m.get('status'),'error':m.get('error'),'depth':m['discovery_depth']} for m in pages],
            'robots_exclusions':excluded,'queue_remaining':len(queue),'accepted_test_items':0}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    (folder/('acquisition_'+stamp+'.report')).write_text(json.dumps(report,indent=2),encoding='utf8')
    return {k:v for k,v in report.items() if k not in {'pages','robots_exclusions'}}|{'pages':len(pages),'robots_exclusions':len(excluded)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--publishers',nargs='*');a=p.parse_args()
    policies=json.loads((ROOT/'reports/holdout1000_v1/source_approvals.json').read_text(encoding='utf8'))
    if a.publishers:policies=[p for p in policies if p['id'] in a.publishers]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(acquire,policies):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
