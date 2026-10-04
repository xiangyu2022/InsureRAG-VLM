import json
import hashlib
import pytest
import requests
from unittest.mock import Mock
from scripts.preflight_holdout1000_sources import fetch
from scripts.holdout_access_guard import AccessGuard,ConservativeRobots,REVIEW_AGENTS
from scripts.acquire_holdout1000_documents import relevant,acquire


def policy():
    return {'id':'synthetic','terms':'https://example.invalid/legal',
            'seeds':['https://example.invalid/consumer/safe'],
            'prefixes':['/consumer/'],'max_pages':1,'minimum_interval_seconds':0,
            'excluded_prefixes':['/consumer/third-party/'],
            'excluded_path_keywords':['dynamic','search','compare','calculator','insurer','phis']}


def guard(text='',row=None):
    row=row or policy()
    parser=ConservativeRobots(text,['CustomResearch/1.0',*REVIEW_AGENTS])
    return AccessGuard({'example.invalid':parser},row,lambda url:relevant(url,'',row))

class Response:
    def __init__(self,url,status=200,body=b'<main>Public synthetic text.</main>',headers=None):
        self.url=url;self.status_code=status;self.body=body;self.headers=headers or {};self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def iter_content(self,chunk_size):yield self.body

def test_redirect_to_unreviewed_host_is_never_requested(tmp_path,monkeypatch):
    response=Response('https://example.invalid/start',302,headers={'Location':'https://other.invalid/private'})
    get=Mock(return_value=response);monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    result=fetch(response.url,tmp_path)
    assert 'outside reviewed' in result['error'] and get.call_count==1 and response.closed

def test_reviewed_same_host_redirect_and_cache_bytes_are_verified(tmp_path,monkeypatch):
    start='https://example.invalid/start';end='https://example.invalid/final'
    get=Mock(side_effect=[Response(start,302,headers={'Location':'/final'}),Response(end)])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    result=fetch(start,tmp_path);assert result['status']==200 and result['final_url']==end
    assert fetch(start,tmp_path)==result and get.call_count==2
    (tmp_path/result['artifact']).write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError,match='provenance'):fetch(start,tmp_path)

def test_cached_denial_does_not_trigger_retry(tmp_path,monkeypatch):
    response=Response('https://example.invalid/denied',403)
    get=Mock(return_value=response);monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    assert fetch(response.url,tmp_path)['status']==403
    assert fetch(response.url,tmp_path)['status']==403 and get.call_count==1


@pytest.mark.parametrize('agent',['GPTBot','OAI-SearchBot','ChatGPT-User','*'])
def test_named_or_wildcard_disallow_cannot_be_bypassed_by_custom_agent(agent,tmp_path,monkeypatch):
    get=Mock();monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    check=guard('User-agent: CustomResearch\nAllow: /\nUser-agent: '+agent+'\nDisallow: /consumer/')
    result=fetch('https://example.invalid/consumer/safe',tmp_path,url_guard=check)
    assert result['status'] is None and 'robots_disallow' in result['error']
    get.assert_not_called()


@pytest.mark.parametrize('target,robots',[
    ('/consumer/denied','User-agent: GPTBot\nDisallow: https://example.invalid/consumer/denied'),
    ('/consumer/search/results',''),('/consumer/third-party/policy',''),
    ('/private',''),('/consumer/%63alculator',''),
    ('/consumer/%2e%2e/private',''),
])
def test_every_redirect_target_is_guarded_before_request(target,robots,tmp_path,monkeypatch):
    start='https://example.invalid/consumer/safe'
    get=Mock(return_value=Response(start,302,headers={'Location':target}))
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    result=fetch(start,tmp_path,url_guard=guard(robots))
    assert result['status'] is None and result['error'] and get.call_count==1


def test_later_redirect_hop_is_also_guarded(tmp_path,monkeypatch):
    start='https://example.invalid/consumer/safe';middle='https://example.invalid/consumer/middle'
    get=Mock(side_effect=[Response(start,302,headers={'Location':middle}),
                          Response(middle,302,headers={'Location':'/consumer/PHIS/123'})])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    assert fetch(start,tmp_path,url_guard=guard())['status'] is None
    assert get.call_count==2


def test_cached_final_url_is_checked_against_current_robots(tmp_path,monkeypatch):
    start='https://example.invalid/consumer/safe';end='https://example.invalid/consumer/final'
    get=Mock(side_effect=[Response(start,302,headers={'Location':end}),Response(end)])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    original=fetch(start,tmp_path)
    assert original['status']==200
    result=fetch(start,tmp_path,url_guard=guard('User-agent: GPTBot\nDisallow: /consumer/final'))
    assert result['status'] is None and 'robots_disallow' in result['error']
    assert get.call_count==2
    assert fetch(start,tmp_path)==original  # cached snapshot was not rewritten


def test_delay_is_maximum_of_all_matching_groups_and_policy():
    parser=ConservativeRobots('User-agent: *\nCrawl-delay: 2\nUser-agent: GPTBot\nCrawl-delay: 7.5\n'
                             'User-agent: CustomResearch\nCrawl-delay: 3\nUser-agent: Unrelated\nCrawl-delay: 999',
                             ['CustomResearch',*REVIEW_AGENTS])
    now=[100.0];sleeps=[]
    def sleep(seconds):sleeps.append(seconds);now[0]+=seconds
    row=policy();check=AccessGuard({'example.invalid':parser},row,lambda url:True,clock=lambda:now[0],sleep=sleep)
    check(row['seeds'][0]);now[0]+=1;check(row['seeds'][0])
    assert sleeps==[6.5]
    check.minimum=10;check(row['seeds'][0]);assert sleeps[-1]==10


def test_safe_path_and_absolute_disallow_globs():
    check=guard('User-agent: GPTBot\nDisallow: https://example.invalid/consumer/PHIS/*\n'
                'User-agent: *\nDisallow: /consumer/secret$')
    check('https://example.invalid/consumer/safe')
    with pytest.raises(ValueError,match='robots_disallow'):check('https://example.invalid/consumer/PHIS/example')
    with pytest.raises(ValueError,match='robots_disallow'):check('https://example.invalid/consumer/secret')


@pytest.mark.parametrize('deny_terms,hash_changed',[(True,False),(False,True),(False,False)])
def test_acquire_fetches_robots_before_terms_and_guards_terms(tmp_path,monkeypatch,deny_terms,hash_changed):
    row=policy();robots='https://example.invalid/robots.txt';terms=row['terms'];seed=row['seeds'][0]
    body=b'User-agent: GPTBot\nDisallow: /legal' if deny_terms else b'User-agent: *\nDisallow: /consumer/secret'
    get=Mock(side_effect=[Response(robots,body=body),Response(terms),Response(seed)])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    monkeypatch.setattr('scripts.acquire_holdout1000_documents.LOCAL',tmp_path)
    if hash_changed:row.update(terms_sha256='0'*64,require_terms_sha256=True)
    result=acquire(row)
    urls=[c.args[0] for c in get.call_args_list]
    if deny_terms:assert urls==[robots] and result['blocked']=='terms_unavailable_or_disallowed'
    elif hash_changed:assert urls==[robots,terms] and result['blocked']=='blocked_terms_changed'
    else:assert urls==[robots,terms,seed] and result['pages']==1


@pytest.mark.parametrize('has_expected_hash',[False,True])
def test_required_terms_hash_must_be_present_and_match(tmp_path,monkeypatch,has_expected_hash):
    row=policy();row['require_terms_sha256']=True
    terms_body=b'<main>Reviewed synthetic license.</main>'
    if has_expected_hash:row['terms_sha256']=hashlib.sha256(terms_body).hexdigest()
    get=Mock(side_effect=[Response('https://example.invalid/robots.txt',body=b'User-agent: *\nDisallow:'),
                          Response(row['terms'],body=terms_body),Response(row['seeds'][0])])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    monkeypatch.setattr('scripts.acquire_holdout1000_documents.LOCAL',tmp_path)
    result=acquire(row)
    assert get.call_count==(3 if has_expected_hash else 2)
    assert result['blocked']==(None if has_expected_hash else 'blocked_terms_changed')


@pytest.mark.parametrize('failure',[requests.exceptions.ProxyError('synthetic proxy failure'),
                                  requests.exceptions.ConnectionError('synthetic connection failure'),
                                  requests.exceptions.Timeout('synthetic timeout')])
def test_acquisition_failure_stops_source_without_requesting_remaining_queue(tmp_path,monkeypatch,failure):
    row=policy();row['seeds'].append('https://example.invalid/consumer/second');row['max_pages']=3
    get=Mock(side_effect=[Response('https://example.invalid/robots.txt',body=b'User-agent: *\nDisallow:'),
                          Response(row['terms']),failure])
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    monkeypatch.setattr('scripts.acquire_holdout1000_documents.LOCAL',tmp_path)
    result=acquire(row)
    assert result['blocked']=='acquisition_error' and result['robots_exclusions']==0
    assert result['acquisition_error']['error_kind']=='acquisition_error'
    assert get.call_count==3 and result['queue_remaining']==1


def test_body_parser_failure_is_acquisition_error_not_guard_exclusion(tmp_path,monkeypatch):
    get=Mock(return_value=Response('https://example.invalid/consumer/safe'))
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.BeautifulSoup',Mock(side_effect=RuntimeError('parser failed')))
    result=fetch('https://example.invalid/consumer/safe',tmp_path,url_guard=guard())
    assert result['status'] is None and result['error_kind']=='acquisition_error'


@pytest.mark.parametrize('terms_redirect,content_redirect',[(False,False),(True,False),(False,True)])
def test_separate_explicit_terms_host_does_not_expand_content_scope(tmp_path,monkeypatch,terms_redirect,content_redirect):
    row=policy();row['terms']='https://legal.invalid/legal'
    responses=[Response('https://example.invalid/robots.txt',body=b'User-agent: *\nDisallow:'),
               Response('https://legal.invalid/robots.txt',body=b'User-agent: *\nDisallow:')]
    responses.append(Response(row['terms'],302,headers={'Location':'/unapproved'}) if terms_redirect else Response(row['terms']))
    if not terms_redirect:
        responses.append(Response(row['seeds'][0],302,headers={'Location':'https://legal.invalid/consumer/data'})
                         if content_redirect else Response(row['seeds'][0]))
    get=Mock(side_effect=responses)
    monkeypatch.setattr('scripts.preflight_holdout1000_sources.requests.get',get)
    monkeypatch.setattr('scripts.acquire_holdout1000_documents.LOCAL',tmp_path)
    result=acquire(row)
    assert get.call_count==(3 if terms_redirect else 4)
    if terms_redirect:assert result['blocked']=='terms_unavailable_or_disallowed'
    elif content_redirect:assert result['pages']==0 and result['robots_exclusions']==1
    else:assert result['blocked'] is None and result['pages']==1
