import json
from unittest.mock import Mock
from scripts.preflight_holdout1000_sources import fetch

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
