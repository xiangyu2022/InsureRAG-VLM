import pytest
from scripts.extract_holdout1000_candidates import html_rejection_reason


@pytest.mark.parametrize('raw,metadata,reason',[
    (b'PK\x03\x04binary<html>not HTML',{'url':'https://example.invalid/consumer','content_type':'text/html'},'binary_signature_not_html'),
    (b'<html>mislabelled download</html>',{'url':'https://example.invalid/consumer','final_url':'https://example.invalid/report.XLSX?version=1'},'non_html_download_extension'),
    (b'<html>wrong media type</html>',{'content_type':'application/vnd.ms-excel'},'non_html_content_type'),
    (b'<html>\x00corruption</html>',{},'binary_nul_requires_dedicated_decoder'),
    (b'plain text without source markup',{},'html_markup_not_detected'),
])
def test_non_html_cannot_enter_the_source_text_corpus(raw,metadata,reason):
    assert html_rejection_reason(raw,metadata)==reason


def test_normal_html_and_xhtml_are_preserved():
    assert html_rejection_reason(b'<!DOCTYPE html><html><p>policy</p></html>',{}) is None
    assert html_rejection_reason(b'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"></html>',{'content_type':'application/xhtml+xml; charset=utf-8'}) is None


@pytest.mark.parametrize('headings,metadata,html_title,expected',[
    ('<h1> </h1><h1>Specific <span>coverage</span></h1>',{'title':'Generic agency'},'Generic site','Specific coverage'),
    ('<h1>&nbsp;<img src="logo.png"></h1>',{'title':'  Cached   title '},'Generic site','Cached title'),
    ('<h1> </h1>',{'title':None},'  Page title  ','Page title'),
    ('',{},None,''),
])
def test_source_title_preserves_nonempty_provenance(headings,metadata,html_title,expected):
    from bs4 import BeautifulSoup
    from scripts.extract_holdout1000_candidates import source_title
    title='' if html_title is None else f'<title>{html_title}</title>'
    soup=BeautifulSoup(f'<html><head>{title}</head><body><main>{headings}</main></body></html>','html.parser')
    assert source_title(soup,soup.main,metadata)==expected


def test_empty_first_heading_does_not_drop_extracted_document_title(tmp_path):
    import hashlib,json
    from scripts.extract_holdout1000_candidates import extract
    raw=b'<main><h1> </h1><h1>Synthetic policy guide</h1><h2>Which losses does this policy cover?</h2><p>This synthetic policy covers eligible fire damage subject to its stated limits.</p></main>'
    folder=tmp_path/'documents'/'us_opm';folder.mkdir(parents=True)
    (folder/'source.html').write_bytes(raw)
    metadata={'status':200,'url':'https://example.invalid/insurance','artifact':'source.html','sha256':hashlib.sha256(raw).hexdigest(),'acquired_utc':'2026-01-01T00:00:00Z','title':'Generic agency'}
    (folder/'source.json').write_text(json.dumps(metadata),encoding='utf8')
    documents,candidates,_=extract(tmp_path)
    assert len(documents)==len(candidates)==1
    assert documents[0]['source_title']==candidates[0]['source_title']=='Synthetic policy guide'
    evidence=candidates[0]['evidence'][0]
    assert documents[0]['text'][evidence['start']:evidence['end']]==candidates[0]['answer']
