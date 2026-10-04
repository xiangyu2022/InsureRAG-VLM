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
