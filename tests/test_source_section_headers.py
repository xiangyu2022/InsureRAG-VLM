import pytest
from bs4 import BeautifulSoup

from src.insurerag_vlm.source_faq_extraction import clean, content_root, extract_original_faq


def test_reviewed_section_headers_survive_without_restoring_site_navigation():
    raw = """<main><header>Site navigation</header><div id="products">
    <header><span>FIXED COVER</span></header><p>Coverage has a fixed stated limit.</p>
    <header><span>ADJUSTABLE COVER</span></header><p>Coverage has a changeable stated limit.</p>
    </div><footer>Footer links</footer></main>"""
    default = clean(content_root(BeautifulSoup(raw, 'html.parser')).get_text(' ', strip=True))
    reviewed = clean(content_root(BeautifulSoup(raw, 'html.parser'),
                                  preserve_headers=('#products > header',)).get_text(' ', strip=True))
    assert default == 'Coverage has a fixed stated limit. Coverage has a changeable stated limit.'
    assert reviewed == ('FIXED COVER Coverage has a fixed stated limit. '
                        'ADJUSTABLE COVER Coverage has a changeable stated limit.')


def test_reviewed_document_and_faq_share_the_same_explicit_header_policy():
    raw = """<main><h2>How does the optional cover work?</h2><div id="answer">
    <header>Optional cover conditions</header><p>The additional cover applies only after
    the insurer accepts the request and collects the stated additional premium.</p>
    </div></main>"""
    options = {'preserve_headers': ('#answer > header',)}
    body = clean(content_root(BeautifulSoup(raw, 'html.parser'), **options).get_text(' ', strip=True))
    answer = extract_original_faq(raw, **options)[0]['answer']
    assert answer.startswith('Optional cover conditions The additional cover')
    assert answer in body


@pytest.mark.parametrize('selector', ['#missing', 'p', 'nav header'])
def test_unreviewable_selector_fails_before_mutating_source(selector):
    soup = BeautifulSoup('<main><!--keep--><nav><header>Navigation</header></nav>'
                         '<p>Retain the original source text.</p></main>', 'html.parser')
    before = str(soup)
    with pytest.raises(ValueError, match='Reviewed header'):
        content_root(soup, preserve_headers=(selector,))
    assert str(soup) == before
