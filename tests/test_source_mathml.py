import pytest
from bs4 import BeautifulSoup

from src.insurerag_vlm.source_faq_extraction import clean, content_root, extract_original_faq
from src.insurerag_vlm.source_mathml import preserve_mathml


def text(html):
    return clean(content_root(BeautifulSoup(html, 'html.parser')).get_text(' ', strip=True))


def test_fraction_operator_survives_in_document_and_faq_evidence():
    raw = '''<main><h2>How is the example benefit calculated?</h2>
    <p>Multiply the eligible annual amount by the assigned-hours fraction
    <math><mo>(</mo><mfrac><mi>Assigned hours</mi><mi>Full hours</mi></mfrac><mo>)</mo></math>
    and retain the stated annual payment limit.</p></main>'''
    body = text(raw)
    answer = extract_original_faq(raw)[0]['answer']
    assert '((Assigned hours) / (Full hours))' in body
    assert answer in body
    assert body[body.index(answer):body.index(answer) + len(answer)] == answer
    assert 'annual payment limit' in answer


def test_nested_fraction_keeps_compound_numerator_and_denominator_grouped():
    raw = '''<main><math><mfrac><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow>
    <mfrac><mn>2</mn><mn>3</mn></mfrac></mfrac></math></main>'''
    assert text(raw) == '((x + y) / (((2) / (3))))'


@pytest.mark.parametrize('thickness', ['0', '0.0', '0px', '0%', '0em', '0rem', '-0.0pt'])
def test_barless_fraction_is_not_misrepresented_as_division(thickness):
    with pytest.raises(ValueError, match='Barless'):
        text(f'<main><math><mfrac linethickness="{thickness}"><mn>8</mn><mn>3</mn></mfrac></math></main>')


@pytest.mark.parametrize('formula', [
    '<mfrac><mn>2</mn></mfrac>',
    '<mfrac><mn>2</mn><mn>3</mn><mn>4</mn></mfrac>',
    '<mfrac>untagged<mn>2</mn><mn>3</mn></mfrac>',
    '<msup><mn>2</mn><mn>3</mn></msup>',
    '<semantics><mi>x</mi><annotation>different text</annotation></semantics>',
    '<mi><mfrac><mn>2</mn><mn>3</mn></mfrac></mi>',
])
def test_unhandled_or_malformed_formula_blocks_extraction(formula):
    with pytest.raises(ValueError, match='MathML'):
        text('<main><math>' + formula + '</math></main>')


def test_math_outside_selected_content_does_not_affect_plain_source():
    assert text('<main><p>Preserve the original benefit conditions.</p></main>'
                '<footer><math><unsupported/></math></footer>') == 'Preserve the original benefit conditions.'


def test_namespace_prefix_and_failed_second_formula_do_not_leave_partial_rewrite():
    assert text('<main><m:math><m:mfrac><m:mi>a</m:mi><m:mi>b</m:mi>'
                '</m:mfrac></m:math></main>') == '((a) / (b))'
    soup = BeautifulSoup('<main><math><mfrac><mi>a</mi><mi>b</mi></mfrac></math>'
                         '<math><msqrt><mi>c</mi></msqrt></math></main>', 'html.parser')
    before = str(soup)
    with pytest.raises(ValueError, match='Unsupported'):
        preserve_mathml(soup.main)
    assert str(soup) == before


def test_direct_math_root_remains_readable_and_repeated_normalization_is_stable():
    soup = BeautifulSoup('<math><mfrac><mn>3</mn><mn>8</mn></mfrac></math>', 'html.parser')
    root = soup.math
    assert preserve_mathml(root) == 1
    assert root.get_text() == '((3) / (8))'
    assert preserve_mathml(root) == 0


def test_inherited_fraction_thickness_requires_review_instead_of_guessing():
    with pytest.raises(ValueError, match='line thickness'):
        text('<main><math><mfrac linethickness="inherit"><mn>3</mn><mn>8</mn></mfrac></math></main>')
