from src.insurerag_vlm.source_faq_extraction import extract_original_faq

A='This synthetic insurance answer has enough words to exercise the intended boundary.'
B='The second independent synthetic answer includes a different rule and its own exception.'
def test_legacy_bold_font_and_toc_are_distinguished():
    raw=f'<main><a href="#one">What does the first policy cover?</a><font><b>What does the first policy cover?</b><br>{A}<br><strong>What is the second policy rule?</strong>{B}</font></main>'
    rows=extract_original_faq(raw)
    assert len(rows)==2 and [r['answer'] for r in rows]==[A,B]

def test_explicit_accordion_resolves_panel_without_picking_tab_navigation():
    raw=f'<main><h3 class="nav-link tab-button">What does this policy cover?</h3><h3><button aria-controls="answer">What does this policy cover?</button></h3><div id="answer">{A}</div><h3>Where does another rule apply?</h3><p>{B}</p></main>'
    rows=extract_original_faq(raw)
    assert len(rows)==2 and rows[0]['answer']==A and rows[0]['extraction_method']=='explicit_accordion_panel'

def test_theme_accordion_does_not_absorb_next_question():
    raw=f'<main><div class="x-acc-item"><button class="x-acc-header">What does this policy cover?</button><div class="x-acc-content">{A}</div></div><div class="x-acc-item"><button class="x-acc-header">Where does another rule apply?</button><div class="x-acc-content">{B}</div></div></main>'
    assert [r['answer'] for r in extract_original_faq(raw)]==[A,B]

def test_subsection_content_is_retained_and_flagged_for_review():
    raw=f'<main><h2>What does this policy cover?</h2><p>{A}</p><h3>Important exception</h3><p>{B}</p><h2>Another topic</h2><p>Not evidence.</p></main>'
    row=extract_original_faq(raw)[0]
    assert A in row['answer'] and B in row['answer'] and 'Not evidence.' not in row['answer']
    assert row['extraction_flags']==['answer_contains_subsections']
def test_legacy_content_column_excludes_comments_and_side_navigation():
    from src.insurerag_vlm.source_faq_extraction import extract_original_faq
    html=b'''<div id="main_content"><div class="content_left_column"><h2>What does the policy cover?</h2>
    <p>The synthetic policy covers damage from fire subject to the stated limits.</p><!-- hidden control marker -->
    </div><div class="content_right_column">Navigation insurance links</div></div>'''
    rows=extract_original_faq(html)
    assert len(rows)==1
    assert rows[0]['answer']=='The synthetic policy covers damage from fire subject to the stated limits.'
def test_role_main_and_processing_instructions_do_not_leak_into_answer():
    from src.insurerag_vlm.source_faq_extraction import extract_original_faq
    html=b'''<div role="main"><h2>Which expenses are covered?</h2>
    <p>The synthetic contract covers eligible repair expenses after the specified deductible.</p>
    <?xml version="1.0"?><a href="/question">Shareable Link to Answer</a></div>
    <aside>Find a plan or phone number here.</aside>'''
    rows=extract_original_faq(html)
    assert rows[0]['answer']=='The synthetic contract covers eligible repair expenses after the specified deductible.'
def test_va_component_attribute_question_keeps_its_own_answer():
    from src.insurerag_vlm.source_faq_extraction import extract_original_faq
    html='<main><va-accordion-item header="Can I receive benefits early?"><p>You must meet both eligibility conditions described in your policy before applying.</p></va-accordion-item><va-accordion-item header="How can I apply for these benefits?"><p>Submit the completed request to the designated benefits office with supporting evidence.</p></va-accordion-item></main>'
    rows=extract_original_faq(html)
    assert len(rows)==2
    assert rows[0]['question']=='Can I receive benefits early?'
    assert 'Submit' not in rows[0]['answer']
    assert rows[1]['extraction_method']=='va_accordion_header_attribute'
