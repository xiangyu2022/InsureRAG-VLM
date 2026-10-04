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
