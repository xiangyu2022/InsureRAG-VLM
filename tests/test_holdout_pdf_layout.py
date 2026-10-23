import hashlib
import pytest
from src.insurerag_vlm.holdout_pdf_layout import fingerprint, render_reviewed_words


def fixture():
    pages=[dict(page_number=1,width=100,height=100,words=[
        dict(text='Before',bbox=[1,1,20,10]),dict(text='After',bbox=[51,1,70,10]),
        dict(text='first day',bbox=[1,20,30,30]),dict(text='receipt',bbox=[51,20,80,30])]),
        dict(page_number=2,width=100,height=100,words=[dict(text='Footnote*',bbox=[1,1,40,10])])]
    layout=dict(status='REVIEWED_SOURCE_ONLY_LAYOUT',qa_inputs_used=False,source_sha256='a'*64,
                review_record_sha256='b'*64,word_pages_sha256=fingerprint(pages),pages=[
        dict(page_number=1,groups=[dict(id='left',role='table_cell',word_indices=[0,2]),
                                   dict(id='right',role='table_cell',word_indices=[1,3])]),
        dict(page_number=2,groups=[dict(id='note',role='table_note',word_indices=[0])])])
    return pages,layout


def test_complete_literal_words_reordered_with_exact_offsets_and_geometry():
    pages,layout=fixture();result=render_reviewed_words(pages,layout,source_sha256='a'*64)
    assert result['text']=='Before first day\n\nAfter receipt\n\f\nFootnote*'
    assert len(result['word_spans'])==5
    for span in result['word_spans']:
        word=pages[span['page_number']-1]['words'][span['word_index']]
        assert result['text'][span['start']:span['end']]==word['text']
        assert span['bbox']==word['bbox']
    for span in result['page_spans']+result['group_spans']:
        assert span['sha256']==hashlib.sha256(result['text'][span['start']:span['end']].encode()).hexdigest()
    assert result['evidence_rebinding_required'] and not result['source_quality_or_rights_cleared']
    assert result==render_reviewed_words(pages,layout,source_sha256='a'*64)


@pytest.mark.parametrize('indices',[[0],[0,2,2],[0,2,9],[0,True],[]])
def test_omitted_duplicate_invalid_and_empty_assignments_fail(indices):
    pages,layout=fixture();layout['pages'][0]['groups'][0]['word_indices']=indices
    with pytest.raises(ValueError):render_reviewed_words(pages,layout,source_sha256='a'*64)


@pytest.mark.parametrize('mutation',[
    lambda p,l:l.update(status='DRAFT'),lambda p,l:l.update(qa_inputs_used=True),
    lambda p,l:l.update(source_sha256='c'*64),lambda p,l:l.update(review_record_sha256=''),
    lambda p,l:p[0]['words'][0].update(text='modified'),
    lambda p,l:l['pages'].pop(),lambda p,l:l['pages'].reverse(),
    lambda p,l:l['pages'][0]['groups'][1].update(id='left'),
    lambda p,l:l['pages'][0]['groups'][0].update(answer='injected'),
    lambda p,l:l.update(answer='injected'),
    lambda p,l:l['pages'][0]['groups'][0].update(role=[]),
    lambda p,l:l['pages'][0].update(question='injected')])
def test_review_hash_page_and_unexpected_payload_fail_closed(mutation):
    pages,layout=fixture();mutation(pages,layout)
    with pytest.raises(ValueError):render_reviewed_words(pages,layout,source_sha256='a'*64)


@pytest.mark.parametrize('bbox',[[0,0,200,10],[20,1,1,10],[1,float('nan'),20,10]])
def test_bad_geometry_fails_even_if_inventory_hash_rebound(bbox):
    pages,layout=fixture();pages[0]['words'][0]['bbox']=bbox
    if any(isinstance(x,float) and x!=x for x in bbox):
        with pytest.raises(ValueError):fingerprint(pages)
    else:
        layout['word_pages_sha256']=fingerprint(pages)
        with pytest.raises(ValueError):render_reviewed_words(pages,layout,source_sha256='a'*64)
