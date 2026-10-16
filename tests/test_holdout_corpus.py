import pytest
from src.insurerag_vlm.holdout_corpus import chunk_document,map_support_span,digest


def fixture():
    text='Limit 100. Exception applies. Header Amount RowA 20 RowB 30. End.'
    d=dict(id='source',source_sha256='a'*64,text=text,normalized_text_sha256=digest(text))
    m=dict(document_id=d['id'],source_sha256=d['source_sha256'],text_sha256=digest(text),
           selection_basis='source_only',ranges=[[0,len(text)]],atomic_ranges=[],atomic_ranges_reviewed=True,
           review=dict(decision='include',record_sha256='b'*64,scope_rationale='Synthetic full source'))
    return d,m


def count(text):return len(text.split())+2


def test_complete_source_is_partitioned_without_gold_or_dropped_characters():
    d,m=fixture();cs=chunk_document(d,m,count,max_tokens=6,max_chars=30)
    assert ''.join(c['text'] for c in cs)==d['text']
    assert len({c['id'] for c in cs})==len(cs)>1
    assert all(c['tokens']<=6 and len(c['text'])<=30 for c in cs)
    assert cs==chunk_document(d,m,count,max_tokens=6,max_chars=30)
    assert all('answer' not in c and 'question' not in c for c in cs)


def test_split_support_requires_every_chunk_and_missing_ranges_fail():
    d,m=fixture();cs=chunk_document(d,m,count,max_tokens=6,max_chars=30)
    a,b=5,len(d['text'])-3;e=dict(document_id=d['id'],start=a,end=b,sha256=digest(d['text'][a:b]))
    assert map_support_span(e,d,cs)==[c['id'] for c in cs]
    with pytest.raises(ValueError,match='uncovered'):map_support_span(e,d,cs[1:])
    with pytest.raises(ValueError,match='Duplicate'):map_support_span(e,d,cs+cs[:1])
    with pytest.raises(ValueError,match='hash'):map_support_span(e|{'sha256':'0'*64},d,cs)
    with pytest.raises(ValueError,match='identity'):map_support_span(e,d,[cs[0]|{'id':'fabricated'}]+cs[1:])


def test_atomic_table_headers_rows_and_qualifiers_are_not_split_or_truncated():
    d,m=fixture();a=d['text'].index('Header');b=d['text'].index(' End.');m['atomic_ranges']=[[a,b]]
    cs=chunk_document(d,m,count,max_tokens=9,max_chars=40)
    assert sum(c['start']<=a and c['end']>=b for c in cs)==1
    with pytest.raises(ValueError,match='Atomic'):chunk_document(d,m,count,max_tokens=5,max_chars=40)
    m['ranges']=[[a+1,len(d['text'])]]
    with pytest.raises(ValueError,match='atomic'):chunk_document(d,m,count)


@pytest.mark.parametrize('change',[
    {'selection_basis':'gold_evidence'}, {'text_sha256':'0'*64},
    {'atomic_ranges_reviewed':False}, {'review':{'decision':'pending'}},
    {'ranges':[[0,10],[9,20]]}, {'ranges':[]},
])
def test_unreviewed_or_changed_or_malformed_source_scope_cannot_build(change):
    d,m=fixture()
    with pytest.raises(ValueError):chunk_document(d,m|change,count)


def test_noncontiguous_permission_scope_is_preserved_and_gap_cannot_map():
    d,m=fixture();m['ranges']=[[0,10],[29,len(d['text'])]]
    cs=chunk_document(d,m,count)
    assert ''.join(c['text'] for c in cs)==d['text'][:10]+d['text'][29:]
    e=dict(document_id=d['id'],start=0,end=35,sha256=digest(d['text'][:35]))
    with pytest.raises(ValueError,match='uncovered'):map_support_span(e,d,cs)
    with pytest.raises(ValueError,match='Token counter'):chunk_document(d,m,lambda _:0)
