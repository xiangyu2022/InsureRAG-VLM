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


def corpus_fixture():
    from copy import deepcopy
    from src.insurerag_vlm.holdout_corpus import source_snapshot_digest
    first, included = fixture()
    second = dict(first, id='distractor', text='Separate source material.')
    second['normalized_text_sha256'] = digest(second['text'])
    disposition = deepcopy(included)
    disposition.update(document_id='distractor', text_sha256=second['normalized_text_sha256'],
                       ranges=[], atomic_ranges=[], review=dict(decision='exclude',
                       record_sha256='c'*64, scope_rationale='Synthetic rights restriction'))
    docs = [first, second]
    manifest = dict(version=1, selection_basis='source_only',
                    source_snapshot_sha256=source_snapshot_digest(docs),
                    entries=[included, disposition])
    return docs, manifest


def test_full_corpus_requires_review_of_nonanswer_sources_before_tokenization():
    from copy import deepcopy
    from src.insurerag_vlm.holdout_corpus import build_corpus
    docs, manifest = corpus_fixture()
    calls = []
    def forbidden(_):
        calls.append(True)
        raise AssertionError('Must validate all source dispositions first')
    for mode in ['missing', 'pending', 'duplicate', 'unreviewed_exclusion']:
        bad = deepcopy(manifest)
        if mode == 'missing': bad['entries'].pop()
        elif mode == 'pending': bad['entries'][1]['review']['decision'] = 'pending'
        elif mode == 'duplicate': bad['entries'].append(deepcopy(bad['entries'][1]))
        else: bad['entries'][1]['review'].pop('record_sha256')
        with pytest.raises(ValueError): build_corpus(docs, bad, forbidden)
    assert calls == []
    chunks = build_corpus(docs, manifest, count)
    assert {c['document_id'] for c in chunks} == {'source'}


def test_corpus_snapshot_binds_missing_changed_and_duplicate_sources():
    from src.insurerag_vlm.holdout_corpus import build_corpus, source_snapshot_digest
    docs, manifest = corpus_fixture()
    assert source_snapshot_digest(docs) == source_snapshot_digest(list(reversed(docs)))
    for changed in [docs[:1], docs + docs[:1],
                    [docs[0], dict(docs[1], source_sha256='d'*64)],
                    [docs[0], dict(docs[1], jurisdiction='different')],
                    [docs[0], dict(docs[1], text='Changed text')]]:
        with pytest.raises(ValueError): build_corpus(changed, manifest, count)


def test_reviewed_distractor_is_searchable_without_gold_and_order_is_stable():
    from src.insurerag_vlm.holdout_corpus import build_corpus
    docs, manifest = corpus_fixture()
    second = manifest['entries'][1]
    second['review']['decision'] = 'include'
    second['ranges'] = [[0, len(docs[1]['text'])]]
    chunks = build_corpus(docs, manifest, count)
    assert {c['document_id'] for c in chunks} == {'source', 'distractor'}
    assert chunks == build_corpus(list(reversed(docs)), manifest, count)
    assert ''.join(c['text'] for c in chunks if c['document_id']=='distractor') == docs[1]['text']
    second['atomic_ranges_reviewed'] = False
    with pytest.raises(ValueError): build_corpus(docs, manifest, count)


def test_exclusion_does_not_hide_ranges_or_allow_an_empty_corpus():
    from src.insurerag_vlm.holdout_corpus import build_corpus
    docs, manifest = corpus_fixture()
    manifest['entries'][1]['ranges'] = [[0, 5]]
    with pytest.raises(ValueError): build_corpus(docs, manifest, count)
    manifest['entries'][1]['ranges'] = []
    manifest['entries'][0]['review']['decision'] = 'exclude'
    manifest['entries'][0]['ranges'] = []
    with pytest.raises(ValueError, match='every source'): build_corpus(docs, manifest, count)


@pytest.mark.parametrize('tail', ['', '  '])
def test_atomic_definition_edges_with_leading_and_trailing_space_are_valid_cuts(tail):
    d, m = fixture()
    text = 'Intro. Term detail.' + tail
    d.update(text=text, normalized_text_sha256=digest(text))
    m.update(text_sha256=digest(text), ranges=[[0,len(text)]], atomic_ranges=[[6,19]])
    chunks = chunk_document(d, m, count, max_tokens=5, max_chars=15)
    assert [c['text'] for c in chunks] == ['Intro.', ' Term detail.' + tail]
    assert ''.join(c['text'] for c in chunks) == text
    assert all(c['text'].strip() and len(c['text']) <= 15 for c in chunks)
    evidence = dict(document_id=d['id'], start=6, end=19, sha256=digest(text[6:19]))
    assert map_support_span(evidence, d, chunks) == [chunks[1]['id']]


def contextual_table_fixture():
    d, m = fixture()
    text = 'Guide. Term Plan Limit. RowA Gold cap 10. RowB Silver cap 20. RowC Bronze cap 30. End.'
    d.update(text=text, normalized_text_sha256=digest(text))
    header = [text.index('Term'), text.index(' RowA')]
    rows = [[text.index(label), text.index('.', text.index(label)) + 1]
            for label in ['RowA', 'RowB', 'RowC']]
    m.update(text_sha256=digest(text), ranges=[[0,len(text)]],
             atomic_ranges=[header] + rows, source_context_reviewed=True,
             source_context_groups=[dict(member_ranges=rows, context_ranges=[header])])
    return d, m, header, rows


def test_complete_rows_keep_source_headers_within_the_actual_encoder_budget():
    d, m, header, rows = contextual_table_fixture()
    chunks = chunk_document(d, m, count, max_tokens=12, max_chars=70)
    assert ''.join(c['text'] for c in chunks) == d['text']
    assert len(chunks) > 1
    assert all(c['tokens'] == count(c['retrieval_text']) <= 12
               and len(c['retrieval_text']) <= 70 for c in chunks)
    for a, b in rows:
        containing = [c for c in chunks if c['start'] <= a < b <= c['end']]
        assert len(containing) == 1
        c = containing[0]
        assert d['text'][header[0]:header[1]] in c['retrieval_text']
        evidence = dict(document_id=d['id'], start=a, end=b, sha256=digest(d['text'][a:b]))
        assert map_support_span(evidence, d, chunks) == [c['id']]
    # Header copies are context, not additional primary offsets or automatic
    # semantic support-alternative clearance.
    a, b = header
    evidence = dict(document_id=d['id'], start=a, end=b, sha256=digest(d['text'][a:b]))
    assert len(map_support_span(evidence, d, chunks)) == 1
    assert any(c['context_spans'] for c in chunks)


def test_row_that_fits_alone_cannot_overflow_when_its_header_is_added():
    from copy import deepcopy
    d, m, _, _ = contextual_table_fixture()
    bare = deepcopy(m)
    bare.pop('source_context_groups')
    bare.pop('source_context_reviewed')
    assert chunk_document(d, bare, count, max_tokens=8, max_chars=70)
    with pytest.raises(ValueError, match='Atomic'):
        chunk_document(d, m, count, max_tokens=8, max_chars=70)
    with pytest.raises(ValueError, match='Atomic'):
        chunk_document(d, m, count, max_tokens=50, max_chars=25)


@pytest.mark.parametrize('mode', ['unreviewed', 'not_atomic', 'empty', 'overlap'])
def test_header_context_needs_explicit_protected_source_relationships(mode):
    d, m, header, rows = contextual_table_fixture()
    if mode == 'unreviewed': m['source_context_reviewed'] = False
    elif mode == 'not_atomic': m['source_context_groups'][0]['context_ranges'] = [[header[0]+1,header[1]]]
    elif mode == 'empty': m['source_context_groups'][0]['member_ranges'] = []
    else: m['source_context_groups'][0]['context_ranges'] = [rows[0]]
    with pytest.raises(ValueError, match='context'):
        chunk_document(d, m, count)


@pytest.mark.parametrize('mode', ['changed_copy', 'missing_copy', 'changed_rendering', 'duplicate_copy'])
def test_copied_header_and_rendered_input_provenance_are_checked(mode):
    d, m, _, rows = contextual_table_fixture()
    chunks = chunk_document(d, m, count, max_tokens=12, max_chars=70)
    c = next(c for c in chunks if c['context_spans'])
    if mode == 'changed_copy': c['context_spans'][0]['text'] += ' injected'
    elif mode == 'missing_copy': c.pop('context_spans')
    elif mode == 'changed_rendering': c['retrieval_text'] = c['text']
    else: c['context_spans'].append(dict(c['context_spans'][0]))
    a, b = rows[0]
    evidence = dict(document_id=d['id'], start=a, end=b, sha256=digest(d['text'][a:b]))
    with pytest.raises(ValueError): map_support_span(evidence, d, chunks)


def test_unreviewed_table_context_blocks_whole_corpus_before_tokenization():
    from src.insurerag_vlm.holdout_corpus import build_corpus, source_snapshot_digest
    d, entry, _, _ = contextual_table_fixture()
    docs, manifest = corpus_fixture()
    docs[0], manifest['entries'][0] = d, entry
    manifest['source_snapshot_sha256'] = source_snapshot_digest(docs)
    entry['source_context_reviewed'] = False
    calls = []
    def forbidden(_):
        calls.append(True)
        raise AssertionError('Review gate must precede tokenizer')
    with pytest.raises(ValueError, match='context'): build_corpus(docs, manifest, forbidden)
    assert calls == []
    entry['source_context_reviewed'] = True
    assert build_corpus(docs, manifest, count, max_tokens=12, max_chars=70)
