"""Small synthetic pages. Mock visual extraction is not recognition-quality evidence."""
import copy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
import pytest

from src.insurerag_vlm.page_ingest import (ingest_pages, observed, sentence_units, context_prefix,
                                         native_extraction, build_page_records, PARSER)
from src.insurerag_vlm.page_schema import PageExtraction, validate_page_record
from src.insurerag_vlm.chroma_knowledge import ClauseStore, normalize_record
from src.insurerag_vlm.agent_mcp import MCPClauseRetriever
from src.insurerag_vlm.agent_graph import ReviewAgent, validate_claims
from src.insurerag_vlm.agent_mcp_cli import EvidenceQuoteController
from src.insurerag_vlm.agent_draft import InvalidAction, Limits


def bbox(x0=0.1, y0=0.1, x1=0.9, y1=0.2):
    return dict(x0=x0, y0=y0, x1=x1, y1=y1)


def page_fixture(path):
    raster = Image.new('RGB', (500, 300), 'white')
    ImageDraw.Draw(raster).text((30, 60), 'FICTIONAL SCAN IMAGE - NOT A REAL POLICY', fill='black')
    buf = BytesIO()
    raster.save(buf, format='PNG')
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 25), 'FICTIONAL HEADER')
        page.insert_text((50, 80), 'Section 1 Coverage')
        page.insert_text((50, 120), 'Fictional source paragraph. A complete sentence stays on this page.')
        page.insert_text((50, 820), '1')
        page = pdf.new_page()
        page.insert_image(page.rect, stream=buf.getvalue())
        page = pdf.new_page()
        page.draw_line((50, 400), (400, 400))
        page.draw_line((50, 400), (50, 100))
        page.insert_text((60, 80), 'Fictional chart with an existing text layer')
        page = pdf.new_page()
        page.insert_text((50, 80), 'Mixed text must not suppress the visual image.')
        page.insert_image(fitz.Rect(50, 150, 400, 350), stream=buf.getvalue())
        pdf.save(path)


class MockVision:
    version = 'MOCK-layout-fixture-v1-not-a-model'
    def __init__(self):
        self.calls = []

    def recognize(self, path):
        self.calls.append(path.name)
        cells = []
        for row, values in enumerate([['Benefits', 'Values'], ['Item', 'Units'], ['Alpha', '10'], ['Beta', '20'], ['Gamma', '30']]):
            for column, value in enumerate(values):
                cells.append(dict(row=row, column=column, row_span=1, column_span=1,
                    text=observed(value), bbox=bbox(.1 + .4 * column, .3 + .04 * row, .5 + .4 * column, .34 + .04 * row)))
        table = dict(bbox=bbox(.1, .3, .9, .55), number=observed('Table 1'), caption=observed('Fictional benefits'),
            header_rows=observed([0, 1]), cells=cells, units=observed(['illustrative units']),
            footnotes=observed(['Fictional example only']), continued=observed(False))
        figure = dict(kind='chart', bbox=bbox(.1, .6, .9, .9), caption=observed('Figure 1: fictional trend'),
            summary=observed('Generated fixture summary: the illustrated line trends upward; exact values are unclear.'),
            axes=observed(['horizontal: time', 'vertical: illustrative units']), legend=observed(None, status='not_found'),
            units=observed(['illustrative units']), numeric_uncertainty=observed(['Exact chart values are unreadable; no numeric estimates.']))
        segments = [dict(kind='header', text='Fictional page header', bbox=bbox(.1, .01, .9, .04), heading_level=observed(None, status='not_applicable')),
                    dict(kind='heading', text='Section 2 Fictional benefits', bbox=bbox(.1, .08, .9, .12), heading_level=observed(1)),
                    dict(kind='paragraph', text='Fictional paragraph: consult the original page for all conditions.', bbox=bbox(.1, .15, .9, .22), heading_level=observed(None, status='not_applicable')),
                    dict(kind='footer', text='Fictional footer - iii', bbox=bbox(.1, .95, .9, .98), heading_level=observed(None, status='not_applicable'))]
        return PageExtraction.model_validate(dict(readability='readable', printed_page=observed('iii'),
            segments=segments, tables=observed([table]), figures=observed([figure])))


def test_page_routing_and_no_silent_ocr(tmp_path):
    source = tmp_path / 'fixture.pdf'
    page_fixture(source)
    original = source.read_bytes()
    records, report = ingest_pages(source, tmp_path / 'pending', doc_id='fictional')
    assert [p['route'] for p in report['pages']] == ['native_text'] + ['compressed_multimodal'] * 3
    assert report['status'] == 'pending' and report['ocr_fallback'] is False
    assert all('multimodal_unavailable' in p['reason'] for p in report['pages'][1:])
    assert {r['page_number'] for r in records} == {1}
    assert source.read_bytes() == original
    assert (tmp_path / 'pending/originals' / (sha256(original).hexdigest() + '.pdf')).read_bytes() == original
    for p in report['pages'][1:]:
        assert max(p['image']['width'], p['image']['height']) <= 2000
        assert p['image']['quality'] == 85
    mock = MockVision()
    records, report = ingest_pages(source, tmp_path / 'complete', doc_id='fictional', recognizer=mock, maximum=100)
    assert report['status'] == 'complete_unreviewed' and len(mock.calls) == 3
    assert {r['page_number'] for r in records} == {1, 2, 3, 4}
    for r in records:
        validate_page_record(r)
        assert all(loc['physical_page'] == r['page_number'] for loc in r['chunk_metadata']['locations'])
        assert r['source_sha256'] == sha256(original).hexdigest()
        assert r['retrieval_text'] == r['context_prefix'] + '\n\n' + r['text']
    native = next(r for r in records if r['page_number'] == 1)
    assert native['page_metadata']['headers']['status'] == 'observed'
    assert native['page_metadata']['footers']['status'] == 'observed'
    assert native['page_metadata']['printed_page']['value'] == '1'


def test_semantic_boundaries_inherited_sections_and_no_character_chopping(tmp_path):
    source = tmp_path / 'source.md'
    source.write_text('# Verified source heading\n\n' + ('A whole sentence stays intact. ' * 15) + '\f' +
                      'Continued paragraph in the next logical page.\n\n' + ('x' * 301), encoding='utf-8')
    records, report = ingest_pages(source, tmp_path / 'out', doc_id='logical', maximum=100)
    assert report['status'] == 'complete_unreviewed'
    assert {r['page_number'] for r in records} == {1, 2}
    sentence_chunks = [r for r in records if r['page_number'] == 1 and r['content_type'] == 'paragraph']
    assert len(sentence_chunks) > 1
    assert all(r['text'].rstrip().endswith('.') for r in sentence_chunks)
    second = next(r for r in records if r['page_number'] == 2)
    context = second['page_metadata']['sections']['value'][0]
    assert context['inherited'] and context['source_page_id'] == 'logical::p000001'
    assert 'inherited from logical::p000001' in second['context_prefix']
    oversized = next(r for r in records if r['text'] == 'x' * 301)
    assert oversized['chunk_metadata']['oversized_unsplit']
    assert all(r['page_metadata']['physical_page']['status'] == 'not_applicable' for r in records)
    assert 'unknown; not stated' in second['context_prefix']


def test_table_isolation_header_inheritance_and_generated_summary(tmp_path):
    source = tmp_path / 'fixture.pdf'
    page_fixture(source)
    records, _ = ingest_pages(source, tmp_path / 'out', doc_id='fictional', recognizer=MockVision(), maximum=100)
    tables = [r for r in records if r['content_type'] == 'table_rows']
    assert tables
    for r in tables:
        meta = r['chunk_metadata']
        assert meta['table']['header_rows']['value'] == [0, 1]
        assert 'Benefits | Values' in r['text'] and 'Item | Units' in r['text']
        assert meta['table']['table_id'] == meta['table_id']
        assert 'Fictional paragraph' not in r['text']
    assert len({r['chunk_metadata']['cross_page_table_group']['value'] for r in tables}) == 1
    figure = next(r for r in records if r['content_type'] == 'figure_summary')
    assert figure['source_text'] is None and figure['summary_text'] == figure['text']
    assert figure['chunk_metadata']['figure']['numeric_uncertainty']['value']
    evidence = {'x': {'text': figure['text'], 'reference': figure}}
    with pytest.raises(InvalidAction, match='retrieval aids'):
        validate_claims([{'text': 'A claimed original quote', 'citations': [{'id': 'x', 'quote': figure['text'][:30]}]}], evidence)


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop('source_sha256'), lambda r: r.pop('page_metadata'),
    lambda r: r.update(page_number=999), lambda r: r['chunk_metadata']['locations'][0].update(physical_page=999),
    lambda r: r.update(context_prefix='Invented chapter 99'), lambda r: r.update(retrieval_text='stale index input'),
    lambda r: r.update(text=r['text'] + 'new content'),
])
def test_missing_metadata_or_stale_prefix_fails(tmp_path, mutation):
    source = tmp_path / 'a.txt'
    source.write_text('A source paragraph with an explicit source identity.', encoding='utf-8')
    records, _ = ingest_pages(source, tmp_path / 'out', doc_id='a')
    mutation(records[0])
    with pytest.raises((ValueError, KeyError)):
        normalize_record(records[0])


def test_multimodal_failure_never_falls_back_to_text_layer(tmp_path):
    source = tmp_path / 'fixture.pdf'
    page_fixture(source)
    class Broken:
        version = 'mock-failed'
        def recognize(self, path):
            raise RuntimeError('recognition unavailable')
    records, report = ingest_pages(source, tmp_path / 'out', doc_id='fixture', recognizer=Broken())
    assert report['status'] == 'pending'
    assert {r['page_number'] for r in records} == {1}
    assert all(p['status'] == 'pending' for p in report['pages'][1:])


def test_chroma_mcp_agent_nested_metadata_and_context_quote_separation(tmp_path):
    source = tmp_path / 'source.md'
    source.write_text('# Fictional heading\n\nOriginal evidence: request a policy copy from the insurer.', encoding='utf-8')
    records, _ = ingest_pages(source, tmp_path / 'out', doc_id='fixture')
    db = tmp_path / 'chroma'
    with ClauseStore(db, writable=True, record_schema=2) as store:
        assert store.upsert(records)['written'] == len(records)
        with pytest.raises(ValueError, match='schema'):
            store.upsert([dict(record_id='legacy', doc_id='legacy', source_sha256='a' * 64, page=1, text='Legacy text', source='source')])
        source.write_text('# Updated explicitly supplied heading\n\nOriginal evidence: request a policy copy from the insurer.', encoding='utf-8')
        updated, _ = ingest_pages(source, tmp_path / 'updated', doc_id='fixture')
        assert [r['record_id'] for r in updated] == [r['record_id'] for r in records]
        assert store.upsert(updated)['written'] == len(updated)
        records = updated
    with pytest.raises(ValueError, match='contract mismatch'):
        with ClauseStore(db):
            pass
    with MCPClauseRetriever(db, 'insurance_clauses', record_schema=2) as retriever:
        found = retriever.search('policy copy insurer', 1)[0]
        restored = retriever.read_evidence(found)
        original = next(r for r in records if r['record_id'] == restored['reference']['record_id'])
        assert restored['reference']['page_metadata'] == original['page_metadata']
        assert restored['reference']['chunk_metadata'] == original['chunk_metadata']
        assert restored['reference']['context_prefix'] == original['context_prefix']
        agent = ReviewAgent(tmp_path / 'review.sqlite', retriever, EvidenceQuoteController(), runtime_id='v2-fixture',
                            limits=Limits(top_k=1, context_tokens=16384), review_abstentions=True)
        state = agent.start('v2', 'policy copy insurer')
        assert state['status'] == 'awaiting_review' and state['released_response'] is None
        row = next(iter(state['evidence'].values()))
        assert row['reference']['page_metadata'] == original['page_metadata']
        eid = next(iter(state['evidence']))
        with pytest.raises(InvalidAction):
            validate_claims([{'text': 'Invented context quote', 'citations': [{'id': eid, 'quote': original['context_prefix']}]}], state['evidence'])


def test_full_table_figure_metadata_roundtrip_and_legacy_reference(tmp_path):
    source = tmp_path / 'fixture.pdf'
    page_fixture(source)
    records, _ = ingest_pages(source, tmp_path / 'out', doc_id='fictional', recognizer=MockVision())
    db = tmp_path / 'v2'
    selected = [next(r for r in records if r['content_type'] == kind) for kind in ('table_rows', 'figure_summary')]
    with ClauseStore(db, writable=True, record_schema=2) as store:
        store.upsert(selected)
    with MCPClauseRetriever(db, 'insurance_clauses', record_schema=2) as retriever:
        found = retriever.search('Fictional benefits chart', 2)
        assert {r['reference']['content_type'] for r in found} == {'table_rows', 'figure_summary'}
        for row in found:
            result = retriever.read_evidence(row)
            original = next(r for r in selected if r['record_id'] == result['reference']['record_id'])
            assert result['reference']['page_metadata'] == original['page_metadata']
            assert result['reference']['chunk_metadata'] == original['chunk_metadata']
    legacy = tmp_path / 'legacy'
    with ClauseStore(legacy, writable=True) as store:
        store.upsert([{'record_id': 'old', 'doc_id': 'old', 'source_sha256': 'a' * 64,
                       'page_number': 1, 'text': 'Old quoted evidence.', 'source': 'old page 1'}])
    with MCPClauseRetriever(legacy, 'insurance_clauses') as retriever:
        row = retriever.search('quoted evidence', 1)[0]
        row['reference'] = {k: row['reference'][k] for k in ('evidence_id', 'record_sha256', 'source_sha256', 'text_sha256', 'doc_id', 'page_id', 'page_number', 'citation')}
        assert retriever.read_evidence(row)['text'] == 'Old quoted evidence.'


def test_new_upload_preserves_pending_original_and_uses_v2(tmp_path):
    from unittest.mock import patch
    import src.insurerag_vlm.app as app
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentTextUnavailableError
    class Handler(app.DemoHandler):
        _pipeline = None
        _data_folder = tmp_path / 'unused'
        _index_dir = tmp_path / 'index'
    def config(**kwargs):
        return ModelConfig(**kwargs, vlm_model='local-extractive', retrieval_model='local-hashing', enable_image_signal=False)
    mixed = tmp_path / 'mixed.pdf'
    page_fixture(mixed)
    with patch.object(app, 'UPLOAD_DIR', tmp_path / 'uploads/current'), patch.object(app, 'ModelConfig', side_effect=config):
        with pytest.raises(DocumentTextUnavailableError, match='Strict page ingestion pending'):
            Handler.activate_upload('mixed.pdf', mixed.read_bytes())
        saved = list((tmp_path / 'uploads/pending-page-v2').rglob(sha256(mixed.read_bytes()).hexdigest() + '.pdf'))
        assert len(saved) == 1 and saved[0].read_bytes() == mixed.read_bytes()
        assert Handler._pipeline is None
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 100), 'Fictional policy paragraph with a source citation.')
            Handler.activate_upload('native.pdf', pdf.tobytes())
        records = Handler._pipeline._ensure_indices(Handler._data_folder)['snippet_meta']
        assert all(r['schema_version'] == 2 and r['context_prefix'] for r in records)


def test_unavailable_figure_summary_stays_pending(tmp_path):
    source = tmp_path / 'fixture.pdf'
    page_fixture(source)
    class UnreadableSummary(MockVision):
        def recognize(self, path):
            value = super().recognize(path).model_dump(mode='json')
            value['figures']['value'][0]['summary'] = observed(None, status='unreadable')
            return PageExtraction.model_validate(value)
    records, report = ingest_pages(source, tmp_path / 'out', doc_id='fixture', recognizer=UnreadableSummary())
    assert report['status'] == 'pending'
    assert {r['page_number'] for r in records} == {1}
    assert all('Figure summary unavailable' in p['reason'] for p in report['pages'][1:])


def test_agent_cli_unicode_json_on_windows_encoding(tmp_path):
    import os
    import subprocess
    import sys
    db = tmp_path / 'db'
    text = 'Fictional Unicode evidence: ' + chr(0x21a9) + ' return for review.'
    with ClauseStore(db, writable=True) as store:
        store.upsert([dict(record_id='unicode', doc_id='unicode', source_sha256='a' * 64,
                           page=1, text=text, source='fictional source')])
    env = dict(os.environ, PYTHONIOENCODING='gbk')
    args = [sys.executable, '-m', 'src.insurerag_vlm.agent_mcp_cli', 'start', '--thread-id', 'unicode',
            '--question', 'return review', '--db', str(tmp_path / 'review.sqlite'), '--store', str(db),
            '--collection', 'insurance_clauses']
    completed = subprocess.run(args, capture_output=True, env=env, timeout=60)
    assert completed.returncode == 0, completed.stderr.decode('ascii', errors='replace')
    result = json.loads(completed.stdout.decode('ascii'))
    assert result['result']['status'] == 'awaiting_review'
    assert result['result']['released_response'] is None
    assert next(iter(result['result']['evidence'].values()))['text'] == text
