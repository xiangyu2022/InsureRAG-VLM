from pathlib import Path
import json

import pytest

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.graph import build_document_graph, build_graph_adjacency, expand_candidate_page_keys, build_explicit_reference_edges
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


def page(key, text, **kwargs):
    return dict(page_key=key, doc_id='d', packet_id='p', text=text, **kwargs)


def expand(pages, seeds={'a'}, **kwargs):
    edges = build_document_graph(pages, [])
    return expand_candidate_page_keys(seeds, build_graph_adjacency(edges), True, True, True, True, **kwargs)


def test_two_hop_literal_trace_cycle_and_reproducibility():
    pages = [page('a', 'Section A: Coverage. See Section B.'),
             page('b', 'Section B: Conditions. Refer to Section C.'),
             page('c', 'Section C: Exception. See Section A.')]
    forward = expand(pages, explicit_only=True)
    backward = expand(list(reversed(pages)), explicit_only=True)
    assert forward == backward
    assert {x['page_key'] for x in forward} == {'b', 'c'}
    for row in forward:
        assert not row['supports_precedence']
        for step in row['path']:
            source = next(p for p in pages if p['page_key'] == step['evidence_page_key'])
            assert step['evidence_span'] in source['text']
    chain = expand(pages[:2] + [page('c', 'Section C: Exception.')], explicit_only=True)
    assert next(x for x in chain if x['page_key'] == 'c')['hop'] == 2
    assert len(expand(pages, max_expansions=1)) == 1
    assert expand(pages, max_hops=0) == []


def test_ambiguous_sections_wrong_policy_and_out_of_packet_do_not_link():
    source = page('a', 'Section A: See Section B.', policy_number='P1')
    target = page('b', 'Section B: Text.', policy_number='P1')
    assert len(build_document_graph([source, target], [])) == 1
    assert build_document_graph([source, target, {**target, 'page_key': 'b2'}], []) == []
    assert build_document_graph([source, {**target, 'policy_number': 'P2'}], []) == []
    assert build_document_graph([source, {**target, 'packet_id': 'other'}], []) == []


def test_printed_page_reference_is_not_a_physical_page_link():
    assert build_document_graph([page('a', 'See page 2 for the exception.'), page('b', 'Exclusion.')], []) == []


def test_hierarchical_section_reference_cannot_fall_back_to_parent():
    source = page('a', 'Coverage. See Section 5(a). Refer to Section 5(b)(2).')
    parent = page('b', 'Section 5. All benefits.')
    assert not build_explicit_reference_edges([source, parent])
    child = page('c', 'Section 5(a). Medical benefits.')
    nested = page('d', 'Section 5(b)(2). Conditions.')
    edges = build_explicit_reference_edges([source, parent, child, nested])
    assert {e['target_page_key'] for e in edges} == {'c', 'd'}
    assert {e['reference_id'] for e in edges} == {'5(a)', '5(b)(2)'}


def test_unsupported_section_suffixes_do_not_silently_truncate():
    parent = page('b', 'Section 5. Parent.')
    for text in ['See Section 5.1.', 'See Section 5 (a).', 'See Section 5(a)-5(b).']:
        assert not build_explicit_reference_edges([page('a', text), parent])
    assert not build_explicit_reference_edges([
        page('a', 'See Section 5.'), page('b', 'Section 5.1. A dotted subsection.')])


def test_repeated_section_mentions_do_not_inflate_edge_count():
    edges = build_explicit_reference_edges([page('a', 'See Section 5(a). Again refer to Section 5(a).'),
                                           page('b', 'Section 5(a). Medical benefits.')])
    assert len(edges) == 1


def test_metadata_is_candidate_only_and_never_transitively_confirmed():
    pages = [page('a', 'Coverage.', clause_types=['coverage'], coverage_tags=['water']),
             page('b', 'Exclusion.', clause_types=['exclusion'], coverage_tags=['water']),
             page('c', 'Endorsement.', document_type='endorsement', coverage_tags=['water'])]
    edges = build_document_graph(pages, [])
    assert edges
    assert all(e['relation_status'] == 'candidate' and not e['supports_precedence'] for e in edges)
    assert all(e['relation'].startswith('candidate_') for e in edges)
    assert expand(pages, explicit_only=True) == []
    assert all(e['hop'] == 1 for e in expand(pages))


def test_external_parent_id_maps_to_same_page_and_mode_off_is_real(tmp_path):
    record = {'record_id': 'external-page', 'doc_id': 'd', 'page': 1, 'citation': 'd.pdf#page=1',
              'text': 'Section A: Coverage. See Section B.', 'policy_number': 'P1'}
    (tmp_path/'rag_pages.jsonl').write_text(json.dumps(record)+'\n')
    (tmp_path/'rag_snippets.jsonl').write_text(json.dumps({**record, 'record_id': 's', 'parent_page_id': 'external-page'})+'\n')
    pipeline = DocumentRetrievalPipeline(ModelConfig(curated_dataset_dir=tmp_path, graph_mode='off'))
    corpus = pipeline._load_curated_corpus(tmp_path)
    assert corpus['pages'][0]['page_key'] == corpus['snippets'][0]['page_key']
    assert corpus['pages'][0]['policy_number'] == 'P1'
    assert pipeline._add_graph_expansion_candidates([], {}, None) == []


def test_invalid_graph_mode_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(graph_mode='guaranteed-correct')


def test_context_reserves_explicit_dependencies_not_high_scoring_distractors():
    pages = [page('a', 'Section A: Coverage. See Section B.', source='a'),
             page('b', 'Section B: Conditions. See Section C.', source='b'),
             page('c', 'Section C: Limit.', source='c'),
             page('x', 'A similar but unrelated example.', source='x')]
    edges = build_document_graph(pages, [])
    indices = {'page_meta': pages, 'graph_adjacency': build_graph_adjacency(edges)}
    ranking = [{'source': s} for s in ['x', 'a', 'c', 'b']]
    pipeline = DocumentRetrievalPipeline(ModelConfig(graph_mode='explicit'))
    chosen = pipeline.select_graph_evidence_bundle(ranking, indices, 3)
    assert [p['source'] for p in chosen] == ['a', 'b', 'c']
    assert chosen[-1]['context_graph_path'][-1]['target_page_key'] == 'c'
    pipeline.config.graph_mode = 'off'
    assert pipeline.select_graph_evidence_bundle(ranking, indices, 3) == ranking[:3]


def test_printed_page_mapping_requires_matching_quotes_label_and_packet():
    ref = dict(target_doc_id='d', target_physical_page=12, target_printed_page_label='10',
               evidence_span='See page 10 for lawsuit options', target_evidence_span='The Right to Sue')
    a = page('a', 'See page 10 for lawsuit options.', page_number=4, source_references=[ref])
    b = page('b', 'The Right to Sue. Available options.', page_number=12, printed_page_label='10')
    edges = build_document_graph([a, b], [])
    assert len(edges) == 1 and edges[0]['target_page_key'] == 'b'
    assert edges[0]['relation_status'] == 'explicit_reference'
    assert not edges[0]['supports_precedence']
    for changed in [{**b, 'printed_page_label': '12'}, {**b, 'text': 'Wrong heading'}, {**b, 'packet_id': 'other'}]:
        with pytest.raises(ValueError):
            build_document_graph([a, changed], [])
