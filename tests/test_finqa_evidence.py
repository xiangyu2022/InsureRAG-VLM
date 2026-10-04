from copy import deepcopy
import pytest
from src.insurerag_vlm.finqa_evidence import table_row_text,page_evidence


def test_table_header_and_values_are_preserved_for_every_row():
    assert table_row_text(['USD millions','2019','2020'],['claims','12','15'])=='USD millions the claims of 2019 is 12 ; the claims of 2020 is 15 ;'
    with pytest.raises(ValueError): table_row_text(['','2019'],['claims'])


def test_evidence_serialization_is_independent_of_gold_and_question():
    r={'filename':'X/2020/page_1.pdf','pre_text':['before'],'post_text':['after'],
       'table':[['','2019','2020'],['claims','12','15']], 'qa':{'question':'secret','gold_inds':{'text_0':'LEAK'}},
       'table_retrieved':[{'score':9999}]}
    altered=deepcopy(r);altered['qa']={'question':'different','gold_inds':{'table_1':'different'}};altered['table_retrieved']=[]
    assert page_evidence(r)==page_evidence(altered)
    assert [e['upstream_evidence_key'] for e in page_evidence(r)]==['text_0','text_1','table_0','table_1']
    assert all('LEAK' not in e['text'] and 'secret' not in e['text'] for e in page_evidence(r))


def test_same_row_number_in_distinct_pages_does_not_collide():
    r={'filename':'X/2020/page_1.pdf','pre_text':['same'],'post_text':[],'table':[['','a'],['b','c']]}
    other={**r,'filename':'X/2020/page_2.pdf'}
    assert {e['id'] for e in page_evidence(r)}.isdisjoint({e['id'] for e in page_evidence(other)})
    assert {e['source_group'] for e in page_evidence(r)}=={'X/2020'}
