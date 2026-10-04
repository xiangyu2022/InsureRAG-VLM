from scripts.summarize_source_generation import clean_answer,numbers
from src.insurerag_vlm.evidence_evaluation import token_f1

def test_inline_citation_ids_do_not_become_amounts_or_lexical_content():
    answer='The waiting period is six months. SOURCE: sourcefaq:10ab605fa9abda6b194a'
    content=clean_answer(answer)
    assert numbers(content)==set()
    assert token_f1(content,'The waiting period is six months.')==1

def test_numbers_do_not_parse_hex_ids_and_preserve_sign_and_percent():
    assert numbers('10ab605fa9 -12.5 10% 1,000')=={'-12.5','10%','1000'}
