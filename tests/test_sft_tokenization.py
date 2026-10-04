"""Regression for prompt-only truncation silently becoming an SFT target."""
import pytest
from src.insurerag_vlm.sft import tokenize_sft_record


class CharacterTokenizer:
    chat_template = None
    eos_token = None

    def __call__(self, text, *, truncation=False, max_length=None, **kwargs):
        ids = list(text.encode('utf-8'))
        return {'input_ids': ids[:max_length] if truncation else ids}


def test_truncated_prompt_is_never_used_as_an_assistant_label():
    with pytest.raises(ValueError, match='no assistant tokens'):
        tokenize_sft_record(CharacterTokenizer(),
            {'record_id':'too-long','question':'What limit?', 'evidence':'e'*500, 'answer':'$500'},64)


def test_supervised_labels_only_cover_assistant_answer():
    result=tokenize_sft_record(CharacterTokenizer(),{'question':'What limit?','answer':'$500'},2048)
    target=bytes(x for x in result['labels'] if x!=-100).decode('utf-8')
    assert target==' $500'
    assert len(result['input_ids'])==len(result['attention_mask'])==len(result['labels'])


def test_invalid_context_length_rejected():
    with pytest.raises(ValueError,match='max_length'):
        tokenize_sft_record(CharacterTokenizer(),{'question':'Q','answer':'A'},0)
