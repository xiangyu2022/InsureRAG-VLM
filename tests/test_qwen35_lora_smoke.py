import json
from types import SimpleNamespace

import pytest

from scripts.smoke_qwen35_lora import (
    MODEL_ID, REVISION, SYNTHETIC_ROWS, expected_target_names, tokenize_rows, validate_snapshot,
    set_portable_adapter_identity,verify_saved_tokenizer,
)


class Tokenizer:
    chat_template = 'mock'

    def __init__(self,mismatch=False,long=False,empty=False):
        self.mismatch,self.long,self.empty = mismatch,long,empty

    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return 'prompt' if kwargs['add_generation_prompt'] else 'complete'

    def __call__(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':False}  # No silent truncation.
        if text=='prompt':
            return {'input_ids':[10,11,12]}
        prefix=[99,11,12] if self.mismatch else [10,11,12]
        return {'input_ids':prefix+([] if self.empty else [20]*300 if self.long else [20,21])}


def test_synthetic_training_rows_have_complete_assistant_only_supervision():
    examples,audit=tokenize_rows(Tokenizer())
    assert len(examples)==2
    assert all(row['split']=='train' and row['record_id'].startswith('synthetic_') for row in SYNTHETIC_ROWS)
    assert all(row['labels']==[-100,-100,-100,20,21] for row in examples)
    assert all(row['assistant_target_tokens']==2 and not row['truncated'] for row in audit)


@pytest.mark.parametrize('tokenizer,message',[
    (Tokenizer(mismatch=True),'exact prefix'),
    (Tokenizer(long=True),'without truncation'),
    (Tokenizer(empty=True),'without truncation'),
])
def test_tokenizer_mismatch_truncation_and_empty_supervision_fail_closed(tokenizer,message):
    with pytest.raises(ValueError,match=message):
        tokenize_rows(tokenizer)


def test_lora_scope_excludes_vision_head_and_arbitrary_suffix_matches():
    names=[
        'model.language_model.layers.0.linear_attn.in_proj_qkv',
        'model.language_model.layers.3.self_attn.q_proj',
        'model.language_model.layers.1.mlp.down_proj',
        'model.visual.blocks.0.attn.q_proj',
        'model.language_model.lm_head',
        'lm_head',
        'model.language_model.layers.0.linear_attn.in_proj_qkv_extra',
    ]
    assert expected_target_names(names)==sorted(names[:3])


def test_expected_small_architecture_has_186_selected_modules():
    names=[]
    for layer in range(24):
        prefix=f'model.language_model.layers.{layer}.'
        attention=('self_attn',['q_proj','k_proj','v_proj','o_proj']) if layer%4==3 else (
            'linear_attn',['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'])
        names.extend(prefix+attention[0]+'.'+name for name in attention[1])
        names.extend(prefix+'mlp.'+name for name in ['gate_proj','up_proj','down_proj'])
    assert len(expected_target_names(names))==186


def test_wrong_local_model_path_cannot_be_labeled_as_pinned_checkpoint(tmp_path):
    with pytest.raises(ValueError,match='pinned Hugging Face snapshot'):
        validate_snapshot(tmp_path)
    snapshot=tmp_path/'models--Qwen--Qwen3.5-0.8B'/'snapshots'/REVISION
    snapshot.mkdir(parents=True)
    with pytest.raises(ValueError,match='Incomplete local snapshot'):
        validate_snapshot(snapshot)


def test_snapshot_provenance_hashes_actual_local_files(tmp_path):
    snapshot=tmp_path/'models--Qwen--Qwen3.5-0.8B'/'snapshots'/REVISION
    snapshot.mkdir(parents=True)
    config={'model_type':'qwen3_5','architectures':['Qwen3_5ForConditionalGeneration'],
            'text_config':{'hidden_size':1024,'num_hidden_layers':24,'vocab_size':248320}}
    (snapshot/'config.json').write_text(json.dumps(config))
    for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        (snapshot/name).write_text('fixture')
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'key':'part.safetensors'}}))
    (snapshot/'part.safetensors').write_bytes(b'fixture-not-real-weights')
    _,_,first=validate_snapshot(snapshot)
    (snapshot/'part.safetensors').write_bytes(b'modified-fixture')
    _,_,second=validate_snapshot(snapshot)
    assert first['part.safetensors']['sha256']!=second['part.safetensors']['sha256']


def test_preflight_is_importable_without_gpu_libraries():
    # Module-level imports are intentionally stdlib only; the offline CI job
    # invokes these tests without Torch, Transformers, PEFT or Accelerate.
    import scripts.smoke_qwen35_lora as smoke
    assert 'torch' not in smoke.__dict__
    assert 'transformers' not in smoke.__dict__


def test_saved_adapter_identity_is_portable_and_revision_pinned():
    config=SimpleNamespace(base_model_name_or_path='C:/machine-specific/cache',revision=None)
    model=SimpleNamespace(peft_config={'default':config})
    set_portable_adapter_identity(model)
    assert config.base_model_name_or_path==MODEL_ID
    assert config.revision==REVISION


def test_saved_tokenizer_parity_covers_inputs_labels_and_attention_masks():
    examples,_=tokenize_rows(Tokenizer())
    _,audit=verify_saved_tokenizer(examples,Tokenizer(),256)
    assert audit['passed'] and audit['records_compared']==2
    examples[0]['labels'][-1]=-100
    with pytest.raises(RuntimeError,match='Saved tokenizer changed'):
        verify_saved_tokenizer(examples,Tokenizer(),256)
