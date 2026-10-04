import base64
import json
import shutil
from pathlib import Path

import fitz
import pytest
from PIL import Image
from scripts.build_visual_counterfactual import design,render_variant,sha
from scripts.eval_visual_counterfactual import verify_fixture,model_request,score_response,SYSTEM,SCHEMA
from src.insurerag_vlm.vlm import VLMClient

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'data/benchmarks/visual_counterfactual_v1'


def test_frozen_images_change_targets_while_questions_stay_identical():
    lock,cases=verify_fixture(FIXTURE)
    spec,expected=design()
    assert cases==expected
    assert len({sha(FIXTURE/v['image']) for v in spec['variants']})==3
    assert lock['frozen_before_model_inference']
    assert all(not c['answer_keys'] for c in cases if not c['answerable'])


@pytest.mark.parametrize('asset', ['assets/visual_A.pdf', 'assets/visual_A.png'])
def test_frozen_manifest_rejects_any_asset_byte_change(tmp_path, asset):
    copy=tmp_path/'fixture'
    shutil.copytree(FIXTURE,copy)
    path=copy/asset
    # Appending bytes can leave the visible document unchanged; the exact
    # inference input must still fail its original frozen checksum.
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Frozen visual fixture changed'):
        verify_fixture(copy)


def test_rebuild_is_locally_repeatable_and_preserves_frozen_content(tmp_path):
    verify_fixture(FIXTURE)
    spec,_=design()
    first=tmp_path/'first'
    second=tmp_path/'second'
    for v in spec['variants']:
        render_variant(v,first,spec['render_dpi'])
        render_variant(v,second,spec['render_dpi'])
        for kind in ('pdf','image'):
            assert sha(first/v[kind])==sha(second/v[kind])

        # Renderer/library builds may encode equivalent assets differently on
        # another OS. Check content across platforms, not compressed file bytes.
        with fitz.open(first/v['pdf']) as rebuilt,fitz.open(FIXTURE/v['pdf']) as frozen:
            assert len(rebuilt)==len(frozen)==1
            page=rebuilt[0]
            assert tuple(page.rect)==tuple(frozen[0].rect)==(0,0,612,792)
            text=' '.join(page.get_text('text').split())
            assert text==' '.join(frozen[0].get_text('text').split())
            for value in (
                f'${v["dwelling_limit"]:,}',f'${v["wind_hail_deductible"]:,}',
                f'${v["personal_property_limit"]:,}',f'${v["personal_liability_limit"]:,}',
                v['excluded_cause'],
            ):
                assert value in text
            assert 'premium' not in text.lower()
            assert all(page.rect.contains(fitz.Rect(word[:4])) for word in page.get_text('words'))
            scale=spec['render_dpi']/72
            rendered=page.get_pixmap(matrix=fitz.Matrix(scale,scale))
            with Image.open(first/v['image']) as rebuilt_image,Image.open(FIXTURE/v['image']) as frozen_image:
                assert rebuilt_image.size==frozen_image.size==(1224,1584)
                assert rebuilt_image.mode==frozen_image.mode=='RGB'
                assert rebuilt_image.tobytes()==rendered.samples


def test_actual_image_transport_receives_question_only_without_gold_or_ocr(monkeypatch):
    monkeypatch.setenv('INSURERAG_USE_OLLAMA','1')
    captured=[]
    class Reply:
        def __init__(self,value):self.value=value
        def raise_for_status(self):pass
        def json(self):return self.value
    def get(url,**kwargs):
        if url.endswith('/api/version'):return Reply({'version':'mock-test'})
        return Reply({'models':[{'name':'qwen3.5:4b','model':'qwen3.5:4b','digest':'frozen-test-digest','capabilities':['vision']} ]})
    def post(url,**kwargs):
        assert url.endswith('/api/chat')
        captured.append(kwargs['json'])
        return Reply({'model':'qwen3.5:4b','done':True,'done_reason':'stop',
            'message':{'content':'{"answer":"","evidence":"","abstain":true}'}})
    monkeypatch.setattr('src.insurerag_vlm.vlm.requests.get',get)
    monkeypatch.setattr('src.insurerag_vlm.vlm.requests.post',post)
    client=VLMClient('ollama:qwen3.5:4b',ollama_base_url='http://localhost:11435',use_hf_api=False,
        hf_api_token=None,openai_api_key=None,anthropic_api_key=None)
    _,cases=verify_fixture(FIXTURE)
    for case in cases:
        model_request(client,case,FIXTURE)
        payload=captured[-1]
        assert payload['messages'][0]=={'role':'system','content':SYSTEM}
        user=payload['messages'][1]
        assert set(user)=={'role','content','images'}
        assert user['content']==case['question']
        assert len(user['images'])==1
        assert base64.b64decode(user['images'][0])==(FIXTURE/case['image']).read_bytes()
        request_text=SYSTEM+'\n'+user['content']
        assert all(c['reference_answer'].lower() not in request_text.lower() for c in cases if c['answerable'])
        assert payload['format']==SCHEMA
    assert len(captured)==12


def test_wrong_value_truncation_and_extra_json_keys_do_not_pass():
    _,cases=verify_fixture(FIXTURE);case=cases[0]
    valid={'answer':case['reference_answer'],'evidence':'Dwelling coverage limit '+case['reference_answer'],'abstain':False}
    assert score_response(case,json.dumps(valid),{'done_reason':'stop'})['evidence_key_pass']
    wrong=dict(valid,answer=str(int(case['reference_answer'])*10))
    assert not score_response(case,json.dumps(wrong),{'done_reason':'stop'})['key_pass']
    assert not score_response(case,json.dumps(valid),{'done_reason':'length'})['key_pass']
    assert not score_response(case,json.dumps(dict(valid,extra=True)),{'done_reason':'stop'})['valid_json']
    absent=next(c for c in cases if not c['answerable'])
    assert score_response(absent,'{"answer":"","evidence":"","abstain":true}',{'done_reason':'stop'})['strict_abstention']
    assert not score_response(absent,'{"answer":"$500","evidence":"","abstain":true}',{'done_reason':'stop'})['strict_abstention']
