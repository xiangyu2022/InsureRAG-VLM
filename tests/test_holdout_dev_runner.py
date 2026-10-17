import json
import pytest
from src.insurerag_vlm.holdout_corpus import digest, source_snapshot_digest
from src.insurerag_vlm.holdout_dev_runner import fingerprint, readiness, run_requests, validate_requests
from src.insurerag_vlm.holdout_retrieval import ARMS


def fixture():
    text='Original synthetic source with an unrelated distractor statement.'
    doc=dict(id='synthetic_source',text=text,normalized_text_sha256=digest(text),source_sha256='a'*64)
    entry=dict(document_id=doc['id'],source_sha256='a'*64,text_sha256=digest(text),selection_basis='source_only',
               ranges=[[0,len(text)]],atomic_ranges=[],atomic_ranges_reviewed=True,
               review=dict(decision='include',scope_rationale='Synthetic fixture',record_sha256='b'*64))
    manifest=dict(version=1,selection_basis='source_only',source_snapshot_sha256=source_snapshot_digest([doc]),entries=[entry])
    requests=[dict(id='synthetic_query',question='What does the invented source say?')]
    clearance=dict(status='READY_FOR_DEV_RETRIEVAL',split='DEV',requests_sha256=fingerprint(requests),
                   source_manifest_sha256=fingerprint(manifest),dev_quality_gate_passed=True,
                   source_family_isolation_reviewed=True,historical_contamination_reviewed=True,
                   support_necessity_reviewed=True,equivalent_support_aliases_reviewed=True,
                   support_mapping_sha256='c'*64,retained_pool_sha256='d'*64,review_record_sha256='e'*64)
    return [doc],manifest,requests,clearance


def test_readiness_reports_real_blockers_without_generation_or_model_runtime():
    docs,m,requests,c=fixture()
    assert readiness(docs,m,requests,c)['ready']
    assert readiness(docs,m,requests,c)['generation_required'] is False
    c['split']='TEST';c['equivalent_support_aliases_reviewed']=False
    m['entries'][0]['review']['decision']='pending'
    result=readiness(docs,m,requests,c)
    assert not result['ready']
    assert any('Corpus:' in v for v in result['blockers'])
    assert any('DEV clearance' in v for v in result['blockers'])
    assert any('equivalent_support_aliases' in v for v in result['blockers'])
    assert any('binding mismatch' in v for v in result['blockers'])


@pytest.mark.parametrize('extra',['answer','task','split','evidence','formula','family_id'])
def test_inference_requests_reject_gold_and_metadata_fields(extra):
    _,_,requests,_=fixture();requests[0][extra]='leaked value'
    with pytest.raises(ValueError,match='question text only'):validate_requests(requests)


def test_failed_intended_query_remains_and_exact_resume_does_not_repeat_models(tmp_path):
    requests=[dict(id='ok',question='One?'),dict(id='fail',question='Two?')];seen=[]
    def infer(request):
        seen.append(request)
        if request['id']=='fail':raise RuntimeError('Synthetic backend failure')
        return dict(orders={arm:['global-distractor'] for arm in ARMS})
    out=tmp_path/'run';summary=run_requests(requests,['global-distractor'],infer,out,{'version':1})
    assert (summary['intended_requests'],summary['successful_requests'],summary['failed_requests'])==(2,1,1)
    def forbidden(_):raise AssertionError('Resume must reuse successes AND failures')
    resumed=run_requests(requests,['global-distractor'],forbidden,out,{'version':1})
    assert resumed['reused_requests']==2 and resumed['failed_requests']==1
    assert seen==requests
    with pytest.raises(ValueError,match='differ'):run_requests(requests,['global-distractor'],infer,out,{'version':2})
    with pytest.raises(ValueError,match='differ'):run_requests(requests+[{ 'id':'new','question':'Changed?'}],['global-distractor'],infer,out,{'version':1})


def test_fabricated_candidate_cannot_be_success_and_modified_resume_is_rejected(tmp_path):
    requests=[dict(id='query',question='Question?')]
    def infer(_):return dict(orders={arm:['outside-corpus'] for arm in ARMS})
    result=run_requests(requests,['real-chunk'],infer,tmp_path/'run',{})
    assert result['failed_requests']==1
    path=tmp_path/'run'/(fingerprint(requests[0])+'.json')
    saved=json.loads(path.read_text());saved['status']='success';path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='content fingerprint'):run_requests(requests,['real-chunk'],infer,tmp_path/'run',{})


def test_interrupted_run_resumes_only_missing_requests(tmp_path):
    requests=[dict(id='a',question='First?'),dict(id='b',question='Second?')];seen=[]
    def interrupted(r):
        seen.append(r['id'])
        if r['id']=='b':raise KeyboardInterrupt()
        return dict(orders={arm:['chunk'] for arm in ARMS})
    with pytest.raises(KeyboardInterrupt):run_requests(requests,['chunk'],interrupted,tmp_path/'run',{})
    def completed(r):
        seen.append(r['id']);return dict(orders={arm:['chunk'] for arm in ARMS})
    result=run_requests(requests,['chunk'],completed,tmp_path/'run',{})
    assert seen==['a','b','b'] and result['reused_requests']==1
    assert result['intended_requests']==result['successful_requests']==2


def test_cli_readiness_stops_before_model_import_on_incomplete_sources(tmp_path, monkeypatch, capsys):
    import sys
    from scripts.run_holdout_dev_retrieval import main
    docs, manifest, requests, clearance = fixture()
    manifest['entries'][0]['review']['decision'] = 'pending'
    paths = {}
    for name, value in [('manifest', manifest), ('requests', requests), ('clearance', clearance)]:
        paths[name] = tmp_path/(name+'.json')
        paths[name].write_text(json.dumps(value), encoding='utf8')
    paths['documents'] = tmp_path/'documents.jsonl'
    paths['documents'].write_text('\n'.join(json.dumps(d) for d in docs), encoding='utf8')
    argv = ['run_holdout_dev_retrieval.py', '--check-only']
    for name, path in paths.items():argv.extend(['--'+name, str(path)])
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'transformers', None)
    assert main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result['ready'] is False and result['generation_required'] is False
