import hashlib,json
from pathlib import Path
from unittest.mock import Mock
import pytest
from scripts import acquire_source_holdout as acquire
from scripts.source_snapshot_io import verified_metadata
from scripts.prepare_source_faq import collect_pairs,extract_html,chunk_text

ANSWER='This synthetic insurance answer includes at least twelve separate words for deterministic extraction tests.'

def snapshot(folder,name='one',answer=ANSWER,question='What does this synthetic policy cover?'):
    raw=('<main><h2>'+question+'</h2><p>'+answer+'</p></main>').encode()
    artifact=name+'.html';(folder/artifact).write_bytes(raw)
    meta={'group':'hia_ireland','url':'https://public.example/'+name,'final_url':'https://public.example/'+name,
          'title':'Synthetic fixture','artifact':artifact,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    path=folder/(name+'.json');path.write_text(json.dumps(meta),encoding='utf8')
    return path,meta


def test_cached_snapshot_is_rechecked_before_reuse_and_publisher_cannot_change(tmp_path,monkeypatch):
    url='https://public.example/faq';name=acquire.sha(url.encode())[:20]
    path,meta=snapshot(tmp_path,name);meta['url']=url;path.write_text(json.dumps(meta),encoding='utf8')
    get=Mock(side_effect=AssertionError('Cache reuse must not make a network request'));monkeypatch.setattr(acquire.requests,'get',get)
    assert acquire.fetch({'group':'hia_ireland','url':url},tmp_path)['sha256']==meta['sha256']
    with pytest.raises(ValueError,match='publisher'):acquire.fetch({'group':'oregon_dfr','url':url},tmp_path)
    (tmp_path/meta['artifact']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='provenance'):acquire.fetch({'group':'hia_ireland','url':url},tmp_path)
    get.assert_not_called()


def test_artifact_path_cannot_escape_the_cache(tmp_path):
    path,meta=snapshot(tmp_path);meta['artifact']='../outside.html';path.write_text(json.dumps(meta),encoding='utf8')
    with pytest.raises(ValueError,match='basename'):verified_metadata(path)


def test_stream_limit_stops_before_reading_unbounded_body_and_preserves_failure_cache(tmp_path,monkeypatch):
    class Response:
        status_code=200;url='https://public.example/large';headers={};closed=False;consumed=0
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def raise_for_status(self):pass
        def iter_content(self,chunk_size):
            for chunk in [b'1234',b'5678',b'must not be consumed']:
                self.consumed+=1;yield chunk
    response=Response();get=Mock(return_value=response);monkeypatch.setattr(acquire.requests,'get',get)
    row={'group':'hia_ireland','url':response.url}
    result=acquire.fetch(row,tmp_path,max_bytes=5)
    assert 'exceeds acquisition limit' in result['error'] and 'artifact' not in result
    assert response.closed and response.consumed==2
    assert not list(tmp_path.glob('*.html'))
    assert acquire.fetch(row,tmp_path,max_bytes=5)['error']==result['error']
    assert get.call_count==1


def test_extraction_cannot_attribute_new_bytes_to_an_old_hash(tmp_path):
    path,meta=snapshot(tmp_path);(tmp_path/meta['artifact']).write_bytes(b'<main>tampered</main>')
    with pytest.raises(ValueError,match='provenance'):collect_pairs(tmp_path)


def test_identical_question_duplicates_are_recorded_but_conflicting_answers_fail(tmp_path):
    snapshot(tmp_path,'a');snapshot(tmp_path,'b')
    pairs,summary=collect_pairs(tmp_path)
    assert len(pairs)==1 and summary['identical_duplicate_questions']==1
    snapshot(tmp_path,'c',answer=ANSWER+' A different exception applies.')
    with pytest.raises(ValueError,match='Conflicting answers'):collect_pairs(tmp_path)


def test_question_heading_and_next_question_do_not_enter_extracted_answer():
    html='<main><h2>What does this synthetic policy cover?</h2><p>'+ANSWER+'</p><h2>When does the next provision apply?</h2><p>'+ANSWER+'</p></main>'
    rows=extract_html(html)
    assert len(rows)==2 and all(r['answer']==ANSWER for r in rows)


def test_definition_question_without_its_own_answer_does_not_steal_next_answer():
    html='<main><dl><dt>What is an unanswered first question?</dt><dt>What is the second question?</dt><dd>'+ANSWER+'</dd></dl></main>'
    rows=extract_html(html)
    assert len(rows)==1 and rows[0]['question']=='What is the second question?'


def test_chunk_budget_is_real_even_for_pathological_single_tokens():
    assert ' '.join(chunk_text('one two three four',max_chars=9))=='one two three four'
    assert all(len(c)<=9 for c in chunk_text('one two three four',max_chars=9))
    with pytest.raises(ValueError):chunk_text('unbrokenlongtoken',max_chars=4)


def test_prepare_cli_refuses_existing_extraction_before_reading_or_writing(tmp_path,monkeypatch):
    from scripts import prepare_source_faq
    original=b'frozen bytes';path=tmp_path/'extracted_pairs.json';path.write_bytes(original)
    monkeypatch.setattr('sys.argv',['prepare','--local-dir',str(tmp_path)])
    with pytest.raises(ValueError,match='already exist'):prepare_source_faq.main()
    assert path.read_bytes()==original


def test_sealing_binds_the_historical_strings_to_the_overlap_audit(tmp_path):
    from scripts.seal_source_holdout import validate_history_audit
    (tmp_path/'historical_inventory.json').write_text(json.dumps({'failures':[]}),encoding='utf8')
    raw=b'{"text":"synthetic history"}\n';path=tmp_path/'historical_texts.jsonl';path.write_bytes(raw)
    audit={'historical_file_sha256':hashlib.sha256(raw).hexdigest()}
    assert validate_history_audit(audit,tmp_path)=={'failures':[]}
    path.write_bytes(raw+b'changed')
    with pytest.raises(ValueError,match='differ from the audited input'):validate_history_audit(audit,tmp_path)


def test_sealing_does_not_create_an_empty_frozen_directory_when_audit_preflight_fails(tmp_path,monkeypatch):
    from scripts import seal_source_holdout as seal
    monkeypatch.setattr(seal,'LOCAL',tmp_path)
    monkeypatch.setattr('sys.argv',['seal','--history-dir',str(tmp_path)])
    with pytest.raises(FileNotFoundError):seal.main()
    assert not (tmp_path/'sealed').exists()
