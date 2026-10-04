import json,sys
from scripts import inventory_source_history

def test_inventory_covers_unknown_text_fields_and_respects_explicit_exclusions(tmp_path,monkeypatch):
    source=tmp_path/'history';source.mkdir()
    (source/'old_eval.json').write_text(json.dumps({'novel_label_key':'A previously evaluated claim about insurance.',
        'question':'Which source supports this claim?','url':'https://public.example/faq'}),encoding='utf8')
    excluded=source/'restricted';excluded.mkdir()
    (excluded/'unread.json').write_text('must not be parsed',encoding='utf8')
    roots=tmp_path/'roots.json';roots.write_text(json.dumps([{'alias':'history','path':str(source),'exclude_relative':['restricted']}]),encoding='utf8')
    out=tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['inventory','--roots',str(roots),'--output',str(out)])
    inventory_source_history.main()
    result=json.loads((out/'historical_inventory.json').read_text(encoding='utf8'))
    texts=[json.loads(l)['text'] for l in (out/'historical_texts.jsonl').read_text(encoding='utf8').splitlines()]
    assert 'a previously evaluated claim about insurance' in texts
    assert result['unique_file_count']==1 and not result['failures']
    assert result['hosts']['public.example']==1
    assert result['exclusions'][0]['relative']=='restricted'

def test_unavailable_root_is_reported_as_a_gap(tmp_path,monkeypatch):
    roots=tmp_path/'roots.json';roots.write_text(json.dumps([{'alias':'absent','path':str(tmp_path/'not_here')}]),encoding='utf8')
    out=tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['inventory','--roots',str(roots),'--output',str(out)])
    inventory_source_history.main()
    result=json.loads((out/'historical_inventory.json').read_text(encoding='utf8'))
    assert result['failures']==[{'path':'absent','error':'Root unavailable'}]
