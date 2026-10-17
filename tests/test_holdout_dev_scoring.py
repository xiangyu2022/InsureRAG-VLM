from copy import deepcopy
import hashlib
import json
import pytest

from src.insurerag_vlm.holdout_dev_runner import fingerprint
from src.insurerag_vlm.holdout_dev_scoring import percentile, score_run, select_arm, write_selection_lock
from src.insurerag_vlm.holdout_retrieval import ARMS


def fixture():
    requests = [dict(id=q,question=f'Invented question {q}?') for q in ['a','b','c','d','refusal']]
    chunks = ['one','alias','two','three','context','distractor']
    items = []
    for r in requests:
        refusal = r['id']=='refusal'
        items.append(dict(id=r['id'],question_sha256=hashlib.sha256(r['question'].encode()).hexdigest(),
                          document_family='large' if r['id'] in 'abc' else r['id'],
                          source_necessity_reviewed=True,global_semantic_alias_review_complete=True,
                          task='insufficient_evidence' if refusal else 'ordinary_qa',
                          answer_relevance_gold_applicable=not refusal,
                          units=[] if refusal else [dict(id='u1',alternative_chunk_sets=[['one'],['alias']])],
                          gap_context_only=[dict(chunk_ids=['context'])] if refusal else []))
    mapping = dict(whole_dev_clearance=True,global_equivalent_alias_review_complete=True,items=items)
    run = dict(version=1,execution='standalone_per_arm',split='DEV',arms=list(ARMS),bindings={},
               requests_sha256=fingerprint(requests),chunk_ids_sha256=fingerprint(chunks))
    records = []
    for arm in ARMS:
        for r in requests:
            # BM25 gets three questions in one family; dense gets the single smaller family.
            good = (arm=='bm25' and r['id'] in 'abc') or (arm=='dense_original' and r['id']=='d')
            row = dict(id=r['id'],arm=arm,request_sha256=fingerprint(r),run_sha256=fingerprint(run),
                       order=['one'] if good else ['context'],status='success',elapsed_seconds=1.0)
            row['record_sha256'] = fingerprint(row); records.append(row)
    return requests,chunks,mapping,run,records


def rehash(row):
    row.pop('record_sha256',None);row['record_sha256']=fingerprint(row)


def test_family_macro_refusal_not_applicable_and_intended_failures():
    args = fixture()
    # One large-family BM25 failure makes question macro favor BM25 (.5 vs .25),
    # while equal-family complete-evidence favors dense (.5 vs 1/3).
    failed = next(r for r in args[-1] if r['id']=='a' and r['arm']=='bm25')
    failed.update(status='failed',order=[],elapsed_seconds=9);rehash(failed)
    report = score_run(*args)
    bm, dense = [report['arms'][a] for a in ARMS[:2]]
    assert bm['intended']==5 and bm['answerable_intended']==4 and bm['answerable_failed']==1
    assert bm['question_macro']['complete_evidence_at_5']==.5
    assert dense['question_macro']['complete_evidence_at_5']==.25
    assert bm['family_macro']['complete_evidence_at_5']==pytest.approx(1/3)
    assert dense['family_macro']['complete_evidence_at_5']==.5
    assert bm['latency_seconds']['p95']==pytest.approx(7.4)
    assert select_arm(report)['arm']=='dense_original'
    refusal = next(r for r in report['cases'] if r['id']=='refusal')
    assert refusal['complete_evidence_at_5'] is None and refusal['recall_at_5'] is None
    assert refusal['gap_context_hit_at_5']==1


def test_equivalent_alias_and_multichunk_necessity_have_different_metrics():
    args = fixture()
    item = args[2]['items'][0]
    item['units'].append(dict(id='u2',alternative_chunk_sets=[['two','three']]))
    row = args[-1][0]; row['order']=['alias','two'];rehash(row)
    case = next(r for r in score_run(*args)['cases'] if r['id']=='a' and r['arm']=='bm25')
    assert case['complete_evidence_at_5']==0 and case['unit_recall_at_5']==.5
    assert case['recall_at_5']==.5
    row['order']=['alias','two','three'];rehash(row)
    case = next(r for r in score_run(*args)['cases'] if r['id']=='a' and r['arm']=='bm25')
    assert case['complete_evidence_at_5']==1 and case['recall_at_5']==.75
    assert case['chunk_complete_at_5']==0


@pytest.mark.parametrize('mutation', ['missing','extra','draft','alias_unreviewed','wrong_question','shared','bad_time'])
def test_incomplete_unreviewed_or_mismatched_runs_cannot_score(mutation):
    args = fixture()
    if mutation=='missing':args[-1].pop()
    elif mutation=='extra':args[-1].append(deepcopy(args[-1][0]))
    elif mutation=='draft':args[2]['whole_dev_clearance']=False
    elif mutation=='alias_unreviewed':args[2]['items'][0]['global_semantic_alias_review_complete']=False
    elif mutation=='wrong_question':args[2]['items'][0]['question_sha256']='a'*64
    elif mutation=='shared':args[3]['execution']='shared'
    else:args[-1][0]['elapsed_seconds']=float('nan');rehash(args[-1][0])
    with pytest.raises(ValueError):score_run(*args)


@pytest.mark.parametrize('field', ['complete_evidence_at_5','recall_at_5','ndcg_at_10','latency','order'])
def test_registered_selection_priority(field):
    report=score_run(*fixture())
    for a in ARMS:
        report['arms'][a]['family_macro']={m:.4 for m in ['complete_evidence_at_5','recall_at_5','ndcg_at_10']}
        report['arms'][a]['latency_seconds']['p95']=1
    favored=ARMS[-1]
    if field=='latency':report['arms'][favored]['latency_seconds']['p95']=.5
    elif field!='order':
        report['arms'][favored]['family_macro'][field]=.6
        if field=='complete_evidence_at_5':report['arms'][favored]['family_macro']['recall_at_5']=.1
        if field in ('complete_evidence_at_5','recall_at_5'):report['arms'][favored]['family_macro']['ndcg_at_10']=.1
    assert select_arm(report)['arm']==(ARMS[0] if field=='order' else favored)


def test_lock_is_immutable_requires_all_artifacts_and_never_authorizes_test(tmp_path):
    report=score_run(*fixture());path=tmp_path/'lock.json'
    with pytest.raises(ValueError):write_selection_lock(path,report,{'code':'a'*64})
    hashes={k:'a'*64 for k in ['dev_data','corpus','models','tokenizers','code','prompt','packing','rubric','protocol']}
    lock=write_selection_lock(path,report,hashes)
    assert lock['test_inference_authorized'] is False
    original=path.read_bytes()
    with pytest.raises(FileExistsError):write_selection_lock(path,report,hashes)
    assert path.read_bytes()==original
    report['arms'][ARMS[0]]['answerable_intended']-=1
    with pytest.raises(ValueError,match='denominators'):select_arm(report)


def test_undefined_quantiles_rejected():
    assert percentile([0,10],.95)==9.5
    for values in [[],[float('inf')],[-1]]:
        with pytest.raises(ValueError):percentile(values,.95)


def cli_fixture(tmp_path):
    from scripts import score_holdout_dev as cli
    from src.insurerag_vlm.holdout_corpus import digest, source_snapshot_digest
    from src.insurerag_vlm.holdout_standalone import run_requests
    requests,ids,mapping,_,_=fixture()
    text='Invented complete source text for a synthetic pipeline test.'
    doc=dict(id='synthetic',text=text,normalized_text_sha256=digest(text),source_sha256='a'*64)
    entry=dict(document_id=doc['id'],source_sha256='a'*64,text_sha256=digest(text),selection_basis='source_only',
               ranges=[[0,len(text)]],atomic_ranges=[],atomic_ranges_reviewed=True,
               review=dict(decision='include',scope_rationale='Invented fixture',record_sha256='b'*64))
    manifest=dict(version=1,selection_basis='source_only',source_snapshot_sha256=source_snapshot_digest([doc]),entries=[entry])
    paths={name:tmp_path/(name+'.json') for name in ['documents','manifest','requests','clearance','support-mapping','output']}
    for name,value in [('documents',doc),('manifest',manifest),('requests',requests),('support-mapping',mapping)]:
        paths[name].write_text(json.dumps(value),encoding='utf8')
    clearance=dict(status='READY_FOR_DEV_RETRIEVAL',split='DEV',requests_sha256=fingerprint(requests),
                   source_manifest_sha256=fingerprint(manifest),dev_quality_gate_passed=True,
                   source_family_isolation_reviewed=True,historical_contamination_reviewed=True,
                   support_necessity_reviewed=True,equivalent_support_aliases_reviewed=True,
                   support_mapping_sha256=cli.sha(paths['support-mapping']),retained_pool_sha256='d'*64,review_record_sha256='e'*64)
    paths['clearance'].write_text(json.dumps(clearance),encoding='utf8')
    paths['run']=tmp_path/'run';paths['run'].mkdir()
    chunks=[dict(id=cid,text='Invented '+cid) for cid in ids]
    (paths['run']/'corpus_chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in chunks),encoding='utf8')
    code=['scripts/run_holdout_dev_retrieval.py','scripts/eval_insuranceqa_scale.py',
          'src/insurerag_vlm/holdout_corpus.py','src/insurerag_vlm/holdout_dev_runner.py',
          'src/insurerag_vlm/holdout_retrieval.py','src/insurerag_vlm/query_adaptation.py',
          'src/insurerag_vlm/reranker.py','src/insurerag_vlm/domain_reranker.py',
          'src/insurerag_vlm/holdout_standalone.py','src/insurerag_vlm/holdout_dev_scoring.py',
          'src/insurerag_vlm/holdout_experiment.py','scripts/score_holdout_dev.py']
    bindings=dict(source_file_sha256=cli.sha(paths['documents']),source_manifest_sha256=cli.sha(paths['manifest']),
                  clearance_sha256=cli.sha(paths['clearance']),execution='standalone',chunks_sha256=fingerprint(chunks),
                  protocol_sha256=cli.sha(cli.ROOT/'reports/holdout1000_v1/evaluation_protocol_v1.json'),
                  code={name:cli.sha(cli.ROOT/name) for name in code},
                  models={'synthetic':{'model.safetensors':'a'*64,'tokenizer_config.json':'b'*64}})
    run_requests(requests,ids,lambda *_:dict(order=['one']),paths['run']/'standalone',bindings)
    argv=['score_holdout_dev.py']
    for name,path in paths.items():argv.extend(['--'+name,str(path)])
    return cli,paths,argv


def test_scoring_cli_full_synthetic_run_and_immutable_lock_without_model_import(tmp_path,monkeypatch,capsys):
    import sys
    cli,paths,argv=cli_fixture(tmp_path)
    monkeypatch.setitem(sys.modules,'torch',None)
    monkeypatch.setitem(sys.modules,'transformers',None)
    digest=json.loads((cli.ROOT/'reports/holdout1000_v1/evaluation_protocol_v1.json').read_text())['generation']['digest']
    files={}
    for name in ['prompt','packing','rubric','generation_model_receipt']:
        p=tmp_path/(name+'.json')
        value=(dict(status='VERIFIED_LOCAL_RUNTIME',model_digest=digest,tokenizer_bound_to_model_digest=True,
                    synthetic_fixture_only=True) if name=='generation_model_receipt' else dict(invented_fixture=name))
        p.write_text(json.dumps(value),encoding='utf8');files[name]=dict(path=p.name,sha256=cli.sha(p))
    spec=tmp_path/'freeze.json'
    spec.write_text(json.dumps(dict(status='REVIEWED_FREEZE_ARTIFACTS',model_and_tokenizer_hashes_verified=True,files=files)))
    lock=tmp_path/'lock.json';argv+=['--lock',str(lock),'--freeze-artifacts',str(spec)]
    monkeypatch.setattr(sys,'argv',argv)
    assert cli.main()==0
    report=json.loads(paths['output'].read_text())
    assert report['intended_requests']==5 and len(report['cases'])==30
    assert json.loads(lock.read_text())['test_inference_authorized'] is False
    before=lock.read_bytes()
    with pytest.raises(FileExistsError):cli.main()
    assert lock.read_bytes()==before


@pytest.mark.parametrize('problem',['source_pending','mapping_changed','missing_code','corpus_changed','shared'])
def test_scoring_cli_rejects_gates_before_writing_scores(tmp_path,monkeypatch,problem):
    import sys
    cli,paths,argv=cli_fixture(tmp_path)
    if problem=='source_pending':
        data=json.loads(paths['manifest'].read_text());data['entries'][0]['review']['decision']='pending'
        paths['manifest'].write_text(json.dumps(data))
    elif problem=='mapping_changed':paths['support-mapping'].write_text('{}')
    elif problem=='corpus_changed':(paths['run']/'corpus_chunks.jsonl').write_text('{"id":"changed"}')
    else:
        p=paths['run']/'standalone/run_manifest.json';data=json.loads(p.read_text())
        if problem=='missing_code':data['bindings']['code'].pop('src/insurerag_vlm/holdout_dev_scoring.py')
        else:data['bindings']['execution']='shared'
        p.write_text(json.dumps(data))
    monkeypatch.setattr(sys,'argv',argv)
    if problem=='source_pending':assert cli.main()==2
    else:
        with pytest.raises(ValueError):cli.main()
    assert not paths['output'].exists()
