"""Independent artifact consistency checks; no model invocation or metric tuning."""
import hashlib,importlib.metadata,json,math,platform,subprocess,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha,verify_test_lock

def lines(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]

def main():
    verify_test_lock('baseline');lock=read(ROOT/'reports/source_holdout_v1/selection.lock.json')
    assert all(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==h for n,h in lock['inference_files_sha256'].items())
    answers={r['id']:r for r in read(LOCAL/'sealed/answers.json')}
    pairs=read(LOCAL/'extracted_pairs.json')
    assert len(pairs)==218 and len(answers)==219
    assert all(len(r['text'])<=2000 for r in answers.values())
    seen=set();reports={};generation_total=0;no_gold_prompt=0
    for split in ['dev','test']:
        cases=read(LOCAL/'sealed'/(split+'.json'));lookup={r['id']:r for r in cases}
        assert not seen & set(lookup);seen.update(lookup)
        for arm in (['baseline','cross_only','original_query','cross_quarter','cross_quarter_timing'] if split=='dev' else ['baseline','cross_quarter']):
            folder=LOCAL/(split+'_'+arm);rows=lines(folder/'retrieval.jsonl');done=read(folder/'completion.json')
            assert sha(folder/'retrieval.jsonl')==done['retrieval_sha256']
            assert Counter(r['id'] for r in rows)==Counter(lookup.keys())
            complete=Counter();total=Counter();recall10=Counter();lost=0
            for r in rows:
                gold=set(lookup[r['id']]['gold_answer_ids']);group=lookup[r['id']]['publisher'];order=r['order']
                assert len(order)==10 and len(set(order))==10 and len(r['context'])<=8000
                assert [s['answer_id'] for s in r['results']]==order
                assert r['question']==lookup[r['id']]['question']
                assert r['reranked_candidates']<=200
                present=gold<=set(order[:5])
                retained=present and all(' '.join(answers[g]['text'].split()) in ' '.join(r['context'].split()) for g in gold)
                lost+=int(present and not retained);complete[group]+=int(retained);total[group]+=1
                recall10[group]+=len(gold & set(order))/len(gold)
            primary=sum(complete[g]/total[g] for g in total)/len(total)
            summary=read(folder/'summary.json')
            assert math.isclose(primary,summary['publisher_macro']['complete_after_packing'],abs_tol=1e-12)
            assert math.isclose(sum(recall10[g]/total[g] for g in total)/len(total),summary['publisher_macro']['recall_at_10'],abs_tol=1e-12)
            assert lost==0
            reports[split+'_'+arm]={'questions':len(rows),'complete_answers':sum(complete.values()),'publisher_macro_primary':primary,'packing_lost_complete':lost}
            if split=='test':assert datetime.fromisoformat(read(folder/'protocol.json')['started_utc'])>datetime.fromisoformat(lock['locked_utc'])
            if not (folder/'generation').exists():continue
            gfolder=folder/'generation';outputs=lines(gfolder/'answers.jsonl');gd=read(gfolder/'completion.json')
            assert sha(gfolder/'answers.jsonl')==gd['answers_sha256']
            expected=Counter((r['id'],cohort) for r in cases for cohort in ['retrieved','empty_control'])
            assert Counter((r['id'],r['cohort']) for r in outputs)==expected
            retrieval={r['id']:r for r in rows}
            for r in outputs:
                assert not r.get('error')
                context='' if r['cohort']=='empty_control' else retrieval[r['id']]['context']
                assert r['context']==context
                assert r['prompt']=='Evidence:\n'+context+'\n\nQuestion:\n'+lookup[r['id']]['question']
                no_gold_prompt+=1
                meta=r['generation'];options=meta['generation_options']
                assert meta['model_digest']=='2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
                assert options['temperature']==0 and options['seed']==42 and options['num_ctx']==4096 and options['num_predict']==384
                assert meta['thinking'] is False
                assert not r['served']['generation_truncated'] and not r['served']['answer_repaired']
            generation_total+=len(outputs)
    assert len(seen)==84 and generation_total==336
    inventory=read(LOCAL/'history_v2/historical_inventory.json')
    assert not inventory['failures']
    audit=read(ROOT/'reports/source_holdout_v1/history_audit.json')
    assert not any(audit['frozen_split_flagged_by_either_audit'].values())
    log=(LOCAL/'full_pytest.txt').read_text(encoding='utf8')
    assert '357 passed, 58 subtests passed' in log
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
            'frozen_inference_files_equal_git_blobs':True,'selection_lock_sha256':sha(ROOT/'reports/source_holdout_v1/selection.lock.json'),
            'independent_retrieval_recalculation':reports,'all_fixed_question_ids_preserved':True,
            'generation_outputs':generation_total,'exact_label_free_prompt_construction_verified':no_gold_prompt,
            'test_started_after_selection_lock':True,'all_model_digests_and_options_verified':True,
            'errors':0,'truncations':0,'answer_repairs':0,
            'cpu_tests':{'passed':357,'subtests_passed':58,'seconds':10.41,'log_sha256':sha(LOCAL/'full_pytest.txt')},
            'offline_smoke':read(ROOT/'reports/source_holdout_v1/offline_smoke.json')['status'],
            'runtime_cleanup':read(LOCAL/'runtime_cleanup.json'),
            'versions':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','scikit-learn','scipy','beautifulsoup4','pytest','threadpoolctl']}},
            'platform':platform.system(),'hardware':'NVIDIA RTX 4070 Laptop GPU, 8 GiB',
            'limits':'Consistency checks do not establish semantic correctness, independent publisher sufficiency, or unseen pretraining.'}
    write(ROOT/'reports/source_holdout_v1/verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
