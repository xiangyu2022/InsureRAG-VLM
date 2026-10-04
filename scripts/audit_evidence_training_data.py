"""Independently verify source evidence, annual-report isolation and negative labels."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from scripts.prepare_reranker_training import normalize
from src.insurerag_vlm.finqa_evidence import page_evidence,normalized_evidence

def load(p):return json.loads(p.read_text(encoding='utf8'))


if __name__=='__main__':
    run=ROOT/'reports/evidence_reranker_v1';data=ROOT/'data/training/evidence_reranker_v1';fixture=ROOT/'data/benchmarks/finqa_evidence_v1'
    for filename in ['data_manifest.lock.json','manifest.lock.json']:
        if (data/filename).exists():
            for n,h in load(data/filename)['files'].items():assert sha(data/n)==h,n
    for n,h in load(fixture/'manifest.lock.json')['files'].items():assert sha(fixture/n)==h,n
    raw=ROOT.parent/'finqa_download';provenance=load(raw/'provenance.json')
    for name,item in provenance['files'].items():assert sha(raw/name)==item['sha256']
    records={r['id']:r for s in ['train','dev','test'] for r in load(raw/f'dataset/{s}.json')}
    pages={}
    for r in records.values():
        source=page_evidence(r)
        if r['filename'] in pages:assert pages[r['filename']]==source
        pages[r['filename']]=source
    expected=[a for p in sorted(pages) for a in pages[p]]
    actual=read_jsonl(fixture/'answers.jsonl');assert actual==expected
    evidence={a['id']:a for a in actual};cases={s:read_jsonl(fixture/f'{s}.jsonl') for s in ['train','valid','test']}
    reports={s:{c['source_group'] for c in rows} for s,rows in cases.items()}
    assert not(reports['train']&reports['valid'] or reports['train']&reports['test'] or reports['valid']&reports['test'])
    for split,rows in cases.items():
        for c in rows:
            r=records[c['upstream_id']];assert c['question']==r['qa']['question']
            assert c['source_page']==r['filename'] and c['source_group']==r['filename'].rsplit('/',1)[0]
            keys=[evidence[a]['upstream_evidence_key'] for a in c['gold_answer_ids']]
            assert keys==list(r['qa']['gold_inds'])
            assert all(normalized_evidence(evidence[a]['text'])==normalized_evidence(r['qa']['gold_inds'][key]) for a,key in zip(c['gold_answer_ids'],keys))
    assert {c['upstream_id'] for c in cases['test']}=={r['id'] for r in load(raw/'dataset/test.json')}
    allanswers=read_jsonl(data/'answers.jsonl');lookup={a['id']:a for a in allanswers};normal={a:normalize(v['text']) for a,v in lookup.items()}
    forbidden=set(load(data/'isolation.json')['forbidden_answer_ids']);groups=read_jsonl(data/('train_groups.jsonl' if (data/'train_groups.jsonl').exists() else 'query_groups.jsonl'))
    duplicates=[];pairs=0
    for g in groups:
        positive=set(g['positive_ids']);negative=set(g.get('negative_ids',[]))
        assert not (positive&negative or (positive|negative)&forbidden)
        assert not ({normal[a] for a in positive}&{normal[a] for a in negative})
        if g['source_domain']=='financial_report':
            assert g['source_group'] in reports['train']
            assert all(lookup[a]['source_group']==g['source_group'] for a in positive|negative)
        if negative:
            assert set(g['anchor_scores'])==positive|negative and set(g['negative_sources'])==negative
        pairs+=len(negative)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'../models/insurerag-condition-listwise-v1-seed-123/epoch-1',local_files_only=True,trust_remote_code=False)
    lengths={}
    for split,rows in cases.items():
        pairs0=[(c,a) for c in rows for a in c['gold_answer_ids']]
        encoded=tokenizer([c['question'] for c,a in pairs0],[evidence[a]['text'] for c,a in pairs0],truncation=False,verbose=False)['input_ids']
        longids=sorted({c['id'] for (c,a),tokens in zip(pairs0,encoded) if len(tokens)>512})
        lengths[split]={'questions':len(rows),'gold_pairs':len(pairs0),'gold_pairs_over_512':sum(len(t)>512 for t in encoded),
                        'question_ids_with_long_gold_pair':longids,'max_pair_tokens':max(map(len,encoded)),
                        'questions_more_than_five_gold_facts':sum(len(c['gold_answer_ids'])>5 for c in rows)}
    keyword=re.compile(r'\b(?:insurance|insured|insurer|insurers|premium|premiums|deductible|deductibles|copay|medicare|medicaid|annuity|annuities|coverage)\b',re.I)
    validtexts={normalize(c['question']) for c in cases['valid']};testtexts={normalize(c['question']) for c in cases['test']}
    company={s:{g.split('/')[0] for g in v} for s,v in reports.items()}
    actual_training_reports={g['source_group'] for g in groups if g['source_domain']=='financial_report'}
    actual_training_companies={r.split('/')[0] for r in actual_training_reports}
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'status':'verified',
                'all_source_rows_and_original_gold_facts_match':True,'positive_negative_or_forbidden_overlap':0,
                'training_questions':len(groups),'training_domains':dict(Counter(g['source_domain'] for g in groups)),
                'training_negative_pairs':pairs,'annual_report_counts':{s:len(v) for s,v in reports.items()},
                'source_page_counts':{s:len({c['source_page'] for c in rows}) for s,rows in cases.items()},
                'company_counts':{s:len(v) for s,v in company.items()},'train_test_shared_companies':len(company['train']&company['test']),
                'fixture_source_counts_precede_training_question_and_negative_filters':True,
                'actual_training_report_count':len(actual_training_reports),'actual_training_company_count':len(actual_training_companies),
                'actual_train_test_shared_companies':len(actual_training_companies&company['test']),
                'valid_test_exact_question_text_overlap':len(validtexts&testtexts),'independent_new_company_claim':False,
                'new_test_questions':len(cases['test']),'new_test_questions_excluded':0,'gold_pair_lengths':lengths,
                'insurance_keyword_rule':keyword.pattern,'insurance_keyword_slices':{s:[c['id'] for c in rows if keyword.search(c['question'])] for s,rows in cases.items()},
                'keyword_slice_is_not_expert_insurance_classification':True,'qa_model_input_or_upstream_retrieved_scores_used':False,
                'fixture_sha256':sha(fixture/'manifest.lock.json'),'data_manifest_sha256':sha(data/'data_manifest.lock.json'),
                'training_manifest_sha256':sha(data/'manifest.lock.json') if (data/'manifest.lock.json').exists() else None,
                'code_sha256':sha(Path(__file__))},run/'data_verification.json')
    print(json.dumps({'status':'verified','training_questions':len(groups),'negative_pairs':pairs,
                      'new_test_questions':len(cases['test']),'valid_test_exact_question_text_overlap':len(validtexts&testtexts),
                      'long_gold_pairs':{s:v['gold_pairs_over_512'] for s,v in lengths.items()}}))
