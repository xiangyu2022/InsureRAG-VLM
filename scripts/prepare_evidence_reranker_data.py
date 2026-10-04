"""Add source-verified financial-report evidence with annual-report isolation."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from scripts.prepare_query_adaptation_data import write_jsonl
from scripts.prepare_reranker_training import normalize
from src.insurerag_vlm.finqa_evidence import page_evidence,normalized_evidence


def main():
    raw=ROOT.parent/'finqa_download';provenance=json.loads((raw/'provenance.json').read_text(encoding='utf8'))
    for name,item in provenance['files'].items(): assert sha(raw/name)==item['sha256']
    original={s:json.loads((raw/f'dataset/{s}.json').read_text(encoding='utf8')) for s in ['train','dev','test']}
    pages={};by_split={};invalid=[]
    for split,rows in original.items():
        cases=[]
        for r in rows:
            evidence=page_evidence(r)
            if r['filename'] in pages: assert pages[r['filename']]==evidence
            else: pages[r['filename']]=evidence
            bykey={e['upstream_evidence_key']:e for e in evidence}
            missing=[k for k,t in r['qa']['gold_inds'].items() if k not in bykey or normalized_evidence(t)!=normalized_evidence(bykey[k]['text'])]
            if missing:
                invalid.append({'id':r['id'],'split':split,'reason':'unresolvable_original_evidence_index','keys':missing})
                continue
            cases.append({'id':'finqa_q_'+hashlib.sha256(r['id'].encode()).hexdigest()[:24],
                          'question':r['qa']['question'],'gold_answer_ids':[bykey[k]['id'] for k in r['qa']['gold_inds']],
                          'domain':'financial_report','source_group':r['filename'].rsplit('/',1)[0],
                          'source_page':r['filename'],'upstream_id':r['id'],'upstream_split':split,
                          'original_evidence_keys':list(r['qa']['gold_inds']),
                          'scope':'all released evidence rows from the supplied company/year report',
                          'annotation_origin':'FinQA original gold supporting-fact indices; not synthetic labels'})
        by_split[split]=cases
    assert invalid==[{'id':'RE/2010/page_120.pdf-2','split':'train','reason':'unresolvable_original_evidence_index','keys':['text_-1']}]
    heldtest={c['source_group'] for c in by_split['test']}
    helddev={c['source_group'] for c in by_split['dev']}
    filtered={
        'test':by_split['test'],
        'valid':[c for c in by_split['dev'] if c['source_group'] not in heldtest],
        'train':[c for c in by_split['train'] if c['source_group'] not in heldtest|helddev],
    }
    reportsets={s:{c['source_group'] for c in rows} for s,rows in filtered.items()}
    assert not (reportsets['train']&reportsets['valid'] or reportsets['train']&reportsets['test'] or reportsets['valid']&reportsets['test'])
    answers=[a for page in sorted(pages) for a in pages[page]]
    assert len({a['id'] for a in answers})==len(answers)
    out=ROOT/'data/benchmarks/finqa_evidence_v1';out.mkdir(parents=True,exist_ok=False)
    write_jsonl(answers,out/'answers.jsonl')
    for split,rows in filtered.items():write_jsonl(rows,out/f'{split}.jsonl')
    for name in ['README.md','LICENSE']:(out/('UPSTREAM_'+name)).write_bytes((raw/name).read_bytes())
    write_json(provenance,out/'provenance.json')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'raw_counts':{s:len(r) for s,r in original.items()},
                'counts':{s:len(r) for s,r in filtered.items()},'annual_reports':{s:len(r) for s,r in reportsets.items()},
                'source_pages':len(pages),'answer_rows':len(answers),'invalid_training_record':invalid,
                'official_page_split_has_no_overlap_but_annual_reports_overlap':True,
                'train_removed_heldout_annual_report':len(by_split['train'])-len(filtered['train']),
                'valid_removed_test_annual_report':len(by_split['dev'])-len(filtered['valid']),
                'gold_fact_text_verified_against_label_independent_source_serialization':True,
                'official_retriever_predictions_and_qa_model_input_never_used':True,
                'test_exclusions':0,'scope_is_report_conditioned_not_global_search_or_program_execution':True},out/'audit.json')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
                'code_sha256':sha(Path(__file__)),'serialization_code_sha256':sha(ROOT/'src/insurerag_vlm/finqa_evidence.py')},out/'manifest.lock.json')
    parent=ROOT/'data/training/query_adaptation_v1'
    oldgroups=read_jsonl(parent/'train_groups.jsonl');oldanswers=read_jsonl(parent/'answers.jsonl')
    allanswers=oldanswers+answers;lookup={a['id']:a for a in allanswers}
    sourceheld=heldtest|helddev
    priorisolation=json.loads((parent/'isolation.json').read_text(encoding='utf8'))
    forbidden=set(priorisolation['forbidden_answer_ids'])|{a['id'] for a in answers if a['source_group'] in sourceheld}
    forbidden_texts={normalize(lookup[a]['text']) for a in forbidden}
    forbidden.update(a['id'] for a in allanswers if not normalize(a['text']) or normalize(a['text']) in forbidden_texts)
    additions=[{'id':c['id'],'question':c['question'],'positive_ids':c['gold_answer_ids'],'source_domain':'financial_report',
                'source_group':c['source_group'],'source_page':c['source_page'],'original_question':True} for c in filtered['train']]
    groups=oldgroups+additions
    # Existing old groups were already near-duplicate filtered against historical tests.
    # Only the newly introduced held-out financial-report questions require this additional check.
    held=filtered['valid']+filtered['test']
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
    matrix=vectorizer.fit_transform([c['question'] for c in held]+[g['question'] for g in groups])
    nearest=NearestNeighbors(n_neighbors=1,metric='cosine',n_jobs=4).fit(matrix[:len(held)])
    similarities=[]
    for offset in range(len(held),matrix.shape[0],128):
        dist,indices=nearest.kneighbors(matrix[offset:offset+128])
        similarities.extend((float(1-d[0]),held[int(i[0])]['id']) for d,i in zip(dist,indices))
    accepted=[];excluded=[];seen=set()
    for g,(sim,nearest_id) in zip(groups,similarities):
        text=normalize(g['question'])
        reason='heldout_answer_text_or_report' if set(g['positive_ids'])&forbidden else 'duplicate_or_empty_question' if not text or text in seen else 'near_heldout_question' if sim>=.92 else None
        if reason:excluded.append({'id':g['id'],'reason':reason,'similarity':sim,'nearest_heldout':nearest_id})
        else:seen.add(text);accepted.append(g)
    train=ROOT/'data/training/evidence_reranker_v1';train.mkdir(parents=True,exist_ok=False)
    write_jsonl(allanswers,train/'answers.jsonl');write_jsonl(accepted,train/'query_groups.jsonl')
    write_json({'forbidden_answer_ids':sorted(forbidden),'forbidden_new_annual_reports':sorted(sourceheld),
                'exclusions':excluded,'near_question_threshold':.92,
                'historical_insuranceqa_shared_labels_retained':True,
                'parent_isolation_sha256':sha(parent/'isolation.json')},train/'isolation.json')
    priorflags=[{'id':g['id'],'new_heldout_id':near,'similarity':sim} for g,(sim,near) in zip(groups[:len(oldgroups)],similarities[:len(oldgroups)]) if sim>=.92]
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'training_questions':len(accepted),
                'source_domains':dict(Counter(g['source_domain'] for g in accepted)),
                'additional_financial_report_questions':sum(g['source_domain']=='financial_report' for g in accepted),
                'removed_previous_queries':sum(e['id'] in {g['id'] for g in oldgroups} for e in excluded),
                'raw_imported_questions':{s:len(r) for s,r in filtered.items()},'answer_candidates':len(allanswers),
                'eligible_training_answers':len(allanswers)-len(forbidden),'exclusion_counts':dict(Counter(e['reason'] for e in excluded)),
                'prior_query_encoder_near_question_flags':priorflags,
                'parent_manifest_sha256':sha(parent/'manifest.lock.json'),'fixture_sha256':sha(out/'manifest.lock.json'),
                'code_sha256':sha(Path(__file__)),
                'files':{n:sha(train/n) for n in ['answers.jsonl','query_groups.jsonl','isolation.json']}},train/'data_manifest.lock.json')
    print(json.dumps({'counts':{s:len(r) for s,r in filtered.items()},'train_domains':dict(Counter(g['source_domain'] for g in accepted)),
                      'additional_exclusions':dict(Counter(e['reason'] for e in excluded)),'answer_rows':len(answers),
                      'prior_question_exposure_flags':priorflags}),flush=True)


if __name__=='__main__':main()
