"""Freeze audited source groups and labels before development inference."""
import argparse,hashlib,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_source_faq import chunk_text, norm
LOCAL=ROOT/'reports/source_holdout_v1/local'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(s):return hashlib.sha256(s.encode()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def validate_history_audit(audit,history_dir):
    inventory=json.loads((history_dir/'historical_inventory.json').read_text(encoding='utf8'))
    if inventory['failures']:raise ValueError('Historical scan incomplete')
    if audit['historical_file_sha256']!=sha(history_dir/'historical_texts.jsonl'):
        raise ValueError('Historical strings differ from the audited input')
    return inventory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--history-dir',type=Path,default=LOCAL);args=parser.parse_args()
    target=LOCAL/'sealed'
    if target.exists():raise ValueError('Sealed artifacts already exist; never overwrite a frozen split')
    audit=json.loads((LOCAL/'overlap_audit.json').read_text(encoding='utf8'))
    if audit['candidate_file_sha256']!=sha(LOCAL/'extracted_pairs.json'):raise ValueError('Audited pairs changed')
    inventory=validate_history_audit(audit,args.history_dir)
    all_pairs=json.loads((LOCAL/'extracted_pairs.json').read_text(encoding='utf8'))
    excluded={r['id'] for r in audit['rows'] if r['excluded']}
    domains={'ccpc_ireland':['ccpc.ie'],'hia_ireland':['hia.ie'],
             'fsra_ontario':['fsrao.ca','fsco.gov.on.ca'], 'oregon_dfr':['dfr.oregon.gov'],
             'australia_privatehealth':['privatehealth.gov.au']}
    seen_publishers={g for g,ds in domains.items() if any(sum(n for h,n in inventory['hosts'].items() if h.endswith(d)) for d in ds)}
    pairs=[r for r in all_pairs if r['id'] not in excluded and r['publisher'] not in seen_publishers]
    groups=sorted({r['publisher'] for r in pairs},key=lambda g:key('source-holdout-v1:42:'+g))
    assignment={g:('dev' if i%2==0 else 'test') for i,g in enumerate(groups)}
    answers={};cases={}
    for row in all_pairs:
        gold=[]
        for text in chunk_text(row['answer']):
            aid='sourcefaq:'+key(row['publisher']+'|'+norm(text))[:20]
            answers.setdefault(aid,{'id':aid,'text':text,'source_group':row['publisher'],
                                     'source_url':row['source_url'],'source_title':row['source_title']})
            if aid not in gold:gold.append(aid)
        cases[row['id']]={**row,'gold_answer_ids':gold,'label_kind':'publisher_heading_adjacent_answer',
                           'expert_adjudicated':False}
    split_cases={}
    target.mkdir(exist_ok=False)
    for split in ['dev','test']:
        selected=[]
        for group in groups:
            if assignment[group]!=split:continue
            eligible=sorted([r for r in pairs if r['publisher']==group],key=lambda r:key(r['id']))[:30]
            selected.extend({**cases[r['id']],'split':split} for r in eligible)
        split_cases[split]=selected
        write(target/(split+'.json'),selected)
    write(target/'answers.json',sorted(answers.values(),key=lambda r:r['id']))
    counts={s:dict(Counter(r['publisher'] for r in rr)) for s,rr in split_cases.items()}
    qualifies=all(len(c)>=3 and sum(c.values())>=30 for c in counts.values())
    manifest={'sealed_utc':datetime.now(timezone.utc).isoformat(),'source_assignment':assignment,'counts':counts,
              'eligible_independent_source_pilot':qualifies,
              'interpretation':'source-disjoint pilot' if qualifies else 'EXPLORATORY: preregistered minimum source/question counts not met',
              'prior_exposed_publishers_excluded':sorted(seen_publishers),'near_or_exact_exclusions':len(excluded),
              'new_answer_chunks':len(answers),'raw_publisher_qa_pairs':len(all_pairs),
              'chunker':'whitespace-normalized, non-overlapping word-boundary chunks <=2000 characters; no label-based span trimming',
              'question_input':'original publisher question, with no gold source or answer injected into inference',
              'new_index_content':'answer chunks only, no original question text indexed',
              'candidate_scope':'all 192231 pinned historical answer records plus every new FAQ answer chunk, identical for every arm',
              'baseline':'original BGE document embeddings; original/adapted query blend .5; dense100 union positive BM25 top100; lexical .2; selected trained cross encoder .5',
              'retrieval_depth':10,'packing_top_k':5,'context_budget_chars':8000,'page_cap_chars':2400,
              'fixed_denominators':True,'selection_test_status':'sealed; no test inference or per-question inspection permitted until selection lock',
              'files_sha256':{p.name:sha(p) for p in target.glob('*.json')},
              'overlap_audit_sha256':sha(LOCAL/'overlap_audit.json'),'preregistration_sha256':sha(ROOT/'reports/source_holdout_v1/preregistration.json'),
              'historical_inventory_sha256':sha(args.history_dir/'historical_inventory.json'),
              'historical_strings_sha256':audit['historical_file_sha256'],
              'historical_inventory_exclusions':inventory.get('exclusions',[]),
              'prepare_script_sha256':sha(ROOT/'scripts/prepare_source_faq.py'),'seal_script_sha256':sha(Path(__file__)),
              'limitations':['Original FAQ questions can omit document/jurisdiction context; other source answers may be semantically valid but unlabeled.',
                             'Full publisher-answer retention is stricter than sufficient semantic evidence and is not answer correctness.',
                             'Pretraining exposure unknown; full historical access limited to listed accessible project assets.',
                             'Near-duplicate screening conservatively excludes cosine upper bounds, so may exclude more than a full-vocabulary cosine screen.']}
    write(target/'manifest.json',manifest)
    write(ROOT/'reports/source_holdout_v1/split_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
