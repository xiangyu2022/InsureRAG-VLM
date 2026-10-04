"""Report all paired regressions without changing examples, labels or selection."""
from collections import Counter,defaultdict
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_insuranceqa_scale import aggregate

def run():
    run=ROOT/'reports/retention_v2';rows=read_jsonl(run/'test_evaluation/predictions.jsonl')
    rows += [{'cohort':'historical_government',**r} for r in read_jsonl(run/'historical_government/predictions.jsonl')]
    cases={c['id']:c for path in ['data/benchmarks/insuranceqa_v2/test.jsonl','data/benchmarks/multidomain_v1/test.jsonl','data/benchmarks/hicric_government_qa_v1/questions.jsonl'] for c in read_jsonl(ROOT/path)}
    answers={a['id']:a['text'] for path in ['data/training/retention_v2/answers.jsonl','data/benchmarks/hicric_government_qa_v1/answers.jsonl'] for a in read_jsonl(ROOT/path)}
    grouped=defaultdict(dict)
    for row in rows:grouped[(row['cohort'],row['id'])][row['arm']]=row
    changes=[];bycohort=defaultdict(Counter);bydomain=defaultdict(Counter)
    for (cohort,qid),arms in grouped.items():
        selected=arms['selected'];case=cases[qid]
        for comparator in ['bge','public','previous','previous_same_anchor']:
            old=arms[comparator];delta=selected['hit_at_10']-old['hit_at_10'];label='win' if delta>0 else 'loss' if delta<0 else 'tie'
            bycohort[cohort+' vs '+comparator][label]+=1
            domain=case.get('domain','government');bydomain[cohort+' / '+domain+' vs '+comparator][label]+=1
            if not delta:continue
            changes.append({'cohort':cohort,'id':qid,'domain':domain,'comparator':comparator,'change':label,
                'question':case['question'],'source_url':case.get('source_url'),
                'question_contains_number':bool(re.search(r'\d',case['question'])),
                'gold_answers':[{'id':a,'text':answers[a]} for a in case['gold_answer_ids']],
                'selected_top3':[{'id':a,'text':answers[a]} for a in selected['top_answer_ids'][:3]],
                'old_top3_ids':old['top_answer_ids'][:3]})
    old_audit=ROOT/'reports/insuranceqa_v2/retrieval_frozen/sensitivity_audit.json'
    excluded={c['id'] for c in json.loads(old_audit.read_text(encoding='utf8'))['near_duplicates']}
    sensitivity={arm:aggregate([r for r in rows if r['cohort']=='historical_insuranceqa' and r['arm']==arm and r['id'] not in excluded])
                 for arm in ['bge','public','previous','selected','selected_unanchored','previous_same_anchor']}
    result={'by_cohort':dict(bycohort),'by_domain':dict(bydomain),'all_changed_cases':changes,
            'historical_prespecified_duplicate_flag_exclusion':sensitivity,'original_sensitivity_audit_sha256':sha(old_audit),
            'selection_lock_sha256':sha(run/'selection.lock.json'),'script_sha256':sha(Path(__file__)),
            'purpose':'Post-evaluation description only; no deletions, relabeling or further tuning.',
            'caveat':'Automatic number-presence flag is not an adjudicated error cause; unlabelled plausible answers may exist.'}
    write_json(result,run/'error_analysis.json');print(json.dumps(result['by_cohort'],indent=2))

if __name__=='__main__':run()
