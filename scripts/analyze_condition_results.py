"""Post-selection failures and source-family sensitivity; never used to refit."""
from collections import Counter,defaultdict
import json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_retention_v2 import source_paired
from src.insurerag_vlm.reranker import blend_scores

def run():
    run=ROOT/'reports/condition_v1';lock=json.loads((run/'selection.lock.json').read_text(encoding='utf8'))
    predictions=read_jsonl(run/'test_evaluation/predictions.jsonl');arms=defaultdict(dict)
    for r in predictions:arms[(r['cohort'],r['arm'])][r['id']]=r
    summaries={};cards=[]
    for cohort,fixture,answerpath,scorefolder in [
        ('historical_insuranceqa','data/benchmarks/insuranceqa_v2/test.jsonl','data/benchmarks/insuranceqa_v2/answers.jsonl','test_legacy_selected'),
        ('historical_multidomain','data/benchmarks/multidomain_v1/test.jsonl','data/training/retention_v2/answers.jsonl','test_mixed_selected'),
        ('historical_government','data/benchmarks/hicric_government_qa_v1/questions.jsonl','reports/condition_v1/historical_government_answers.jsonl','test_oldgov_selected'),
        ('fresh_government','data/benchmarks/condition_v1/test.jsonl','data/benchmarks/condition_v1/answers.jsonl','test_fresh_selected')]:
        cases=read_jsonl(ROOT/fixture);answers={a['id']:a['text'] for a in read_jsonl(ROOT/answerpath)};rows={r['id']:r for r in read_jsonl(run/scorefolder/'scores.jsonl')};counts=Counter();wins=[];losses=[]
        for c in cases:
            p=arms[(cohort,'selected')][c['id']];old=arms[(cohort,'previous_default')][c['id']];r=rows[c['id']];gold=set(c['gold_answer_ids'])
            if p['hit_at_10']:counts['hit_at_10']+=1
            else:
                if not gold&set(r['candidate_ids']):stage='absent_from_union'
                else:
                    pool=r['candidate_ids']
                    if lock['config']['pool']=='hybrid100':
                        order=np.argsort(-blend_scores(r['dense'],r['sparse'],lexical_weight=.2),kind='stable')[:100];pool=[pool[i] for i in order]
                    stage='candidate_pool_cutoff' if not gold&set(pool) else 'reranking_below_top10'
                counts[stage]+=1
                cards.append({'cohort':cohort,'id':c['id'],'question':c['question'],'stage':stage,'source':c.get('source_url',c.get('source_group')),
                    'previous_hit10':bool(old['hit_at_10']),'selected_gold_rank':next((i+1 for i,a in enumerate(p['top_answer_ids']) if a in gold),None),
                    'gold_answers':[{'id':a,'text':answers[a]} for a in c['gold_answer_ids']],
                    'top3':[{'id':a,'text':answers[a]} for a in p['top_answer_ids'][:3]]})
            if p['hit_at_10']>old['hit_at_10']:wins.append(c['id'])
            elif p['hit_at_10']<old['hit_at_10']:losses.append(c['id'])
        summaries[cohort]={'n':len(cases),'failure_stages':dict(counts),'wins_vs_previous_default':wins,'losses_vs_previous_default':losses}
    fresh=read_jsonl(ROOT/'data/benchmarks/condition_v1/test.jsonl')
    families=[{**c,'source_group':re.sub(r'(aca-part-\d+).*',r'\1',c['source_group'])} for c in fresh]
    sensitivity={arm:source_paired([arms[('fresh_government','selected')][c['id']] for c in fresh],
        [arms[('fresh_government',arm)][c['id']] for c in fresh],families) for arm in ['bge','previous_default','previous_matched','public_matched']}
    result={'summaries':summaries,'fresh_government_version_family_sensitivity':sensitivity,
        'selection_lock_sha256':sha(run/'selection.lock.json'),'post_test_diagnostic_only':True,
        'labels_modified':False,'failure_count_caveat':'A miss means missing original labeled evidence; plausible unlabelled answers remain possible.'}
    write_json(result,run/'failure_followup.json');write_json(cards,run/'failure_cases.json')
    print(json.dumps({n:{'n':v['n'],'stages':v['failure_stages'],'wins':len(v['wins_vs_previous_default']),'losses':len(v['losses_vs_previous_default'])} for n,v in summaries.items()},indent=2))

if __name__=='__main__':run()
