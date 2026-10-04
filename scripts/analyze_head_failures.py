"""Diagnose previous head-ranking regressions; never create training examples."""
from collections import Counter
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from src.insurerag_vlm.reranker import blend_scores

if __name__=='__main__':
    prior=ROOT/'reports/query_adaptation_v1';chosen='seed_42_epoch_2_a50'
    cases={r['id']:r for r in read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/test.jsonl')}
    answers={r['id']:r['text'] for r in read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl')}
    preds={(r['arm'],r['id']):r for r in read_jsonl(prior/'test/predictions.jsonl') if r['cohort']=='insuranceqa'}
    scores={r['id']:r for r in read_jsonl(prior/'test/insuranceqa_scores.jsonl')}
    bad=[];reasons=Counter()
    for q,c in cases.items():
        old,new=preds['original',q],preds[chosen,q]
        if not old['hit_at_1'] or new['hit_at_1']:continue
        row=scores[q];arm=row['arms'][chosen];cross=dict(zip(row['candidate_ids'],row['cross']))
        ids=arm['candidate_ids'];cs=np.array([cross[a] for a in ids]);order=np.argsort(-cs,kind='stable')
        pure=[ids[i] for i in order];puregold=[i+1 for i,a in enumerate(pure) if a in c['gold_answer_ids']]
        best=min(puregold) if puregold else None
        category='fixed_reranker_alone_has_gold_first' if best==1 else 'fixed_reranker_also_misranks' if best else 'gold_missing_from_adapted_candidates'
        reasons[category]+=1
        bad.append({'id':q,'question':c['question'],'category':category,'pure_reranker_gold_rank':best,
                    'old_top1':{'id':old['top_answer_ids'][0],'text':answers[old['top_answer_ids'][0]]},
                    'adapted_top1':{'id':new['top_answer_ids'][0],'text':answers[new['top_answer_ids'][0]]},
                    'gold_answers':[{'id':a,'text':answers[a]} for a in c['gold_answer_ids']],
                    'gold_still_in_top10':bool(new['hit_at_10']),'used_for_training':False})
    out=ROOT/'reports/evidence_reranker_v1';out.mkdir(parents=True,exist_ok=True)
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'historical_diagnostic_only':True,
                'prior_test_reused_not_blind':True,'first_rank_regressions':len(bad),'categories':dict(reasons),'cases':bad,
                'prior_predictions_sha256':sha(prior/'test/predictions.jsonl'),'code_sha256':sha(Path(__file__))},out/'head_failure_diagnosis.json')
    print(json.dumps({'first_rank_regressions':len(bad),'categories':dict(reasons)}))
