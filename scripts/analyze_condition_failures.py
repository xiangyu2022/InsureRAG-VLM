"""Diagnose current retrieval ceilings and retain conservative manual-review notes."""
from collections import Counter,defaultdict
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from src.insurerag_vlm.reranker import blend_scores

def run():
    previous=ROOT/'reports/retention_v2';out=ROOT/'reports/condition_v1';out.mkdir(exist_ok=True)
    cases={c['id']:c for c in read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/test.jsonl')}
    preds={r['id']:r for r in read_jsonl(previous/'test_evaluation/predictions.jsonl') if r['cohort']=='historical_insuranceqa' and r['arm']=='selected'}
    scores=read_jsonl(previous/'test_legacy_selected/scores.jsonl');counts=Counter();detail=[]
    for row in scores:
        gold=set(cases[row['id']]['gold_answer_ids']);union=set(row['candidate_ids']);pool=np.argsort(-blend_scores(row['dense'],row['sparse'],lexical_weight=.2),kind='stable')[:100]
        poolids={row['candidate_ids'][i] for i in pool}
        reason='hit_at_10' if preds[row['id']]['hit_at_10'] else 'missing_from_union' if not gold&union else 'lost_by_pool_cutoff' if not gold&poolids else 'reranking_below_10'
        counts[reason]+=1;detail.append({'id':row['id'],'question':cases[row['id']]['question'],'category':reason})
    notes={
      'clear_intent_or_condition_confusion':['iqa_v2_test_00398','iqa_v2_test_00475','iqa_v2_test_00627','iqa_v2_test_01940','gov2_q_c54dbbc71341f9a3'],
      'plausible_unlabelled_alternative_needs_expert_review':['iqa_v2_test_00104','iqa_v2_test_00147','iqa_v2_test_00439','iqa_v2_test_00905','iqa_v2_test_01413','iqa_v2_test_01495','iqa_v2_test_01558','iqa_v2_test_01655','iqa_v2_test_01672','iqa_v2_test_01755','iqa_v2_test_01820'],
      'temporal_scope_or_subjective_annotation':['iqa_v2_test_00171','iqa_v2_test_00397','iqa_v2_test_00725','iqa_v2_test_01409','iqa_v2_test_01731']}
    report={'current_checkpoint':'retention_v2/epoch-2','historical_faq_n':len(scores),'failure_stage_counts':dict(counts),
        'stage_details':detail,'assistant_review_triage':notes,'review_scope':'Qualitative triage of 20 historical FAQ regressions and one government regression; not independent expert adjudication; no relabeling.',
        'planned_response':['Mine negatives with the current student, including same-document neighbors for training-only government sources.',
            'Add unused publisher-written government Q/A and original numeric/time/condition-oriented SQuAD training questions.',
            'Reduce penalties on teacher-supported unlabelled alternatives, retaining original positives.',
            'Validate union200 vs hybrid100 candidate pool; compare the previous model with identical fusion.'],
        'historical_test_use':'Error-type diagnosis only; never converted into training questions, labels or negatives.',
        'code_sha256':sha(Path(__file__)),'input_predictions_sha256':sha(previous/'test_evaluation/predictions.jsonl')}
    write_json(report,out/'failure_diagnosis.json');print(json.dumps({'counts':dict(counts),'triage_counts':{k:len(v) for k,v in notes.items()}},indent=2))

if __name__=='__main__':run()
