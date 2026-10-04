"""Describe model-confusing original training labels without relabeling them."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha


if __name__=='__main__':
    data=ROOT/'data/training/evidence_reranker_v1';groups=read_jsonl(data/'train_groups.jsonl')
    answers={a['id']:a for a in read_jsonl(data/'answers.jsonl')};counts=defaultdict(Counter);examples=defaultdict(list)
    condition=re.compile(r'\b(?:when|how long|if|after|before|until|unless|eligible|eligibility|require|requires|what happens|which|who)\b',re.I)
    numeric=re.compile(r'\b(?:how much|percent|percentage|ratio|average|increase|decrease|difference|total|net)\b|\d',re.I)
    for g in groups:
        scores=g['anchor_scores'];best=max(scores[a] for a in g['positive_ids']);worst=max(g['negative_ids'],key=lambda a:scores[a])
        d=g['source_domain'];counts[d]['unique_training_questions']+=1
        tags=['all']+(['condition_or_subject_keyword'] if condition.search(g['question']) else [])+(['numeric_keyword'] if numeric.search(g['question']) else [])
        for tag in tags:
            counts[d][tag+'_questions']+=1
            if scores[worst]>=best:counts[d][tag+'_negative_ties_or_outranks_best_positive']+=1
        if scores[worst]>=best and len(examples[d])<12:
            examples[d].append({'id':g['id'],'question':g['question'],'source_group':g.get('source_group'),
                                'gold':[{'id':a,'text':answers[a]['text'],'score':scores[a]} for a in g['positive_ids']],
                                'confusing_unlabelled':{'id':worst,'text':answers[worst]['text'],'score':scores[worst]},
                                'expert_review_completed':False,'negative_may_be_unlabelled_relevant':True})
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'training_only':True,'labels_modified':False,
                'counts':{d:dict(c) for d,c in counts.items()},'examples':dict(examples),
                'keyword_categories_are_heuristics_not_expert_annotations':True,
                'training_manifest_sha256':sha(data/'manifest.lock.json'),'code_sha256':sha(Path(__file__))},ROOT/'reports/evidence_reranker_v1/training_error_profile.json')
    print(json.dumps({d:{'n':c['unique_training_questions'],'confusing':c['all_negative_ties_or_outranks_best_positive']} for d,c in counts.items()}))
