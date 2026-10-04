"""Create a blinded training-only review queue; never invent expert relevance labels."""
from collections import Counter
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import write_jsonl

def run():
    data=ROOT/'data/training/condition_v1';groups=read_jsonl(data/'train_groups.jsonl');answers={a['id']:a for a in read_jsonl(data/'answers.jsonl')}
    queue=[];key=[]
    for domain,budget in [('insuranceqa',20),('government',20),('general',10)]:
        eligible=[g for g in groups if g['source_domain']==domain and max(g['teacher_scores'][a] for a in g['negative_ids'])>=max(g['teacher_scores'][a] for a in g['positive_ids'])-1]
        eligible.sort(key=lambda g:hashlib.sha256(('review|'+g['id']).encode()).hexdigest())
        for g in eligible[:budget]:
            positive=max(g['positive_ids'],key=lambda a:g['teacher_scores'][a]);negative=max(g['negative_ids'],key=lambda a:g['teacher_scores'][a])
            ids=[positive,negative];random.Random(g['id']).shuffle(ids);reviewid='review_'+hashlib.sha256(g['id'].encode()).hexdigest()[:12]
            queue.append({'review_id':reviewid,'question':g['question'],'domain':domain,'use':'training-only manual review; not applied to any current run',
                'question_source_group':g.get('source_group'),'candidates':[{'candidate':f'passage_{i+1}','text':answers[a]['text'],'source_url':answers[a].get('source_url'),
                    'label':None,'reason':None} for i,a in enumerate(ids)],'reviewer':None,'reviewed_at':None})
            key.append({'review_id':reviewid,'training_question_id':g['id'],'candidate_ids':{f'passage_{i+1}':a for i,a in enumerate(ids)},
                'original_author_positive':positive,'original_unlabeled_negative':negative,'public_teacher_scores':{a:g['teacher_scores'][a] for a in ids}})
    out=ROOT/'reports/condition_v1/label_review';out.mkdir(exist_ok=False)
    write_jsonl(queue,out/'blind_queue.jsonl');write_jsonl(key,out/'mapping_do_not_show_reviewer.jsonl')
    write_json({'questions':len(queue),'domain_counts':dict(Counter(q['domain'] for q in queue)),'expert_labels_completed':0,
        'training_manifest_sha256':sha(data/'manifest.lock.json'),'code_sha256':sha(Path(__file__)),
        'files':{n:sha(out/n) for n in ['blind_queue.jsonl','mapping_do_not_show_reviewer.jsonl']},
        'label_schema':['directly_answers','partially_answers','wrong_scope_or_condition','insufficient_context','uncertain'],
        'guard':'No held-out queries included. Blank annotations are not training labels; adjudication and a new frozen dataset version are required before any use.'},out/'manifest.json')
    print(json.dumps({'questions':len(queue),'expert_labels_completed':0}))
if __name__=='__main__':run()
