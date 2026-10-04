"""Add frozen previous-specialist logits to the same source-isolated training pairs."""
from datetime import datetime,timezone
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import write_jsonl
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

def run():
    import torch
    torch.set_num_threads(4);start=time.perf_counter()
    parent=ROOT/'data/training/condition_v1';out=ROOT/'data/training/condition_listwise_v1';run=ROOT/'reports/condition_listwise_v1'
    plan=json.loads((run/'selection_protocol.json').read_text(encoding='utf8'))
    assert plan['mining_code_sha256']==sha(Path(__file__))
    lock=json.loads((parent/'manifest.lock.json').read_text(encoding='utf8'))
    for n,h in lock['files'].items():assert sha(parent/n)==h
    out.mkdir(parents=True,exist_ok=False)
    groups=read_jsonl(parent/'train_groups.jsonl');answers={a['id']:a['text'] for a in read_jsonl(parent/'answers.jsonl')}
    # These scores were computed by the same frozen specialist during training-only mining.
    modelpath=ROOT/'../models/insurerag-retention-v2/epoch-2'
    assert lock['student_miner']['files_sha256']['model.safetensors']==sha(modelpath/'model.safetensors')==plan['initial_weights_sha256']
    missing=[]
    for g in groups:
        g['anchor_scores']=dict(g['student_mining_scores'])
        missing.extend((g,a) for a in g['positive_ids']+g['negative_ids'] if a not in g['anchor_scores'])
    model=DomainCrossEncoder(modelpath,'cuda',64,512)
    for i in range(0,len(missing),256):
        batch=missing[i:i+256];scores=model.score_pairs([(g['question'],answers[a]) for g,a in batch])
        for (g,a),score in zip(batch,scores):g['anchor_scores'][a]=float(score)
        if i%8192==0:print(json.dumps({'anchor_pairs':min(i+256,len(missing)),'total':len(missing)}),flush=True)
    for g in groups:
        g['anchor_scores']={a:g['anchor_scores'][a] for a in g['positive_ids']+g['negative_ids']}
    for n in ['answers.jsonl','source_isolation.json']:(out/n).write_bytes((parent/n).read_bytes())
    write_jsonl(groups,out/'train_groups.jsonl')
    manifest={**{k:v for k,v in lock.items() if k not in ['files','code_sha256','created_utc','seconds']},
        'created_utc':datetime.now(timezone.utc).isoformat(),'parent_condition_training_sha256':sha(parent/'manifest.lock.json'),
        'anchor_model':model.fingerprint(),'new_anchor_pair_computations':model.pairs_scored,'seconds':time.perf_counter()-start,
        'same_questions_positives_negatives_as_parent':True,'code_sha256':sha(Path(__file__)),
        'files':{n:sha(out/n) for n in ['answers.jsonl','source_isolation.json','train_groups.jsonl']}}
    write_json(manifest,out/'manifest.lock.json');print(json.dumps({'groups':len(groups),'new_anchor_pairs':model.pairs_scored}),flush=True)

if __name__=='__main__':run()
