"""Frozen-selected model on the historical 416-question government regression set."""
from datetime import datetime,timezone
import json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import canonical_url
from scripts.eval_retention_v2 import predict,source_paired
from scripts.eval_insuranceqa_scale import aggregate
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def run():
    import torch
    torch.set_num_threads(4);start=time.perf_counter();run=ROOT/'reports/retention_v2'
    lock=json.loads((run/'selection.lock.json').read_text(encoding='utf8'));assert lock['selected_on']=='valid'
    modelpath=Path(lock['model_path']);assert sha(modelpath/'model.safetensors')==lock['selected_weights_sha256']
    old=ROOT/'reports/hicric_government_qa_v1/transfer_v1';summary=json.loads((old/'summary.json').read_text(encoding='utf8'))
    assert sha(old/'scores.jsonl')==summary['scores_sha256']
    rows=read_jsonl(old/'scores.jsonl');cases=read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/questions.jsonl')
    cases=[{**c,'source_group':canonical_url(c['source_url'])} for c in cases]
    assert [r['id'] for r in rows]==[c['id'] for c in cases]
    answers=read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl')+read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl')
    lookup={a['id']:a['text'] for a in answers};q={c['id']:c['question'] for c in cases};out=run/'historical_government';out.mkdir(exist_ok=False)
    model=DomainCrossEncoder(modelpath,'cuda',64,512)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_lock_sha256':sha(run/'selection.lock.json'),
              'source_scores_sha256':sha(old/'scores.jsonl'),'model':model.fingerprint(),'code_sha256':sha(Path(__file__)),
              'status':'Historical regression; not a new independent test','answer_candidates':27829}
    write_json(protocol,out/'protocol.json');scored=[]
    with (out/'scores.jsonl').open('w',encoding='utf8') as handle:
        for i in range(0,len(rows),8):
            batch=rows[i:i+8];scores=model.score_pairs([(q[r['id']],lookup[a]) for r in batch for a in r['candidate_ids']]);offset=0
            for row in batch:
                n=len(row['candidate_ids']);new={**row,'cross':scores[offset:offset+n].tolist()};offset+=n
                scored.append(new);handle.write(json.dumps(new)+'\n')
    public=[{**r,'cross':r['cross_base']} for r in rows];previous=[{**r,'cross':r['cross_trained']} for r in rows]
    beta=lock['domain_weight'];arms={n:predict(r,public,cases,b,n) for n,r,b in [('bge',public,1.),('public',public,1.),('previous',previous,1.),
                 ('selected',scored,beta),('selected_unanchored',scored,1.),('previous_same_anchor',previous,beta)]}
    with (out/'predictions.jsonl').open('w',encoding='utf8') as handle:
        for result in arms.values():
            for row in result:handle.write(json.dumps(row)+'\n')
    report={'status':'completed','seconds':time.perf_counter()-start,'summaries':{n:aggregate(r) for n,r in arms.items()},
            'selected_minus_comparators':{n:source_paired(arms['selected'],r,cases) for n,r in arms.items() if n!='selected'},
            'scores_sha256':sha(out/'scores.jsonl'),'predictions_sha256':sha(out/'predictions.jsonl'),
            'protocol_sha256':sha(out/'protocol.json'),'historical_test':True}
    write_json(report,out/'summary.json');print(json.dumps(report,indent=2))

if __name__=='__main__':run()
