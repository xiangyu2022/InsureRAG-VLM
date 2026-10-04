"""Summarize preregistered old-model controls on validation, without selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.eval_evidence_reranker import specs,partitions,RUN,check_contract
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.evidence_ranking import evidence_metrics,summarize_evidence

if __name__=='__main__':
    plan=check_contract();result={}
    for co,qfile,_,_ in specs('valid'):
        cases=read_jsonl(ROOT/qfile);candidates=read_jsonl(RUN/'valid_candidates'/f'{co}.jsonl')
        scores=read_jsonl(RUN/'valid_scores/previous'/f'{co}.jsonl');rows={w:[] for w in plan['cross_weights']}
        for q,r,s in zip(cases,candidates,scores):
            assert q['id']==r['id']==s['id'] and r['candidate_ids']==s['candidate_ids']
            for w in rows:
                order=rank_row({**r,'cross':s['cross']},{'pool':'union200','lexical_weight':.2,'cross_weight':w})
                rows[w].append(evidence_metrics(order,q['gold_answer_ids'],r['candidate_ids']))
        for part,ix in partitions(co,cases).items():
            result[f'{co}/{part}']={str(w):summarize_evidence([r[i] for i in ix]) for w,r in rows.items()}
    write_json({'validation_only':True,'changes_selection':False,'results':result,
                'selection_protocol_sha256':sha(RUN/'selection_protocol.json'),'code_sha256':sha(Path(__file__))},RUN/'validation_old_controls.json')
    print(json.dumps({key:{w:{m:r[m] for m in ['hit_at_1','hit_at_10','mrr_at_100','all_evidence_at_5']} for w,r in arms.items()} for key,arms in result.items()}))
