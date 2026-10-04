"""Read completed validation scores for progress reporting; never select or train."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha
from scripts.eval_evidence_reranker import RUN,check_contract,specs,partitions
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.evidence_ranking import evidence_metrics,summarize_evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);args=p.parse_args()
    plan=check_contract();folder=RUN/'valid_scores'/args.model
    completion=json.loads((folder/'completion.json').read_text(encoding='utf8'))
    assert completion['status']=='completed'
    result={}
    for co,qfile,_,_ in specs('valid'):
        assert sha(folder/f'{co}.jsonl')==completion['score_files_sha256'][f'{co}.jsonl']
        questions=read_jsonl(ROOT/qfile);raw=read_jsonl(RUN/'valid_candidates'/f'{co}.jsonl');scores=read_jsonl(folder/f'{co}.jsonl')
        metrics={w:[] for w in plan['cross_weights']}
        for q,r,s in zip(questions,raw,scores):
            assert q['id']==r['id']==s['id'] and r['candidate_ids']==s['candidate_ids']
            for w in metrics:
                order=rank_row({**r,'cross':s['cross']},{'pool':'union200','lexical_weight':.2,'cross_weight':w})
                metrics[w].append(evidence_metrics(order,q['gold_answer_ids'],r['candidate_ids']))
        for part,ix in partitions(co,questions).items():
            result[f'{co}/{part}']={str(w):{k:v for k,v in summarize_evidence([rows[i] for i in ix]).items()
                                          if k in ['n','hit_at_1','hit_at_10','mrr_at_100','all_evidence_at_5']}
                                   for w,rows in metrics.items()}
    print(json.dumps({'model':args.model,'validation_only':True,'selection_changed':False,'metrics':result},indent=2))
