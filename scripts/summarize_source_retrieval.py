import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.evidence_evaluation import retrieval_metrics,packing_metrics,aggregate_by_publisher
from scripts.retrieve_source_holdout import LOCAL,read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--arm',required=True);a=p.parse_args()
    if a.split=='test' and not (ROOT/'reports/source_holdout_v1/selection.lock.json').exists():raise ValueError('Test sealed')
    cases=read(LOCAL/'sealed'/(a.split+'.json'));lookup={r['id']:r for r in cases}
    answers={r['id']:r['text'] for r in read(LOCAL/'sealed/answers.json')}
    run=LOCAL/(a.split+'_'+a.arm);completion=read(run/'completion.json')
    if sha(run/'retrieval.jsonl')!=completion['retrieval_sha256']:raise ValueError('Incomplete retrieval')
    rows=[];times=[]
    for line in (run/'retrieval.jsonl').read_text(encoding='utf8').splitlines():
        r=json.loads(line);case=lookup[r['id']]
        rows.append({'id':r['id'],'publisher':case['publisher'],
                     **retrieval_metrics(r['order'],case['gold_answer_ids']),
                     **packing_metrics(r['order'],{g:answers[g] for g in case['gold_answer_ids']},r['context'])})
        times.append(r['retrieval_seconds']+r['amortized_query_encode_seconds'])
    import numpy as np
    keys=[k for k in rows[0] if k not in ['id','publisher','gold_chunks']]
    result=aggregate_by_publisher(rows,list(lookup),keys)
    result.update(arm=a.arm,split=a.split,multi_gold_questions=sum(len(c['gold_answer_ids'])>1 for c in cases),
                  latency_seconds={'p50':float(np.median(times)),'p95':float(np.quantile(times,.95)),
                                   'definition':'per-query ranking/scoring/packing plus amortized batch query encoding; model/index setup excluded'},
                  denominator_ids_sha256=sha(LOCAL/'sealed'/(a.split+'.json')),
                  retrieval_sha256=sha(run/'retrieval.jsonl'),rows=rows)
    write(run/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows']},indent=2))
if __name__=='__main__':main()
