"""Candidate 1: reuse identical scored candidates, change only fusion to pure cross."""
import json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,HISTORICAL,read,write,sha

def main():
    from scripts.eval_insuranceqa_reranker import rank_row
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    source=LOCAL/'dev_baseline';out=LOCAL/'dev_cross_only';out.mkdir(exist_ok=False)
    completion=read(source/'completion.json')
    if sha(source/'retrieval.jsonl')!=completion['retrieval_sha256']:raise ValueError('Baseline changed')
    answers={r['id']:r for r in read(LOCAL/'sealed/answers.json')}
    with (HISTORICAL/'data/training/evidence_reranker_v1/answers.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);answers[r['id']]=r
    config={'pool':'union200','lexical_weight':.2,'cross_weight':1.0}
    protocol={**read(source/'protocol.json'),'arm':'cross_only','rank_config':config,
              'replay_utc':datetime.now(timezone.utc).isoformat(),'scoring_reused_from_sha256':sha(source/'retrieval.jsonl'),
              'latency_status':'inherited identical scoring cost plus measured fusion/packing; independently rerun selected candidate before final latency guardrail',
              'script_sha256':sha(Path(__file__))}
    write(out/'protocol.json',protocol)
    pipeline=DocumentRetrievalPipeline(ModelConfig(vlm_model='local-extractive',retrieval_model='local-hashing',max_context_chars=8000,max_page_chars=2400,max_answer_pages=5))
    count=0
    with (out/'retrieval.jsonl').open('w',encoding='utf8') as handle:
        for line in (source/'retrieval.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);t=time.perf_counter();order=rank_row(r['candidate_scores'],config)
            selected=[{'rank':i+1,'answer_id':aid,**{k:v for k,v in answers[aid].items() if k!='id'}} for i,aid in enumerate(order[:10])]
            pages=[{'source':s['answer_id'],'text_snippet':s['text'],'document_type':'public_qa','primary_clause_type':'general',
                    'section_anchor':' | '.join(str(s.get(k,'')) for k in ['source_group','source_url'])} for s in selected[:5]]
            r.update(arm='cross_only',order=order[:10],results=selected,context=pipeline.pack_long_context(pages,5))
            r['replayed_fusion_packing_seconds']=time.perf_counter()-t
            handle.write(json.dumps(r,ensure_ascii=False)+'\n');count+=1
    write(out/'completion.json',{'questions':count,'retrieval_sha256':sha(out/'retrieval.jsonl'),'protocol_sha256':sha(out/'protocol.json')})
    print(json.dumps({'replayed':count}))
if __name__=='__main__':main()
