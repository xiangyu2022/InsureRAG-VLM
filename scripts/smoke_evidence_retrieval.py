"""Exercise four actual-model retrieval routes after fixed evaluation."""
from datetime import datetime,timezone
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,write_json

if __name__=='__main__':
    out=ROOT/'reports/evidence_reranker_v1/smoke';out.mkdir(exist_ok=False)
    queries=[('insuranceqa','What happens when a term life insurance policy expires?',None),
             ('condition','When can a health plan impose cost sharing on preventive services?',None),
             ('fiqa','How are term life and whole life insurance different?',None),
             ('finqa','What is the interest expense sensitivity to a 100 basis point change in LIBOR?','ADI/2009')]
    results=[]
    for corpus,q,report in queries:
        target=out/f'{corpus}.json';command=[sys.executable,'scripts/query_evidence_reranker.py','--corpus',corpus,'--question',q,'--device','cuda','--output',str(target)]
        if report:command+=['--report',report]
        with (out/f'{corpus}.log').open('w',encoding='utf8') as f:subprocess.run(command,cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)
        result=json.loads(target.read_text(encoding='utf8'))
        assert len(result['results'])==5 and result['generation_calls']==0 and result['query_encoder_passes']==2
        assert 1<=result['reranked_candidates']<=200 and result['annual_report_scope']==report
        results.append({'corpus':corpus,'report':report,'output_sha256':sha(target),'seconds_including_loading':result['seconds_including_loading'],
                        'top_answer_ids':[r['answer_id'] for r in result['results']]})
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','quality_expert_adjudicated':False,
                'runs':results,'query_script_sha256':sha(ROOT/'scripts/query_evidence_reranker.py')},out/'summary.json')
    print(json.dumps(results))
