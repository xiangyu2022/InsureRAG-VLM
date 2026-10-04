"""Measure positive-pair input truncation without changing labels or evaluation subsets."""
from collections import defaultdict
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha

def run():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'../models/ms-marco-MiniLM-L6-v2',local_files_only=True,trust_remote_code=False)
    fixture=ROOT/'data/benchmarks/condition_v1';answers={a['id']:a['text'] for a in read_jsonl(fixture/'answers.jsonl')}
    cases=read_jsonl(fixture/'train_additions.jsonl')+read_jsonl(fixture/'test.jsonl');lengths=defaultdict(list);long=[]
    for i in range(0,len(cases),128):
        batch=cases[i:i+128];encoded=tokenizer([c['question'] for c in batch],[answers[c['gold_answer_ids'][0]] for c in batch],truncation=False,return_length=True,verbose=False)
        for c,length in zip(batch,encoded['length']):
            lengths[c['split']+'_'+c['domain']].append(int(length))
            if length>512:long.append({'id':c['id'],'split':c['split'],'domain':c['domain'],'pair_tokens':int(length)})
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'max_sequence_length':512,'policy':'longest_first; no cases removed or relabeled',
        'counts':{k:{'questions':len(v),'over_512':sum(n>512 for n in v),'percent_over_512':100*sum(n>512 for n in v)/len(v),
                     'median_tokens':float(np.median(v)),'p95_tokens':float(np.quantile(v,.95)),'max_tokens':max(v)} for k,v in lengths.items()},
        'long_pairs':long,'fixture_sha256':sha(fixture/'manifest.lock.json'),'code_sha256':sha(Path(__file__)),
        'interpretation':'Input truncation only; does not prove the decisive condition or gold answer span was lost.'}
    write_json(report,ROOT/'reports/condition_v1/length_audit.json');print(json.dumps(report['counts'],indent=2))

if __name__=='__main__':run()
