#!/usr/bin/env python3
"""Audit actual tokenizer supervision without training or loading model weights."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.sft import read_sft_records,format_sft_messages,_render_prompt_and_full_text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/04_curated/sft_dataset.jsonl')
    p.add_argument('--tokenizer',default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--revision',required=True,help='Exact Hugging Face commit hash')
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-lengths',type=int,nargs='+',default=[1024,2048,4096])
    args=p.parse_args()
    if len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision):
        p.error('--revision must be an exact 40-character commit hash')
    from transformers import AutoTokenizer
    import transformers,tokenizers
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,revision=args.revision,
        cache_dir=str(args.cache_dir),trust_remote_code=False)
    records=read_sft_records(args.dataset)
    rows=[]
    question_pages=defaultdict(set)
    for item in records:
        prompt,full=_render_prompt_and_full_text(tokenizer,format_sft_messages(item))
        prompt_ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
        full_ids=tokenizer(full,add_special_tokens=False)['input_ids']
        prefix=full_ids[:len(prompt_ids)]==prompt_ids
        row={'record_id':item.get('record_id'),'answerable':item.get('answerable'),
             'prompt_tokens':len(prompt_ids),'full_tokens':len(full_ids),
             'prompt_is_exact_token_prefix':prefix,
             'assistant_tokens_untruncated':max(0,len(full_ids)-len(prompt_ids))}
        for limit in args.max_lengths:
            row[f'assistant_tokens_at_{limit}']=max(0,min(len(full_ids),limit)-len(prompt_ids))
        rows.append(row)
        if item.get('answerable'):
            question_pages[item['question']].add(item.get('source',''))
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'role':'tokenization_and_dataset_audit_not_training',
        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'tokenizer':args.tokenizer,'tokenizer_revision':args.revision,
        'versions':{'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},
        'records':len(rows),'prompt_prefix_mismatches':sum(not r['prompt_is_exact_token_prefix'] for r in rows),
        'full_token_quantiles':dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile([r['full_tokens'] for r in rows],[.5,.9,.95,.99,1])))),
        'truncation':{str(limit):{'zero_assistant_tokens':sum(r[f'assistant_tokens_at_{limit}']==0 for r in rows),
            'partially_truncated_answers':sum(0<r[f'assistant_tokens_at_{limit}']<r['assistant_tokens_untruncated'] for r in rows),
            'fully_retained_answers':sum(r[f'assistant_tokens_at_{limit}']==r['assistant_tokens_untruncated'] and r['assistant_tokens_untruncated']>0 for r in rows)} for limit in args.max_lengths},
        'distinct_answerable_question_templates':len(question_pages),
        'templates_with_multiple_gold_pages':sum(len(sources)>1 for sources in question_pages.values()),
        'top_question_templates':Counter(r['question'] for r in records if r.get('answerable')).most_common(5),
        'limitations':['This checks tokenization, not training convergence or model answer quality.',
          'Repeated generic questions are not uniquely grounded retrieval queries across the full corpus.',
          'The current full curated dataset is not the historical 3231-row retrieval-conditioned training artifact.',
          'No new Qwen3.5 fine-tuning is asserted.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
