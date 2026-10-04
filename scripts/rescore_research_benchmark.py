#!/usr/bin/env python3
"""Audit recorded responses with scoring v2; never rerun a model or overwrite a run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.eval_research_benchmark import (
    SCORE_VERSION, context_from_prompt, read_jsonl, score_response, sha,
    summarize, verify_manifest, write_json,
)


def rescore_rows(rows, items, pages):
    by_id = {item['id']:item for item in items}
    by_source = {page['citation']:page for page in pages}
    output, seen = [], set()
    for original in rows:
        row = dict(original)
        identity = (row['id'],row['mode'])
        if identity in seen:
            raise ValueError('Duplicate request identity: '+str(identity))
        seen.add(identity)
        item = by_id[row['id']]
        for field in ['question','answerable','document_scope','split']:
            if row.get(field)!=item[field]:
                raise ValueError('Recorded request differs from frozen annotation: '+row['id']+'/'+field)
        generation = row.get('generation') or {}
        complete = (not row.get('error') and generation.get('done') is True
            and generation.get('done_reason')!='length' and not generation.get('truncated',False))
        row['previous_scores'] = row['scores']
        row['previous_score_version'] = row['scores'].get('score_version','v1_corpus_quote')
        row['scores'] = score_response(row.get('raw_response',''),item,by_source,row.get('ranked_sources',[]),
            observed_context=context_from_prompt(row.get('prompt')),generation_complete=complete)
        if row['mode']=='oracle':
            row['scores']['retrieval_hit'] = None
        output.append(row)
    return output


def main(args):
    source = args.run.resolve()
    output = args.output.resolve()
    if output==source or output.exists():
        raise FileExistsError('Choose a new output directory; original results are immutable')
    metadata_path = source/'run_metadata.json'
    prediction_path = source/'predictions.jsonl'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    rows = read_jsonl(prediction_path)
    if not metadata.get('finished_utc') or metadata.get('completed_requests')!=len(rows):
        raise ValueError('Only completed runs with an exact recorded request count may be rescored')
    if metadata.get('predictions_sha256')!=sha(prediction_path):
        raise ValueError('Recorded predictions hash does not match original run metadata')
    _, items, pages = verify_manifest(args.benchmark,args.corpus)
    lock_sha = sha(args.benchmark/'manifest.lock.json')
    if metadata.get('benchmark_lock_sha256')!=lock_sha:
        raise ValueError('Original run used a different frozen benchmark')
    rescored = rescore_rows(rows,items,pages)
    output.mkdir(parents=True,exist_ok=False)
    target = output/'predictions.jsonl'
    target.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rescored),encoding='utf-8')
    write_json(summarize(rescored),output/'summary.json')
    report = {
        'role':'posthoc_scoring_audit_without_new_inference','score_version':SCORE_VERSION,
        'primary_answer_metric':'context_grounded_key_pass',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_run':str(source),'source_predictions_sha256':sha(prediction_path),
        'source_run_metadata_sha256':sha(metadata_path),'benchmark_lock_sha256':lock_sha,
        'predictions_sha256':sha(target),'requests':len(rescored),
        'code_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/'scripts/eval_research_benchmark.py']},
        'changed_historical_grounded_key_pass':sum(bool(row['previous_scores'].get('grounded_key_pass'))!=bool(row['scores']['grounded_key_pass']) for row in rescored),
        'historical_passes_without_context_support':sum(bool(row['scores']['grounded_key_pass']) and not row['scores']['context_grounded_key_pass'] for row in rescored),
        'limitations':[
            'This is a posthoc audit of the exact recorded prompt and response, not an independent model rerun.',
            'Historical grounded_key_pass is retained; compare context_grounded_key_pass only across scoring-v2 runs.',
            'Missing prompt or completion metadata fails the new primary metric closed.',
            'String containment and answer-key checks are not semantic entailment or production error rates.',
        ],
    }
    write_json(report,output/'rescore_metadata.json')
    print(json.dumps(report,indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Completed immutable original run directory')
    p.add_argument('--output',type=Path,required=True,help='New audit directory; must not already exist')
    p.add_argument('--benchmark',type=Path,default=ROOT/'data/benchmarks/research_v1')
    p.add_argument('--corpus',type=Path,default=ROOT/'data/04_curated')
    return p


if __name__=='__main__':
    main(parser().parse_args())
