#!/usr/bin/env python3
"""Image-only local VLM diagnostic; question text never contains page transcription."""
import argparse
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse
import requests

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.eval_research_benchmark import answer_key_match,sha,read_jsonl,write_json,wilson,runtime_environment
from src.insurerag_vlm.vlm import VLMClient

SYSTEM=('Read only the attached image. Return JSON with exactly answer (short string), evidence '
        '(a short transcription of the image text supporting that answer), and abstain (boolean). '
        'If the requested fact is absent or unreadable, set abstain=true and both strings empty. '
        'Otherwise set abstain=false. Do not infer policy terms from general insurance knowledge.')
SCHEMA={'type':'object','properties':{'answer':{'type':'string'},'evidence':{'type':'string'},'abstain':{'type':'boolean'}},
        'required':['answer','evidence','abstain'],'additionalProperties':False}
SCORE_VERSION='visual-counterfactual-v1-exact-keys'


def verify_fixture(folder):
    lock=json.loads((folder/'manifest.lock.json').read_text(encoding='utf-8'))
    for name,digest in lock['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Frozen visual fixture changed: '+name)
    cases=read_jsonl(folder/'cases.jsonl')
    if len(cases)!=12 or sum(c['answerable'] for c in cases)!=9:raise ValueError('Unexpected diagnostic composition')
    for field in ['dwelling','deductible','exclusion','premium']:
        group=[c for c in cases if c['field']==field]
        if len(group)!=3 or len({c['question'] for c in group})!=1:raise ValueError('Questions differ across image variants')
        if field!='premium' and len({c['reference_answer'] for c in group})!=3:raise ValueError('Counterfactual targets must differ')
    return lock,cases


def model_request(client,case,folder):
    # Deliberately pass only the exact question and PNG path. No reference answer,
    # label, extracted PDF text, design values, or OCR content enters the request.
    return client.generate_with_images(case['question'],[folder/case['image']],system=SYSTEM,response_format=SCHEMA)


def score_response(case,raw,generation):
    scores={'valid_json':False,'answer_key_match':False,'evidence_key_match':False,
        'strict_abstention':False,'completed_generation':False,'key_pass':False,'evidence_key_pass':False}
    try:parsed=json.loads(raw)
    except (TypeError,json.JSONDecodeError):return scores
    if not isinstance(parsed,dict) or set(parsed)!={'answer','evidence','abstain'} or not isinstance(parsed['answer'],str) or not isinstance(parsed['evidence'],str) or not isinstance(parsed['abstain'],bool):return scores
    completed=generation.get('done_reason')=='stop' and not generation.get('truncated',False)
    scores.update({'valid_json':True,'parsed':parsed,'completed_generation':completed})
    if case['answerable'] and not parsed['abstain']:
        scores['answer_key_match']=answer_key_match(parsed['answer'],case['answer_keys'])
        scores['evidence_key_match']=answer_key_match(parsed['evidence'],case['evidence_keys'])
        scores['key_pass']=scores['answer_key_match'] and completed
        scores['evidence_key_pass']=scores['key_pass'] and scores['evidence_key_match']
    if not case['answerable']:
        scores['strict_abstention']=parsed['abstain'] and not parsed['answer'].strip() and not parsed['evidence'].strip() and completed
    return scores


def summarize(rows):
    supported=[r for r in rows if r['answerable']];unsupported=[r for r in rows if not r['answerable']]
    result={'label':'Synthetic image-only counterfactual diagnostic, not general visual accuracy',
        'score_version':SCORE_VERSION,'cases':len(rows),'runtime_errors':sum(bool(r.get('error')) for r in rows),
        'supported_key_pass':wilson(sum(r['scores']['key_pass'] for r in supported),len(supported)),
        'supported_evidence_key_pass':wilson(sum(r['scores']['evidence_key_pass'] for r in supported),len(supported)),
        'strict_unsupported_abstention':wilson(sum(r['scores']['strict_abstention'] for r in unsupported),len(unsupported)),
        'counterfactual_triplets':{}}
    for field in ['dwelling','deductible','exclusion']:
        group=[r for r in supported if r['field']==field]
        result['counterfactual_triplets'][field]={'variants_observed':len(group),
            'all_three_changed_targets_correct':len(group)==3 and all(r['scores']['key_pass'] for r in group),
            'predicted_answers':{r['variant_id']:r['scores'].get('parsed',{}).get('answer') for r in group}}
    result['limit_note']='Nine answerable and three absent-fact cases share three clean synthetic images; Wilson intervals ignore image clustering and do not establish population reliability.'
    return result


def main(args):
    parsed=urlparse(args.endpoint)
    if parsed.scheme!='http' or parsed.hostname not in {'localhost','127.0.0.1','::1'} or parsed.username or parsed.password:raise ValueError('Explicit local HTTP endpoint required')
    if min(args.num_ctx,args.num_predict,args.timeout)<=0:raise ValueError('Context, output cap, and timeout must be positive')
    lock,cases=verify_fixture(args.benchmark)
    if args.verify_only:
        print(json.dumps({'validated_cases':len(cases),'lock_sha256':sha(args.benchmark/'manifest.lock.json')}));return
    if (args.output/'predictions.jsonl').exists():raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    options={'temperature':0,'seed':42,'num_ctx':args.num_ctx,'num_predict':args.num_predict,
        'top_k':40,'top_p':1.,'repeat_penalty':1.,'presence_penalty':0.}
    client=VLMClient('ollama:'+args.model,provider='ollama',ollama_base_url=args.endpoint,
        generation_options=options,thinking=False,request_timeout=args.timeout,use_hf_api=False,
        hf_api_token=None,openai_api_key=None,anthropic_api_key=None)
    metadata={'started_utc':datetime.now(timezone.utc).isoformat(),'fixture_lock_sha256':sha(args.benchmark/'manifest.lock.json'),
        'fixture':lock,'score_version':SCORE_VERSION,'backend':client.backend_metadata(),
        'system_prompt':SYSTEM,'json_schema':SCHEMA,'generation_options':options,'runtime_environment':runtime_environment(),
        'input_note':'One PNG plus exact question text; no PDF extraction/OCR/transcription/gold values are sent as text.',
        'request_schedule':[c['id'] for c in cases],
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/build_visual_counterfactual.py',ROOT/'scripts/eval_research_benchmark.py',ROOT/'src/insurerag_vlm/vlm.py']}}
    write_json(metadata,args.output/'run_metadata.json');rows=[]
    for case in cases:
        row={key:case[key] for key in ['id','variant_id','field','question','image','answerable','reference_answer']}
        row['image_sha256']=sha(args.benchmark/case['image'])
        row['request_text']={'system':SYSTEM,'user':case['question']}
        row['request_text_sha256']=sha(json.dumps(row['request_text'],sort_keys=True))
        start=time.perf_counter()
        try:
            raw=model_request(client,case,args.benchmark)
            generation=client.last_generation_metadata
            if generation.get('input_modality')!='text+image' or generation.get('image_count')!=1:raise RuntimeError('Image input was not recorded by runtime')
            if [i.get('sha256') for i in generation.get('images',[])]!=[row['image_sha256']]:raise RuntimeError('Runtime image differs from frozen image')
            tags=requests.get(args.endpoint.rstrip('/')+'/api/tags',timeout=10);tags.raise_for_status()
            current=next((m for m in tags.json()['models'] if m.get('name')==args.model or m.get('model')==args.model),None)
            if not current or current.get('digest')!=metadata['backend']['model_digest']:raise RuntimeError('Model digest changed')
            row.update({'raw_response':raw,'generation':generation,'scores':score_response(case,raw,generation)})
        except Exception as exc:
            row.update({'error':type(exc).__name__+': '+str(exc),'scores':score_response(case,'',{})})
        row['wall_seconds']=time.perf_counter()-start;rows.append(row)
        with (args.output/'predictions.jsonl').open('a',encoding='utf-8') as fh:fh.write(json.dumps(row,ensure_ascii=False)+'\n')
        write_json(summarize(rows),args.output/'summary.json')
        print(case['id'],row['scores'],flush=True)
    metadata['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(metadata,args.output/'run_metadata.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,default=ROOT/'data/benchmarks/visual_counterfactual_v1')
    parser.add_argument('--endpoint',default='http://localhost:11435');parser.add_argument('--model',default='qwen3.5:4b')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--num-ctx',type=int,default=8192);parser.add_argument('--num-predict',type=int,default=192)
    parser.add_argument('--timeout',type=float,default=600)
    main(parser.parse_args())
