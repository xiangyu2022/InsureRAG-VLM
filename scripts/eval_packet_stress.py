#!/usr/bin/env python3
"""Exercise real PDF ingestion and query_structured; synthetic regression only."""
import argparse
import json
import re
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_packet_stress import materialize_pdfs
from scripts.eval_research_benchmark import answer_key_match,key_present,sha,read_jsonl,write_json,wilson,runtime_environment
from src.insurerag_vlm.answer_safety import is_explicit_abstention


def canonical_source(source):
    source=str(source).strip().replace('\\','/').strip('[]<> ')
    match=re.search(r'([^/]+\.pdf)#page=(\d+)',source,re.I)
    return (match.group(1).lower()+'#page='+str(int(match.group(2)))) if match else source.lower()


def raw_sources(answer):
    return [canonical_source(value) for value in re.findall(r'[^\s\[\]<>]*\.pdf#page=\d+',str(answer),re.I)]


def context_sections(prompt):
    context=str(prompt).split('Context:',1)[-1].split('\n\nQuestion:',1)[0]
    matches=list(re.finditer(r'^SOURCE:\s*(.+?)\s*$',context,re.M))
    return {canonical_source(m.group(1)):context[m.end():matches[i+1].start() if i+1<len(matches) else len(context)] for i,m in enumerate(matches)}


def score_case(case,result,prompt=''):
    raw=str(result.get('raw_answer',''))
    final=str(result.get('answer',''))
    expected={canonical_source(s) for s in case['gold_sources']}
    supplied=context_sections(prompt)
    retrieved={canonical_source(p['source']):str(p.get('text_snippet','')) for p in result.get('source_ranking',[])}
    generation=result.get('backend_metadata',{}).get('last_generation',{}) if result.get('backend_metadata') else {}
    completed=bool(result.get('generation_used')) and generation.get('done_reason')!='length' and not generation.get('truncated',False)
    outputs={}
    for name,answer,abstain,sources in [
        ('raw',raw,is_explicit_abstention(raw),raw_sources(raw)),
        ('served',final,bool(result.get('abstain')),[canonical_source(c['source']) for c in result.get('citations',[])])]:
        keymatch=case['answerable'] and not abstain and answer_key_match(answer,case['answer_keys'])
        forbidden=any(key_present(answer,key) for key in case.get('forbidden_keys',[]))
        citation=bool(expected) and expected.issubset(set(sources))
        source_present=bool(sources) and all(s in supplied for s in sources)
        context_keys=bool(case['answerable'] and any(answer_key_match(supplied.get(s,''),case['answer_keys']) for s in sources))
        retrieved_keys=bool(case['answerable'] and any(answer_key_match(retrieved.get(s,''),case['answer_keys']) for s in sources))
        outputs[name]={'answer_key_match':bool(keymatch),'citation_match':citation,
            'sources':sources,'abstain':abstain,'forbidden_key_present':forbidden,
            'supported_key_source_pass':bool(keymatch and citation and not forbidden),
            'context_source_present':source_present,'context_answer_key_present':context_keys,
            'completed_generation':completed,
            'context_key_source_pass':bool(keymatch and citation and not forbidden and source_present and context_keys and completed),
            'retrieved_evidence_key_source_pass':bool(keymatch and citation and not forbidden and retrieved_keys),
            'unsupported_abstention':bool(not case['answerable'] and abstain),
            'empty_served_abstention':bool(name=='served' and not case['answerable'] and abstain and not answer.strip() and not sources)}
    return outputs


def verify_fixture(folder):
    lock=json.loads((folder/'manifest.lock.json').read_text(encoding='utf-8'))
    for name,digest in lock['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Frozen synthetic fixture changed: '+name)
    spec=json.loads((folder/'packets.json').read_text(encoding='utf-8'))
    cases=read_jsonl(folder/'cases.jsonl')
    sources={p['packet_id']:{canonical_source(d['filename']+'#page='+str(n)) for d in p['documents'] for n in range(1,len(d['pages'])+1)} for p in spec['packets']}
    if len(cases)!=24 or len({c['id'] for c in cases})!=24:raise ValueError('Expected 24 unique cases')
    for case in cases:
        if case['answerable'] and (not answer_key_match(case['reference_answer'],case['answer_keys']) or not case['gold_sources']):raise ValueError('Invalid reference: '+case['id'])
        if any(canonical_source(s) not in sources[case['packet_id']] for s in case['gold_sources']):raise ValueError('Unknown gold source: '+case['id'])
        if not case['answerable'] and (case['answer_keys'] or case['gold_sources']):raise ValueError('Unsupported case has gold: '+case['id'])
    return lock,spec,cases


def validate_pdf_ingestion(spec,folder):
    from src.insurerag_vlm.data import load_documents
    details={}
    for packet in spec['packets']:
        docs=load_documents(folder/packet['packet_id'])
        expected={canonical_source(d['filename']+'#page='+str(n)):body for d in packet['documents'] for n,body in enumerate(d['pages'],1)}
        actual={canonical_source(d.metadata['source']):d for d in docs}
        if set(actual)!=set(expected):raise ValueError('PDF page/source mismatch: '+packet['packet_id'])
        for source,body in expected.items():
            if ' '.join(body.split()) not in ' '.join(actual[source].text.split()):raise ValueError('PDF extraction lost text: '+source)
            if actual[source].metadata.get('source_origin')!='synthetic_stress_fixture':raise ValueError('Synthetic provenance lost')
        details[packet['packet_id']]={'loaded_pages':len(docs),'sources':sorted(actual)}
    return details


def summarize(rows):
    supported=[r for r in rows if r['answerable']];unsupported=[r for r in rows if not r['answerable']]
    output={'label':'AI-authored synthetic PDF regression; not real-policy or held-out accuracy',
        'cases':len(rows),'answerable_cases':len(supported),'unsupported_cases':len(unsupported),
        'runtime_errors':sum(bool(r.get('error')) for r in rows),
        'answer_repaired_count':sum(bool(r.get('result',{}).get('answer_repaired')) for r in rows)}
    for stage in ['raw','served']:
        output[stage]={}
        for key in ['answer_key_match','citation_match','supported_key_source_pass','context_key_source_pass','retrieved_evidence_key_source_pass']:
            output[stage][key]=wilson(sum(bool(r['scores'][stage][key]) for r in supported),len(supported))
        output[stage]['unsupported_abstention']=wilson(sum(bool(r['scores'][stage]['unsupported_abstention']) for r in unsupported),len(unsupported))
    if rows:
        durations=[r['wall_seconds'] for r in rows]
        output['wall_seconds']={'p50':float(np.percentile(durations,50)),'p95':float(np.percentile(durations,95)),'total':sum(durations)}
    output['interval_note']='Wilson intervals are descriptive only; synthetic cases share six authored packets and are not a random population sample.'
    return output


def main(args):
    for name in ['top_k','max_context_chars','num_ctx','num_predict','timeout']:
        if getattr(args,name)<=0:raise ValueError(name+' must be positive')
    if args.limit is not None and args.limit<=0:raise ValueError('limit must be positive')
    endpoint=urlparse(args.endpoint)
    if endpoint.scheme!='http' or endpoint.hostname not in {'localhost','127.0.0.1','::1'} or endpoint.username or endpoint.password:raise ValueError('Explicit loopback HTTP endpoint required')
    if args.retrieval_model!='local-hashing' and not Path(args.retrieval_model).is_dir():raise ValueError('Embedding must be local-hashing or an existing local checkpoint')
    lock,spec,cases=verify_fixture(args.benchmark)
    if (args.output/'predictions.jsonl').exists():raise FileExistsError('Use a new output directory; results cannot be overwritten')
    args.output.mkdir(parents=True,exist_ok=True)
    pdf_folder=args.output/'generated_packets'
    pdf_hashes=materialize_pdfs(spec,pdf_folder)
    extraction=validate_pdf_ingestion(spec,pdf_folder)
    if args.validate_only:
        write_json({'lock_sha256':sha(args.benchmark/'manifest.lock.json'),'pdf_sha256':pdf_hashes,'ingestion':extraction},args.output/'validation.json')
        print(json.dumps({'validated_cases':len(cases),'pdfs':len(pdf_hashes),'pages':sum(v['loaded_pages'] for v in extraction.values())}))
        return
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    from src.insurerag_vlm.vlm import _OLLAMA_SYSTEM_PROMPT
    tags=requests.get(args.endpoint.rstrip('/')+'/api/tags',timeout=10);tags.raise_for_status()
    model_info=next((m for m in tags.json()['models'] if m.get('name')==args.model or m.get('model')==args.model),None)
    if not model_info or not model_info.get('digest'):raise ValueError('Exact local model tag/digest not available')
    selected=cases[:args.limit] if args.limit else cases
    options={'temperature':0,'seed':42,'num_ctx':args.num_ctx,'num_predict':args.num_predict,
        'top_k':40,'top_p':1.,'repeat_penalty':1.,'presence_penalty':0.}
    metadata={'started_utc':datetime.now(timezone.utc).isoformat(),'label':lock['label'],
        'benchmark_lock_sha256':sha(args.benchmark/'manifest.lock.json'),'model_info':model_info,
        'run_config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'runtime_environment':runtime_environment(),'pdf_sha256':pdf_hashes,'ingestion':extraction,
        'generation_options':options,'adapter_status':'No SFT adapter; exact Ollama base-model tag only',
        'evaluation_path':'actual PDF load_documents -> index -> query_structured; same-call raw_answer versus served output',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/build_packet_stress.py',ROOT/'src/insurerag_vlm/hybrid_pipeline.py',ROOT/'src/insurerag_vlm/vlm.py',ROOT/'src/insurerag_vlm/answer_safety.py',ROOT/'src/insurerag_vlm/retriever.py',ROOT/'src/insurerag_vlm/data.py',ROOT/'src/insurerag_vlm/config.py']},
        'request_schedule':[c['id'] for c in selected],'index_build_seconds':{}}
    write_json(metadata,args.output/'run_metadata.json')
    pipelines={};rows=[]
    for case in selected:
        packet=case['packet_id'];folder=pdf_folder/packet
        if packet not in pipelines:
            config=ModelConfig(retrieval_model=args.retrieval_model,vlm_model='ollama:'+args.model,vlm_provider='ollama',
                ollama_base_url=args.endpoint,ollama_generation_options=options,vlm_thinking=False,
                vlm_request_timeout=args.timeout,vlm_expected_digest=model_info['digest'],
                use_hf_api=False,hf_api_token=None,openai_api_key=None,anthropic_api_key=None,
                retrieval_mode='hybrid_text',corpus_source='documents',enable_image_signal=False,
                index_dir=args.output/'indices'/packet,max_retrievals=args.top_k,max_answer_pages=args.top_k,
                max_context_chars=args.max_context_chars)
            start=time.perf_counter();pipeline=DocumentRetrievalPipeline(config);pipeline.build_index(folder)
            metadata['index_build_seconds'][packet]=time.perf_counter()-start
            pipelines[packet]=pipeline
            metadata['prompt_contract_sha256']=sha(json.dumps({'system':_OLLAMA_SYSTEM_PROMPT,'template':config.prompt_template},sort_keys=True))
        pipeline=pipelines[packet];client=pipeline.vlm_client
        original=client.generate;captured={}
        def capture(prompt):
            captured['prompt']=prompt
            captured['prompt_sha256']=sha(json.dumps({'system':_OLLAMA_SYSTEM_PROMPT,'user':prompt},sort_keys=True))
            return original(prompt)
        client.generate=capture
        row={'id':case['id'],'packet_id':packet,'question':case['question'],'category':case['category'],
            'answerable':case['answerable'],'reference_answer':case['reference_answer'],'gold_sources':case['gold_sources']}
        start=time.perf_counter()
        try:
            result=pipeline.query_structured(case['question'],folder,top_k=args.top_k)
            post_tags=requests.get(args.endpoint.rstrip('/')+'/api/tags',timeout=10);post_tags.raise_for_status()
            post_model=next((m for m in post_tags.json()['models'] if m.get('name')==args.model or m.get('model')==args.model),None)
            if not post_model or post_model.get('digest')!=model_info['digest']:raise RuntimeError('Model digest changed during request')
            row['result']=result
            row['scores']=score_case(case,result,captured.get('prompt',''))
        except Exception as exc:
            row['error']=type(exc).__name__+': '+str(exc)
            row['scores']=score_case(case,{})
        finally:
            client.generate=original
        row.update(captured);row['wall_seconds']=time.perf_counter()-start
        rows.append(row)
        with (args.output/'predictions.jsonl').open('a',encoding='utf-8') as fh:fh.write(json.dumps(row,ensure_ascii=False)+'\n')
        write_json(summarize(rows),args.output/'summary.json')
        write_json(metadata,args.output/'run_metadata.json')
        print(case['id'],row['scores']['served'],flush=True)
    metadata['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(metadata,args.output/'run_metadata.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,default=ROOT/'data/benchmarks/packet_stress_v1')
    parser.add_argument('--endpoint',default='http://localhost:11435')
    parser.add_argument('--model',default='qwen3.5:4b')
    parser.add_argument('--retrieval-model',default='local-hashing')
    parser.add_argument('--output','--output-dir',dest='output',type=Path,required=True)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--max-context-chars',type=int,default=8000)
    parser.add_argument('--num-ctx',type=int,default=8192)
    parser.add_argument('--num-predict',type=int,default=192)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--validate-only',action='store_true')
    main(parser.parse_args())
