#!/usr/bin/env python3
"""Frozen, document-scoped research benchmark; local Ollama only.

Uses the actual hybrid pipeline's query_with_ranking path for retrieval + generation.
The diagnostic JSON output contract is independent of the application's answer repair.
Oracle mode intentionally injects gold pages and is never reported as retrieval success.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEM = (
    'Answer only from the provided archived insurance-guide evidence. Do not use outside knowledge. '
    'A general guide or worked example does not identify the questioner\'s own policy terms or private facts. '
    'Return JSON with exactly: answer (short string), evidence (verbatim supporting quote), '
    'source (exact SOURCE identifier), abstain (boolean). If the evidence does not establish the answer, '
    'set abstain=true and all three strings to empty. Otherwise set abstain=false. '
    'Use no extra commentary and never substitute a general example for a person\'s own policy amount.'
)
PROMPT_TEMPLATE = 'Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:'
SCHEMA = {'type':'object','properties':{
    'answer':{'type':'string'}, 'evidence':{'type':'string'}, 'source':{'type':'string'},
    'abstain':{'type':'boolean'}}, 'required':['answer','evidence','source','abstain'],
    'additionalProperties':False}
SCORE_VERSION = 'v2_context_audit'


def sha(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')


def normalize(text):
    text = re.sub(r'(?<=\d),(?=\d)', '', str(text).lower())
    text = text.replace('–','-').replace('—','-')
    return re.sub(r'\s+',' ', re.sub(r'[^a-z0-9.%]+',' ',text)).strip()


def key_present(answer, key):
    value = normalize(key)
    if not value:
        return False
    # Word/numeric boundaries prevent $2,500 from matching $25,000 or $2,500.50.
    if re.fullmatch(r'\d+(?:\.\d+)?', value):
        tokens = re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?!\w|\.\d)',normalize(answer))
        return any(Decimal(token)==Decimal(value) for token in tokens)
    else:
        pattern = r'(?<!\w)' + re.escape(value) + r'(?!\w)'
    return bool(re.search(pattern, normalize(answer)))


def answer_key_match(answer, groups):
    return bool(groups) and all(any(key_present(answer,key) for key in group) for group in groups)


def whitespace(text):
    return re.sub(r'\s+',' ',str(text)).strip()


def wilson(successes, total):
    if not total:
        return {'successes':0,'total':0,'rate':None,'ci95_low':None,'ci95_high':None}
    z = 1.959963984540054
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return {'successes':int(successes),'total':int(total),'rate':p,
            'ci95_low':max(0.,center-radius),'ci95_high':min(1.,center+radius)}


def verify_manifest(folder, corpus):
    lock = json.loads((folder/'manifest.lock.json').read_text(encoding='utf-8'))
    for name, expected in lock['files'].items():
        if sha(folder/name) != expected:
            raise ValueError('Frozen benchmark file changed: '+name)
    for path, expected in lock['corpus_files'].items():
        actual = corpus/Path(path).name
        if sha(actual) != expected:
            raise ValueError('Frozen corpus changed: '+str(actual))
    rows = read_jsonl(folder/'dev.jsonl')+read_jsonl(folder/'test.jsonl')
    pages = read_jsonl(corpus/'rag_pages.jsonl')
    by_source = {row['citation']:row for row in pages}
    scopes = {split:{doc for row in rows if row['split']==split for doc in row['document_scope']} for split in ['dev','test']}
    if scopes['dev'] & scopes['test']:
        raise ValueError('Development/test document overlap')
    for row in rows:
        if row['answerable']:
            if not answer_key_match(row['reference_answer'],row['answer_keys']):
                raise ValueError('Reference answer does not satisfy keys: '+row['id'])
            for gold in row['gold']:
                page = by_source[gold['source']]
                if page['doc_id'] not in row['document_scope'] or gold['evidence_span'] not in page['text']:
                    raise ValueError('Gold span absent from scoped page: '+row['id'])
        elif row['gold'] or row['answer_keys']:
            raise ValueError('Unsupported row has gold evidence: '+row['id'])
    return lock, rows, pages


def scoped_records(pages, snippets, scope):
    """Operational filtering uses document identifiers only; it receives no gold labels."""
    allowed = set(scope)
    return ([row for row in pages if row['doc_id'] in allowed],
            [row for row in snippets if row['doc_id'] in allowed])


def context_from_prompt(prompt):
    """Recover only the evidence section of our recorded prompt contract."""
    if not isinstance(prompt,str) or not prompt.startswith('Context:\n'):
        return None
    context, separator, _ = prompt[len('Context:\n'):].rpartition('\n\nQuestion:\n')
    return context if separator else None


def supplied_source_sections(context):
    """Keep source attribution: a quote from another supplied page cannot count."""
    if not isinstance(context,str):
        return {}
    headers = list(re.finditer(r'^SOURCE:[ \t]*([^\r\n]+)',context,re.MULTILINE))
    sections = {}
    for index, header in enumerate(headers):
        end = headers[index+1].start() if index+1<len(headers) else len(context)
        sections.setdefault(header.group(1).strip(),[]).append(context[header.end():end])
    return sections


def score_response(raw, item, pages_by_source, ranked_sources, *, observed_context=None,
                   generation_complete=False):
    # Historical four-check grounded_key_pass is retained unchanged. The v2
    # primary metric additionally requires evidence actually supplied to a
    # completed generation, rather than merely found elsewhere in the corpus.
    output = {'score_version':SCORE_VERSION,
        'generation_complete':generation_complete is True,
        'valid_json':False, 'abstain':False, 'strict_abstention':False,
        'completed_strict_abstention':False,
        'answer_key_match':False,'citation_match':False,'verbatim_evidence':False,
        'gold_span_covered':False,'grounded_key_pass':False,
        'source_in_context':False,'evidence_in_context':False,'context_grounded_key_pass':False,
        'retrieval_hit':bool(set(ranked_sources) & {gold['source'] for gold in item['gold']}) if item['answerable'] else None}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError,TypeError):
        return output
    if not isinstance(parsed,dict) or set(parsed)!={'answer','evidence','source','abstain'} or not all(isinstance(parsed.get(key),str) for key in ['answer','evidence','source']) or not isinstance(parsed.get('abstain'),bool):
        return output
    output.update({'valid_json':True,'parsed':parsed,'abstain':parsed['abstain']})
    output['strict_abstention'] = parsed['abstain'] and not any(parsed[key].strip() for key in ['answer','evidence','source'])
    output['completed_strict_abstention'] = output['strict_abstention'] and output['generation_complete']
    if item['answerable'] and not parsed['abstain']:
        source, quote = parsed['source'].strip(), whitespace(parsed['evidence'])
        output['answer_key_match'] = answer_key_match(parsed['answer'],item['answer_keys'])
        output['citation_match'] = source in {gold['source'] for gold in item['gold']}
        page = pages_by_source.get(source)
        output['verbatim_evidence'] = bool(page and len(quote)>=12 and quote in whitespace(page['text']))
        output['gold_span_covered'] = any(whitespace(gold['evidence_span']) in quote for gold in item['gold'])
        output['grounded_key_pass'] = output['answer_key_match'] and output['citation_match'] and output['verbatim_evidence'] and output['gold_span_covered']
        sections = supplied_source_sections(observed_context)
        output['source_in_context'] = source in sections
        output['evidence_in_context'] = bool(len(quote)>=12 and any(quote in whitespace(section) for section in sections.get(source,[])))
        output['context_grounded_key_pass'] = (output['grounded_key_pass'] and output['source_in_context']
            and output['evidence_in_context'] and output['generation_complete'])
    return output


class LocalOllamaClient:
    """Explicit loopback transport: no installed-model fallback, cloud API, or paid API."""
    def __init__(self,args):
        parsed = urlparse(args.endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in {'localhost','127.0.0.1','::1'} or parsed.username or parsed.password:
            raise ValueError('Only explicit local HTTP Ollama endpoints are allowed')
        self.endpoint = args.endpoint.rstrip('/')
        self.model = args.model
        self.timeout = args.timeout
        self.options = {'temperature':0,'seed':42,'num_ctx':args.num_ctx,'num_predict':args.num_predict,
                        'top_k':40,'top_p':1.0,'repeat_penalty':1.0,'presence_penalty':0.0}
        self.last_generation_metadata = {}
        self.last_prompt = None
        tags = requests.get(self.endpoint+'/api/tags',timeout=10)
        tags.raise_for_status()
        self.model_info = next((row for row in tags.json()['models'] if row.get('name')==self.model or row.get('model')==self.model),None)
        if not self.model_info or not self.model_info.get('digest'):
            raise RuntimeError('Exact model tag/digest not found: '+self.model)
        version = requests.get(self.endpoint+'/api/version',timeout=10)
        version.raise_for_status()
        self.version = version.json()

    def is_real_llm(self):
        return True

    def backend_label(self):
        return 'Ollama benchmark JSON · '+self.model

    def backend_metadata(self):
        return {'provider':'ollama','requested_model':self.model,'resolved_model':self.model,
                'model_digest':self.model_info['digest'],'model_details':self.model_info.get('details'),
                'ollama_version':self.version,'endpoint':self.endpoint,'generation_options':self.options,
                'thinking':False,'last_generation':self.last_generation_metadata}

    def answer_trace(self,*,invoked,force_extractive=False):
        return {'generation_used':bool(invoked and not force_extractive), 'answer_backend':self.backend_label(),
                'backend_metadata':self.backend_metadata()}

    def generate(self,prompt):
        self._assert_digest_unchanged()
        self.last_prompt = prompt
        self.last_generation_metadata = {}
        payload = {'model':self.model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                   'stream':False,'format':SCHEMA,'think':False,'options':self.options,'keep_alive':'10m'}
        started = time.perf_counter()
        response = requests.post(self.endpoint+'/api/chat',json=payload,timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if data.get('model')!=self.model:
            raise RuntimeError('Ollama response model differs from requested exact tag')
        if data.get('done') is not True:
            raise RuntimeError('Ollama did not complete the response')
        content = data.get('message',{}).get('content')
        if not isinstance(content,str) or not content.strip():
            raise RuntimeError('Ollama returned no answer content')
        self._assert_digest_unchanged()
        self.last_generation_metadata = {key:data.get(key) for key in ['model','done','done_reason','total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']}
        self.last_generation_metadata['truncated'] = data.get('done_reason')=='length'
        self.last_generation_metadata['wall_seconds'] = time.perf_counter()-started
        self.last_generation_metadata['prompt_sha256'] = sha(json.dumps(payload['messages'],sort_keys=True))
        self.last_generation_metadata['returned_thinking_characters'] = len(data.get('message',{}).get('thinking',''))
        # Ollama reports loaded model memory, including actual VRAM allocation.
        try:
            loaded = requests.get(self.endpoint+'/api/ps',timeout=10)
            loaded.raise_for_status()
            self.last_generation_metadata['loaded_runtime'] = next((row for row in loaded.json().get('models',[]) if row.get('name')==self.model or row.get('model')==self.model),None)
        except (requests.RequestException,ValueError):
            self.last_generation_metadata['loaded_runtime'] = None
        return content

    def _assert_digest_unchanged(self):
        response = requests.get(self.endpoint+'/api/tags',timeout=10)
        response.raise_for_status()
        current = next((row for row in response.json().get('models',[]) if row.get('name')==self.model or row.get('model')==self.model),None)
        if not current or current.get('digest')!=self.model_info['digest']:
            raise RuntimeError('Installed model digest changed during benchmark; result rejected')


def runtime_environment():
    packages = {}
    for name in ['numpy','requests','torch','transformers','scikit-learn']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python':platform.python_version(),'platform':platform.platform(),
            'machine':platform.machine(),'processor':platform.processor(),
            'logical_cpu_count':os.cpu_count(),'package_versions':packages,
            'memory_note':'Per-request Ollama api/ps reports model size and size_vram; physical RAM is not inferred.'}


def make_pipeline(args,scope,pages,snippets,client):
    from src.insurerag_vlm.config import ModelConfig
    from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
    key = sha(json.dumps(sorted(scope)))[:12]
    base = args.output/'artifacts'/key
    corpus = base/'curated'
    corpus.mkdir(parents=True,exist_ok=True)
    chosen_pages,chosen_snippets = scoped_records(pages,snippets,scope)
    for name,rows in [('rag_pages.jsonl',chosen_pages),('rag_snippets.jsonl',chosen_snippets)]:
        (corpus/name).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    config = ModelConfig(retrieval_model=args.retrieval_model,vlm_model='local-extractive',
        use_hf_api=False,hf_api_token=None,openai_api_key=None,anthropic_api_key=None,
        retrieval_mode=args.retrieval_mode,corpus_source='curated',enable_image_signal=False,
        curated_dataset_dir=corpus,index_dir=base/'index',max_retrievals=args.top_k,
        max_answer_pages=args.top_k,max_context_chars=args.max_context_chars,
        prompt_template=PROMPT_TEMPLATE)
    pipeline = DocumentRetrievalPipeline(config)
    if client is not None:
        pipeline.vlm_client = client
    pipeline.build_index(corpus)
    return pipeline,corpus,len(chosen_pages)


def oracle_context(item,pages):
    scoped = [row for row in pages if row['doc_id'] in item['document_scope']]
    if item['answerable']:
        wanted = {gold['source'] for gold in item['gold']}
        selected = [row for row in scoped if row['citation'] in wanted]
        context_type = 'full_gold_pages'
    else:
        # Fixed document context without consulting an answer; absence is in the scoped guide,
        # which cannot establish a person's private policy/identity values.
        selected = sorted(scoped,key=lambda row:(row['doc_id'],row['page']))[:3]
        context_type = 'first_three_scoped_pages_unsupported'
    text = '\n---\n'.join('SOURCE: '+row['citation']+'\n'+row['text'] for row in selected)
    return text,[row['citation'] for row in selected],context_type


def summarize(rows):
    if any(row.get('scores',{}).get('score_version')!=SCORE_VERSION for row in rows):
        raise ValueError('Cannot label old or mixed scores as v2; run the separate historical rescore audit first')
    output = {}
    for mode in sorted({row['mode'] for row in rows}):
        group = [row for row in rows if row['mode']==mode]
        supported = [row for row in group if row['answerable']]
        unsupported = [row for row in group if not row['answerable']]
        metrics = {'score_version':SCORE_VERSION,'primary_answer_metric':'context_grounded_key_pass',
            'cases':len(group),'answerable_cases':len(supported),'unsupported_cases':len(unsupported),
            'transport_errors':sum(bool(row.get('error')) for row in group)}
        for key in ['answer_key_match','citation_match','verbatim_evidence','gold_span_covered','grounded_key_pass',
                    'source_in_context','evidence_in_context','context_grounded_key_pass']:
            if mode!='retrieval-only':
                metrics[key] = wilson(sum(bool(row['scores'].get(key)) for row in supported),len(supported))
        if mode in {'retrieved','retrieval-only'}:
            metrics['retrieval_hit_at_k'] = wilson(sum(bool(row['scores'].get('retrieval_hit')) for row in supported),len(supported))
        if mode!='retrieval-only':
            metrics['strict_unsupported_abstention'] = wilson(sum(bool(row['scores'].get('strict_abstention')) for row in unsupported),len(unsupported))
            metrics['completed_strict_unsupported_abstention'] = wilson(sum(bool(row['scores'].get('completed_strict_abstention')) for row in unsupported),len(unsupported))
            metrics['generation_complete'] = wilson(sum(bool(row['scores'].get('generation_complete')) for row in group),len(group))
            metrics['json_contract_success'] = wilson(sum(bool(row['scores'].get('valid_json')) for row in group),len(group))
        durations = [row['wall_seconds'] for row in group]
        metrics['wall_seconds'] = {'p50':float(np.percentile(durations,50)),'p95':float(np.percentile(durations,95)),'total':sum(durations)}
        output[mode] = metrics
    return output


def main(args):
    for name in ['top_k','max_context_chars','num_ctx','num_predict','timeout']:
        if getattr(args,name)<=0:
            raise ValueError(name+' must be positive')
    if args.limit is not None and args.limit<=0:
        raise ValueError('limit must be positive')
    lock,items,pages = verify_manifest(args.benchmark,args.corpus)
    selected = [row for row in items if row['split']==args.split]
    if args.limit:
        if args.split!='dev':
            raise ValueError('Partial test runs are prohibited; use dev for smoke tests')
        selected = selected[:args.limit]
    if args.validate_only:
        print(json.dumps({'status':'validated','selected_cases':len(selected),'lock_sha256':sha(args.benchmark/'manifest.lock.json')},indent=2))
        return
    if args.retrieval_model!='local-hashing' and not Path(args.retrieval_model).is_dir():
        raise ValueError('Only local-hashing or an existing local embedding checkpoint directory is allowed')
    if (args.output/'predictions.jsonl').exists():
        raise FileExistsError('Output already contains predictions; select a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    snippets = read_jsonl(args.corpus/'rag_snippets.jsonl')
    client = None if args.mode=='retrieval-only' else LocalOllamaClient(args)
    modes = ['retrieved','oracle'] if args.mode=='both' else [args.mode]
    pages_by_source = {row['citation']:row for row in pages}
    pipeline_cache = {}
    metadata = {'started_utc':datetime.now(timezone.utc).isoformat(),'benchmark_lock_sha256':sha(args.benchmark/'manifest.lock.json'),
        'benchmark':lock,'split':args.split,'case_count':len(selected),'partial_dev_smoke':bool(args.limit),
        'model':client.backend_metadata() if client else None,
        'run_config':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'prompt_contract_sha256':sha(json.dumps({'system':SYSTEM,'template':PROMPT_TEMPLATE,'schema':SCHEMA},sort_keys=True)),
        'code_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/'src/insurerag_vlm/hybrid_pipeline.py',ROOT/'src/insurerag_vlm/retriever.py',ROOT/'src/insurerag_vlm/config.py']},
        'adapter_status':'No project SFT adapter is loaded; exact local Ollama tag and digest only.',
        'evaluation_path':'hybrid_pipeline.query_with_ranking plus benchmark JSON generation contract; no app repair or query_structured postprocessing.',
        'score_version':SCORE_VERSION,'primary_answer_metric':'context_grounded_key_pass',
        'metric_warning':'Deterministic answer-key/evidence contract checks are not semantic accuracy or a production insurance error rate. Historical grounded_key_pass checks full corpus pages; context_grounded_key_pass additionally checks the exact supplied source section and completed generation. Wilson intervals treat questions as independent and do not account for document clustering.'}
    metadata['runtime_environment'] = runtime_environment()
    metadata['request_schedule'] = [{'id':item['id'],'mode':mode} for item in selected for mode in modes]
    metadata['index_build_seconds'] = {}
    write_json(metadata,args.output/'run_metadata.json')
    results = []
    for item in selected:
        for mode in modes:
            row = {'id':item['id'],'split':item['split'],'mode':mode,'question':item['question'],
                   'answerable':item['answerable'],'document_scope':item['document_scope']}
            started = time.perf_counter()
            ranked_sources = []
            if client:
                client.last_prompt = None
                client.last_generation_metadata = {}
            try:
                if mode in {'retrieved','retrieval-only'}:
                    scope_key = tuple(sorted(item['document_scope']))
                    if scope_key not in pipeline_cache:
                        index_started = time.perf_counter()
                        pipeline_cache[scope_key] = make_pipeline(args,scope_key,pages,snippets,client)
                        metadata['index_build_seconds']['|'.join(scope_key)] = time.perf_counter()-index_started
                    pipeline,corpus,page_count = pipeline_cache[scope_key]
                    # Index construction excluded from query latency and reported separately.
                    started = time.perf_counter()
                    result = pipeline.query_with_ranking(item['question'],corpus,top_k=args.top_k,force_extractive=mode=='retrieval-only')
                    raw = result['answer']
                    ranked_sources = [page['source'] for page in result['source_ranking']]
                    row.update({'ranked_sources':ranked_sources,'scoped_page_count':page_count,
                                'source_ranking':result['source_ranking']})
                else:
                    context,sources,context_type = oracle_context(item,pages)
                    raw = client.generate(PROMPT_TEMPLATE.format(context=context,question=item['question']))
                    row.update({'oracle_sources':sources,'oracle_context_type':context_type})
                row['raw_response'] = raw
                generation = client.last_generation_metadata if client else {}
                row['scores'] = score_response(raw,item,pages_by_source,ranked_sources,
                    observed_context=context_from_prompt(client.last_prompt) if client else None,
                    generation_complete=generation.get('done') is True and generation.get('done_reason')!='length' and not generation.get('truncated',False))
                if mode=='oracle':
                    row['scores']['retrieval_hit'] = None
                if client:
                    row['generation'] = client.last_generation_metadata
                    row['prompt'] = client.last_prompt
                    row['generation_used'] = client.last_prompt is not None
            except Exception as exc:
                row['error'] = type(exc).__name__+': '+str(exc)
                row['scores'] = score_response('',item,pages_by_source,ranked_sources)
                if client:
                    row['generation'] = client.last_generation_metadata
                    row['prompt'] = client.last_prompt
                    row['generation_used'] = client.last_prompt is not None
                if mode=='oracle':
                    row['scores']['retrieval_hit'] = None
            row['wall_seconds'] = time.perf_counter()-started
            results.append(row)
            with (args.output/'predictions.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            write_json(summarize(results),args.output/'summary.json')
            print(json.dumps({'id':row['id'],'mode':mode,'seconds':round(row['wall_seconds'],2),
                              'scores':{key:value for key,value in row['scores'].items() if key!='parsed'},'error':row.get('error')}),flush=True)
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['completed_requests'] = len(results)
    metadata['request_errors'] = sum('error' in row for row in results)
    metadata['predictions_sha256'] = sha(args.output/'predictions.jsonl')
    write_json(metadata,args.output/'run_metadata.json')
    print(json.dumps(summarize(results),indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=ROOT/'data/benchmarks/research_v1')
    p.add_argument('--corpus',type=Path,default=ROOT/'data/04_curated')
    p.add_argument('--endpoint',default='http://localhost:11435')
    p.add_argument('--model',default=None,help='Exact installed Ollama tag; no automatic fallback')
    p.add_argument('--output','--output-dir',dest='output',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--mode',choices=['both','retrieved','oracle','retrieval-only'],default='both')
    p.add_argument('--retrieval-model',default='local-hashing')
    p.add_argument('--retrieval-mode',choices=['hybrid_text','hybrid_multimodal','dense_only'],default='hybrid_text')
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--max-context-chars',type=int,default=8000)
    p.add_argument('--num-ctx',type=int,default=8192)
    p.add_argument('--num-predict',type=int,default=192)
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--limit',type=int,default=None,help='Dev smoke only; prohibited for test')
    p.add_argument('--validate-only',action='store_true')
    return p


if __name__=='__main__':
    args = parser().parse_args()
    if not args.validate_only and args.mode!='retrieval-only' and not args.model:
        raise SystemExit('--model is required for generation')
    main(args)
