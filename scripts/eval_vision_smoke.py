#!/usr/bin/env python3
"""Ad hoc visual transport/reading check on a public PDF; not a vision benchmark."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import requests
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.vlm import VLMClient

URL='https://insurance.delaware.gov/wp-content/uploads/sites/15/2022/09/Auto-Insurance-Guide.pdf'
CASES=[
    {'id':'visual_bodily_individual','page':5,'question':'On this page, what is the maximum bodily-injury amount paid to one person in a single accident under minimum coverage?','expected_number':'25000'},
    {'id':'visual_bodily_aggregate','page':5,'question':'On this page, what is the maximum bodily-injury amount paid to all people in one accident under minimum coverage?','expected_number':'50000'},
    {'id':'visual_pip_funeral','page':5,'question':'What maximum funeral-expense amount does this page include in PIP coverage?','expected_number':'5000'},
    {'id':'visual_missing_personal_premium','page':5,'question':'What is the exact annual premium on my personal auto insurance policy?','expected_abstain':True},
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',default='http://127.0.0.1:11435')
    p.add_argument('--model',default='qwen3.5:4b')
    p.add_argument('--pdf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--structured-output',action='store_true',help='Enforce the JSON schema through Ollama format')
    args=p.parse_args()
    if (args.output/'results.json').exists():raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    doc=pymupdf.open(args.pdf)
    client=VLMClient('ollama:'+args.model,ollama_base_url=args.endpoint,
        generation_options={'seed':42,'temperature':0,'num_ctx':8192,'num_predict':192},request_timeout=600)
    results=[]
    for case in CASES:
        image_path=args.output/f'page_{case["page"]}.png'
        if not image_path.exists():doc[case['page']-1].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(image_path)
        prompt=('Read only the attached archived public-guide page. A public guide does not establish my personal policy. '
          'Return a JSON object with answer (short string) and abstain (boolean). '
          'When the page cannot establish the answer, set abstain=true and answer="". '
          'No explanation or markdown. Question: '+case['question'])
        start=time.perf_counter()
        schema={'type':'object','properties':{'answer':{'type':'string'},'abstain':{'type':'boolean'}},
                'required':['answer','abstain'],'additionalProperties':False}
        answer=client.generate_with_images(prompt,[image_path],response_format=schema if args.structured_output else None)
        try:parsed=json.loads(answer)
        except ValueError:parsed={}
        if case.get('expected_abstain'):
            passed=parsed.get('abstain') is True and parsed.get('answer')==''
        else:
            numbers=re.findall(r'(?<![\w.])\d+(?:\.\d+)?',re.sub(r'(?<=\d),(?=\d)','',str(parsed.get('answer',''))))
            passed=parsed.get('abstain') is False and case['expected_number'] in numbers
        results.append({'case':case,'prompt':prompt,'raw_response':answer,'expected_key_pass':passed,
            'wall_seconds':time.perf_counter()-start,'generation':client.last_generation_metadata})
        print(json.dumps(results[-1]),flush=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'role':'ad_hoc_visual_transport_smoke_not_heldout_accuracy',
        'structured_output':args.structured_output,
        'source_url':URL,'pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        'model':client.backend_metadata(),'results':results,'passed':sum(r['expected_key_pass'] for r in results),'total':len(results),
        'limitations':['Four author-selected questions on one digitally generated public page.',
            'No OCR/scanning robustness, diverse tables, human expert rating, or image-retrieval performance is established.',
            'The normal RAG pipeline still sends text; this is an explicit opt-in image-generation API check.',
            'Expected numeric keys are not semantic validation. Inspect the image and raw answers.']}
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
