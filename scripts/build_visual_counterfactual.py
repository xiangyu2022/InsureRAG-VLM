#!/usr/bin/env python3
"""Freeze three same-layout synthetic declarations images and 12 questions."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/benchmarks/visual_counterfactual_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value,path):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def design():
    rng=random.Random(314159)
    dwelling=rng.sample(list(range(350000,951000,1000)),3)
    deductibles=rng.sample(list(range(1150,4951,100)),3)
    exclusions=['Flood','Earthquake','Sewer backup'];rng.shuffle(exclusions)
    variants=[{'id':letter,'dwelling_limit':dwelling[i],'wind_hail_deductible':deductibles[i],
        'excluded_cause':exclusions[i],'personal_property_limit':73000,'personal_liability_limit':300000,
        'pdf':'assets/visual_'+letter+'.pdf','image':'assets/visual_'+letter+'.png'} for i,letter in enumerate(['A','B','C'])]
    questions=[('dwelling','What is the dwelling coverage limit shown on this declarations page?'),
        ('deductible','What is the wind/hail deductible shown on this declarations page?'),
        ('exclusion','Which cause of loss is explicitly excluded on this declarations page?'),
        ('premium','What is the annual premium amount shown on this declarations page?')]
    cases=[]
    for field,question in questions:
        for v in variants:
            value={'dwelling':v['dwelling_limit'],'deductible':v['wind_hail_deductible'],'exclusion':v['excluded_cause'],'premium':''}[field]
            labels={'dwelling':['dwelling','coverage a'],'deductible':['wind','hail'],'exclusion':['exclusion','excluded']}
            cases.append({'id':v['id']+'_'+field,'variant_id':v['id'],'field':field,'question':question,
                'image':v['image'],'answerable':field!='premium','reference_answer':str(value),
                'answer_keys':[[str(value)]] if field!='premium' else [],
                'evidence_keys':[labels[field],[str(value)]] if field!='premium' else [],
                'unsupported_reason':'No annual premium or premium amount appears anywhere on the page.' if field=='premium' else None})
    return {'version':'visual_counterfactual_v1','seed':314159,'render_dpi':144,
        'label':'AI-authored synthetic image-only counterfactual diagnostic; not real-policy visual accuracy',
        'variants':variants,'question_note':'The same four questions are reused verbatim across all three image variants.'},cases


def render_variant(variant,folder,dpi=144):
    import fitz
    document=fitz.open();page=document.new_page(width=612,height=792)
    navy=(.08,.18,.29);gray=(.35,.4,.46)
    page.draw_rect(fitz.Rect(0,0,612,126),color=navy,fill=navy)
    page.insert_text((40,39),'SYNTHETIC DECLARATIONS',fontname='hebo',fontsize=23,color=(1,1,1))
    page.insert_text((40,70),'Image-only software diagnostic',fontsize=14,color=(.85,.9,.95))
    page.insert_text((40,97),'Invented document. No legal force. No customer data.',fontsize=11,color=(.85,.9,.95))
    rows=[('DWELLING COVERAGE LIMIT',f'${variant["dwelling_limit"]:,}'),
          ('PERSONAL PROPERTY LIMIT',f'${variant["personal_property_limit"]:,}'),
          ('PERSONAL LIABILITY LIMIT',f'${variant["personal_liability_limit"]:,}'),
          ('WIND / HAIL DEDUCTIBLE',f'${variant["wind_hail_deductible"]:,}')]
    y=185
    for label,value in rows:
        page.insert_text((42,y),label,fontname='hebo',fontsize=13,color=gray)
        page.insert_text((396,y),value,fontname='hebo',fontsize=20,color=navy)
        page.draw_line((42,y+18),(570,y+18),color=(.85,.87,.9),width=1)
        y+=76
    page.draw_rect(fitz.Rect(40,505,572,609),color=(.65,.24,.18),fill=(1,.96,.94),width=1)
    page.insert_text((57,534),'EXPLICIT EXCLUSION',fontname='hebo',fontsize=14,color=(.55,.16,.12))
    page.insert_text((57,570),variant['excluded_cause'],fontname='hebo',fontsize=24,color=(.4,.12,.1))
    page.insert_text((42,668),'Only the stated limits and deductible are shown on this page.',fontsize=12,color=gray)
    page.insert_text((42,746),'Visual fixture '+variant['id']+'  |  Physical page 1',fontsize=10,color=gray)
    document.set_metadata({'title':'Synthetic declarations visual '+variant['id'],'author':'AI-authored diagnostic'})
    path=folder/variant['pdf'];path.parent.mkdir(parents=True,exist_ok=True)
    document.save(path,garbage=4,deflate=True,no_new_id=True)
    page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72)).save(folder/variant['image'])
    extracted=page.get_text('text');document.close()
    for value in [f'${variant["dwelling_limit"]:,}',f'${variant["wind_hail_deductible"]:,}',variant['excluded_cause']]:
        if value not in extracted:raise ValueError('Render text missing: '+value)
    if 'premium' in extracted.lower():raise ValueError('Unsupported premium unexpectedly rendered')


def build(folder):
    folder=Path(folder)
    if (folder/'manifest.lock.json').exists():raise FileExistsError('Frozen image fixture exists; create a new version instead')
    folder.mkdir(parents=True,exist_ok=True);spec,cases=design()
    write_json(spec,folder/'design.json')
    (folder/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases),encoding='utf-8')
    for variant in spec['variants']:render_variant(variant,folder,spec['render_dpi'])
    names=['design.json','cases.jsonl']+[v[k] for v in spec['variants'] for k in ['pdf','image']]
    write_json({'version':spec['version'],'frozen_before_model_inference':True,'label':spec['label'],
        'files':{name:sha(folder/name) for name in names},'cases':12,'answerable':9,'unsupported':3},folder/'manifest.lock.json')
    return spec,cases


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();spec,cases=build(args.output)
    print(json.dumps({'variants':len(spec['variants']),'cases':len(cases),'lock_sha256':sha(args.output/'manifest.lock.json')}))
