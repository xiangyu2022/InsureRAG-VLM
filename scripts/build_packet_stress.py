#!/usr/bin/env python3
"""Author deterministic synthetic packet fixtures. Never a real-policy benchmark."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/benchmarks/packet_stress_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value,path):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def build_spec():
    rng=random.Random(42)
    amounts=rng.sample(list(range(1100,9900,100)),12)
    a_ded,b_collision,b_comp,d_base,d_new,e_one,e_two,e_ded1,e_ded2,f_camera,f_laptop,f_jewelry=amounts
    a_limit,b_limit,c_limit=535000,85000,265000
    packets=[];cases=[]
    def packet(letter,description,documents):
        row={'packet_id':'SCENARIO-'+letter,'description':description,'documents':documents}
        packets.append(row)
        return row
    def doc(name,title,pages,role='base_policy',**meta):
        return {'filename':name+'.pdf','title':title,'pages':pages,'metadata':{
            'document_role':role,'source_origin':'synthetic_stress_fixture',
            'source_name':name+'.pdf','source_authority':'Invented regression fixture; no legal authority',**meta}}
    def case(letter,n,question,reference,keys,sources,category,unsupported_reason=None,forbidden_keys=None):
        cases.append({'id':f'{letter}{n:02}','packet_id':'SCENARIO-'+letter,'question':question,
            'answerable':bool(keys),'reference_answer':reference,'answer_keys':keys,'gold_sources':sources,
            'category':category,'unsupported_reason':unsupported_reason,
            'forbidden_keys':forbidden_keys or [],'author':'AI-authored synthetic regression; not expert adjudicated'})
    def money(value):return f'${value:,}'
    packet('A','Multi-coverage declarations with separate values.',[
        doc('SCENARIO-A','Synthetic property declarations',[
            f'Policy identifier SYN-A-001. Coverage A dwelling limit: {money(a_limit)}. Coverage C personal property limit: {money(b_limit)}. These are separate coverage limits, not a combined limit.',
            f'For policy SYN-A-001, the property damage deductible is {money(a_ded)} per loss. The personal liability limit is $305,000 per occurrence. The liability limit is not the property deductible.'
        ],'declarations',policy_number='SYN-A-001')])
    case('A',1,'For policy SYN-A-001, what is the Coverage A dwelling limit?',money(a_limit),[[str(a_limit)]],['SCENARIO-A.pdf#page=1'],'numeric_coverage')
    case('A',2,'For policy SYN-A-001, what is the Coverage C personal property limit?',money(b_limit),[[str(b_limit)]],['SCENARIO-A.pdf#page=1'],'numeric_coverage')
    case('A',3,'For policy SYN-A-001, what is the property damage deductible per loss?',money(a_ded),[[str(a_ded)]],['SCENARIO-A.pdf#page=2'],'numeric_coverage')
    case('A',4,'For policy SYN-A-001, what is the personal liability limit per occurrence?','$305,000',[['305000']],['SCENARIO-A.pdf#page=2'],'numeric_coverage')
    packet('B','Actual declarations alongside a clearly labeled educational example.',[
        doc('SCENARIO-B-DECLARATIONS','Synthetic auto declarations',[
            f'Policy identifier SYN-B-001. Collision deductible: {money(b_collision)}. Comprehensive deductible: {money(b_comp)}. These are the actual deductible amounts declared in this invented policy. The annual premium is not stated in this packet.'
        ],'declarations',policy_number='SYN-B-001'),
        doc('SCENARIO-B-GUIDE','Educational worked example only',[
            'This page is an educational example and is not a declaration for SYN-B-001. In this example only, collision repair costs are $2,900 and the collision deductible is $500, so the example insurer payment is $2,400. None of these example amounts establish the terms of any issued policy.'
        ],'guide')])
    case('B',1,'What collision deductible is actually declared for policy SYN-B-001, rather than the guide example?',money(b_collision),[[str(b_collision)]],['SCENARIO-B-DECLARATIONS.pdf#page=1'],'example_vs_policy')
    case('B',2,'What comprehensive deductible is actually declared for policy SYN-B-001?',money(b_comp),[[str(b_comp)]],['SCENARIO-B-DECLARATIONS.pdf#page=1'],'example_vs_policy')
    case('B',3,'In the educational collision example only, what is the example deductible?','$500',[['500']],['SCENARIO-B-GUIDE.pdf#page=1'],'explicit_example')
    case('B',4,'What exact annual premium is payable for policy SYN-B-001?','',[],[],'missing_value','No annual premium is provided.')
    packet('C','Declarations and a claim report whose policy identifiers do not match.',[
        doc('SCENARIO-C-DECLARATIONS','Synthetic property declarations',[
            f'Policy identifier SYN-C-001. Dwelling limit: {money(c_limit)}. This declarations page supplies no policy effective or expiration date. It does not list an approved settlement amount for any claim.'
        ],'declarations',policy_number='SYN-C-001'),
        doc('SCENARIO-C-CLAIM','Unmatched synthetic claim report',[
            'Claim identifier SYN-CLAIM-OTHER. Policy identifier on this claim report: SYN-OTHER-009. Reported cause of loss: hail. Reported repair estimate: $8,600. No coverage acceptance or approved insurer payment is recorded. This report does not identify policy SYN-C-001.'
        ],'claim_form',policy_number='SYN-OTHER-009')])
    case('C',1,'What dwelling limit is declared for policy SYN-C-001?',money(c_limit),[[str(c_limit)]],['SCENARIO-C-DECLARATIONS.pdf#page=1'],'matched_identifier')
    case('C',2,'What policy identifier is printed on claim report SYN-CLAIM-OTHER?','SYN-OTHER-009',[['SYN-OTHER-009']],['SCENARIO-C-CLAIM.pdf#page=1'],'claim_identifier')
    case('C',3,'What approved insurer payment does this packet establish for claim SYN-CLAIM-OTHER under policy SYN-C-001?','',[],[],'mismatched_claim','Claim references a different policy and no payment is approved.')
    case('C',4,'What is the exact policy expiration date for SYN-C-001?','',[],[],'missing_value','No policy period is supplied.')
    packet('D','An explicit dated endorsement that replaces one base-policy limit.',[
        doc('SCENARIO-D-BASE','Synthetic base policy',[
            f'Policy identifier SYN-D-001. Base form effective January 1, 2026. Section WB: Water backup coverage limit is {money(d_base)}. Later issued endorsements explicitly replacing Section WB control that limit. Earthquake coverage is not described on this page.'
        ],'base_policy',policy_number='SYN-D-001',effective_date='2026-01-01'),
        doc('SCENARIO-D-ENDORSEMENT','Synthetic issued endorsement WB-01',[
            f'Policy identifier SYN-D-001. Endorsement WB-01 effective April 1, 2026. This issued endorsement replaces the entire Section WB water backup limit in the January 1, 2026 base form. The water backup coverage limit is now {money(d_new)}. No other coverage limit is established by this endorsement.'
        ],'endorsement',policy_number='SYN-D-001',effective_date='2026-04-01',form_code='WB-01')])
    case('D',1,'For policy SYN-D-001 after April 1, 2026, what water backup limit is stated in the issued WB-01 endorsement?',money(d_new),[[str(d_new)]],['SCENARIO-D-ENDORSEMENT.pdf#page=1'],'explicit_endorsement')
    case('D',2,'What water backup limit was stated in the January 1, 2026 base form for SYN-D-001, before the endorsement?',money(d_base),[[str(d_base)]],['SCENARIO-D-BASE.pdf#page=1'],'base_version')
    case('D',3,'What effective date is stated on endorsement WB-01 for policy SYN-D-001?','April 1, 2026',[['April 1 2026','2026-04-01']],['SCENARIO-D-ENDORSEMENT.pdf#page=1'],'endorsement_date')
    case('D',4,'What earthquake coverage dollar limit is established for policy SYN-D-001?','',[],[],'missing_coverage','Neither document establishes an earthquake limit.')
    packet('E','Conflicting unsequenced versions without dates or precedence.',[
        doc('SCENARIO-E-VERSION-A','Synthetic unsequenced version A',[
            f'Policy identifier SYN-E-001. Version label A. Equipment coverage limit: {money(e_one)}. Equipment deductible: {money(e_ded1)}. This copy has no effective date, issue date, signature, or precedence statement. The packet also includes a conflicting version B; which version controls is not established.'
        ],'base_policy',policy_number='SYN-E-001',version_id='A'),
        doc('SCENARIO-E-VERSION-B','Synthetic unsequenced version B',[
            f'Policy identifier SYN-E-001. Version label B. Equipment coverage limit: {money(e_two)}. Equipment deductible: {money(e_ded2)}. This copy has no effective date, issue date, signature, or precedence statement. The packet also includes a conflicting version A; which version controls is not established.'
        ],'base_policy',policy_number='SYN-E-001',version_id='B')])
    case('E',1,'Which single equipment coverage limit currently controls policy SYN-E-001 across the conflicting unsequenced versions?','',[],[],'unresolved_conflict','Conflicting limits with no effective date or precedence.')
    case('E',2,'Which single equipment deductible currently controls policy SYN-E-001 across the conflicting unsequenced versions?','',[],[],'unresolved_conflict','Conflicting deductibles with no effective date or precedence.')
    case('E',3,'Which version, A or B, was issued later for policy SYN-E-001?','',[],[],'missing_precedence','Neither version supplies an issue date or ordering rule.')
    case('E',4,'Without deciding which version controls, what equipment coverage limit is printed in version A of SYN-E-001?',money(e_one),[[str(e_one)]],['SCENARIO-E-VERSION-A.pdf#page=1'],'explicit_version_fact')
    packet('F','Near-duplicate scheduled property pages with different item limits.',[
        doc('SCENARIO-F-SCHEDULE','Synthetic scheduled property',[
            f'Policy identifier SYN-F-001. Schedule item CAMERA. The scheduled camera coverage limit is {money(f_camera)}. This limit applies only to the camera, not to laptop or jewelry. No blanket limit for all property is stated.',
            f'Policy identifier SYN-F-001. Schedule item LAPTOP. The scheduled laptop coverage limit is {money(f_laptop)}. This limit applies only to the laptop, not to camera or jewelry. No blanket limit for all property is stated.',
            f'Policy identifier SYN-F-001. Schedule item JEWELRY. The scheduled jewelry coverage limit is {money(f_jewelry)}. This limit applies only to jewelry, not to camera or laptop. No blanket limit for all property is stated.'
        ],'schedule',policy_number='SYN-F-001')])
    for n,item,value in [(1,'camera',f_camera),(2,'laptop',f_laptop),(3,'jewelry',f_jewelry)]:
        case('F',n,f'For SYN-F-001, what is the scheduled {item} coverage limit?',money(value),[[str(value)]],[f'SCENARIO-F-SCHEDULE.pdf#page={n}'],'near_duplicate_citation')
    case('F',4,'What single blanket coverage limit for all property is stated for SYN-F-001?','',[],[],'missing_aggregate','Individual item limits do not establish an overall blanket limit.')
    return {'version':'packet_stress_v1','seed':42,'status':'AI-authored synthetic regression, not held-out policy accuracy',
        'numeric_values':amounts,'packets':packets},cases


def materialize_pdfs(spec,output):
    """Use actual PDFs through the same local-file loader as user uploads."""
    import fitz
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for packet in spec['packets']:
        folder=output/packet['packet_id'];folder.mkdir(parents=True,exist_ok=True)
        entries=[]
        for document in packet['documents']:
            pdf=fitz.open()
            for number,body in enumerate(document['pages'],1):
                page=pdf.new_page(width=612,height=792)
                page.insert_text((45,42),'SYNTHETIC REGRESSION FIXTURE - NO LEGAL FORCE',fontsize=10,color=(.6,.15,.15))
                page.insert_text((45,70),document['title'],fontsize=14)
                remaining=page.insert_textbox(fitz.Rect(45,100,567,710),body,fontsize=12,lineheight=1.5)
                if remaining<0:raise ValueError('Fixture PDF overflow: '+document['filename'])
                page.insert_text((45,751),f'{document["filename"]} - physical page {number}',fontsize=9)
            path=folder/document['filename']
            pdf.set_metadata({'title':document['title'],'author':'AI-authored synthetic diagnostic'})
            pdf.save(path,garbage=4,deflate=True,no_new_id=True);pdf.close()
            hashes[str(path.relative_to(output)).replace('\\','/')]=sha(path)
            entries.append({'path':document['filename'],'packet_id':packet['packet_id'],**document['metadata']})
        write_json({'documents':entries},folder/'packet_manifest.json')
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if (args.output/'manifest.lock.json').exists():raise FileExistsError('Frozen fixture exists; create a new version instead')
    args.output.mkdir(parents=True,exist_ok=True)
    spec,cases=build_spec()
    write_json(spec,args.output/'packets.json')
    (args.output/'cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
    write_json({'version':'packet_stress_v1','seed':42,'frozen_before_model_inference':True,
        'files':{name:sha(args.output/name) for name in ['packets.json','cases.jsonl']},
        'label':'Synthetic regression only; no real customer documents; no production or held-out accuracy claim'},args.output/'manifest.lock.json')
    print(json.dumps({'cases':len(cases),'answerable':sum(c['answerable'] for c in cases),'packets':len(spec['packets'])}))


if __name__=='__main__':main()
