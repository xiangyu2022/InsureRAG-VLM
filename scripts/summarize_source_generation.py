"""Descriptive answer metrics; lexical/numeric overlap is not semantic accuracy."""
import argparse,json,re,sys
from decimal import Decimal
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha
from src.insurerag_vlm.evidence_evaluation import token_f1,abstention_metrics,citation_markers

def clean_answer(text):return re.split(r'(?i)\bsources?\s*:',text,maxsplit=1)[0].strip()
def numbers(text,version=3):
    if version==2:
        # Preserve the published run's diagnostic implementation for exact replay.
        return {str(Decimal(m.group(1).replace(',','')))+('%' if m.group(2) else '')
                for m in re.finditer(r'(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)(%)?(?![\w.])',text)}
    if version!=3:raise ValueError('Unsupported numeric metric version')
    result=set()
    for match in re.finditer(r'(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)(%)?',text):
        suffix=text[match.end():]
        # Validate the greedy token after matching, so percent and decimal
        # groups cannot backtrack into a misleading partial number.
        if suffix and (suffix[0].isalnum() or suffix[0]=='_' or re.match(r'\.\d',suffix)):continue
        result.add(str(Decimal(match.group(1).replace(',','')))+('%' if match.group(2) else ''))
    return result

def summarize(split,arm,output_name='summary.json',metric_version=3):
    if split=='test' and not (ROOT/'reports/source_holdout_v1/selection.lock.json').exists():raise ValueError('Test sealed')
    cases=read(LOCAL/'sealed'/(split+'.json'));lookup={r['id']:r for r in cases}
    run=LOCAL/(split+'_'+arm)/'generation';completed=read(run/'completion.json')
    if Path(output_name).name!=output_name or not output_name.endswith('.json'):raise ValueError('Output must be a JSON filename')
    if (run/output_name).exists():raise ValueError('Preserve previous summaries; supply a fresh --output-name')
    if sha(run/'answers.jsonl')!=completed['answers_sha256']:raise ValueError('Generation changed')
    rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    if Counter((r['id'],r['cohort']) for r in rows)!=Counter((r['id'],cohort) for r in cases for cohort in ['retrieved','empty_control']):
        raise ValueError('Fixed positive/control denominator mismatch')
    metrics=[];served=[];raw=[]
    for r in rows:
        positive=r['cohort']=='retrieved';s=r.get('served',{});a=clean_answer(r.get('raw_answer',''));ref=lookup[r['id']]['answer']
        markers=citation_markers(r.get('raw_answer',''),version=metric_version);ids=markers['ids']
        known=set(re.findall(r'(?m)^SOURCE:\s*([^\n]+)',r['context']))
        served_content=clean_answer(s.get('answer',''))
        pred_nums=numbers(a,metric_version);ref_nums=numbers(ref,metric_version);sn=numbers(served_content,metric_version)
        row={'id':r['id'],'publisher':lookup[r['id']]['publisher'],'cohort':r['cohort'],'error':r.get('error'),
             'raw_token_f1':token_f1(a,ref) if positive else None,
             'served_token_f1':token_f1(served_content,ref) if positive else None,
             'raw_abstained_heuristic':bool(s.get('explicit_abstention',True)),
             'served_abstained':bool(s.get('abstain',True)),
             'citation_ids':len(ids),'valid_citation_ids':sum(i in known for i in ids),'unknown_citation_ids':[i for i in ids if i not in known],
             'no_source_sentinels':markers['sentinels'],
             'served_citation_ids':len(s.get('citations',[])),
             'served_valid_citation_ids':sum(c.get('source') in known for c in s.get('citations',[])),
             'raw_numeric_count':len(pred_nums),'reference_numeric_count':len(ref_nums),
             'raw_literal_numeric_precision':len(pred_nums & ref_nums)/len(pred_nums) if pred_nums and positive else None,
             'raw_all_literal_numbers_in_reference':float(pred_nums<=ref_nums) if pred_nums and positive else None,
             'served_all_literal_numbers_in_reference':float(sn<=ref_nums) if sn and positive else None,
             'support_reason':s.get('citation_support_reason'),'wall_seconds':r['wall_seconds'],
             'truncated':bool(s.get('generation_truncated')),'repaired':bool(s.get('answer_repaired'))}
        metrics.append(row)
        failed=bool(row['error']) if metric_version>=3 else False
        served.append({'answerable':positive,'abstained':row['served_abstained'],'failed':failed})
        raw.append({'answerable':positive,'abstained':row['raw_abstained_heuristic'],'failed':failed})
    pos=[r for r in metrics if r['cohort']=='retrieved']
    answered=[r for r in pos if not r['served_abstained']]
    numeric=[r for r in pos if r['raw_all_literal_numbers_in_reference'] is not None]
    snumeric=[r for r in pos if r['served_all_literal_numbers_in_reference'] is not None]
    citations=sum(r['citation_ids'] for r in metrics)
    valid=sum(r['valid_citation_ids'] for r in metrics)
    citation_cohorts={}
    for cohort in ['retrieved','empty_control']:
        group=[r for r in metrics if r['cohort']==cohort]
        ids=sum(r['citation_ids'] for r in group);good=sum(r['valid_citation_ids'] for r in group)
        served_ids=sum(r['served_citation_ids'] for r in group);served_good=sum(r['served_valid_citation_ids'] for r in group)
        citation_cohorts[cohort]={'questions':len(group),'raw_emitted_ids':ids,'raw_valid_ids':good,
            'raw_id_precision':good/ids if ids else None,'raw_questions_with_invalid_id':sum(bool(r['unknown_citation_ids']) for r in group),
            'raw_questions_without_source_id':sum(r['citation_ids']==0 for r in group),
            'raw_questions_with_no_source_sentinel':sum(bool(r['no_source_sentinels']) for r in group),
            'served_emitted_ids':served_ids,'served_valid_ids':served_good,'served_id_precision':served_good/served_ids if served_ids else None}
    times=[r['wall_seconds'] for r in pos]
    memory=[];tps=[]
    for r in rows:
        if r.get('resources',{}).get('gpu'):
            fields=r['resources']['gpu'].split(',')
            try:memory.append(float(fields[1]))
            except (ValueError,IndexError):pass
        g=r.get('generation',{}).get('last_generation',{})
        if g.get('eval_duration'):tps.append(g.get('eval_count',0)/(g['eval_duration']/1e9))
    report={'schema_version':metric_version,'metric_correction':'Remove emitted SOURCE fields consistently from both raw and served lexical/numeric metrics; numeric tokens exclude alphanumeric IDs.'+(' Version 3 preserves sentence-final numbers/percentages, does not consume prose after an empty SOURCE line, and never credits failed requests as answers or refusals; published version 2 artifacts remain unchanged.' if metric_version==3 else ''),
            'split':split,'arm':arm,'original_answerable_questions':len(pos),'synthetic_empty_controls':len(rows)-len(pos),
            'errors':sum(bool(r['error']) for r in metrics),'raw_abstention_heuristic':abstention_metrics(raw),
            'served_abstention':abstention_metrics(served),'raw_content_token_f1':float(np.mean([r['raw_token_f1'] for r in pos])),
            'served_content_token_f1_all_answerable':float(np.mean([r['served_token_f1'] for r in pos])),
            'served_content_token_f1_answered_only':float(np.mean([r['served_token_f1'] for r in answered])) if answered else None,
            'raw_numeric_output_questions':len(numeric),'raw_all_literal_numbers_in_reference':float(np.mean([r['raw_all_literal_numbers_in_reference'] for r in numeric])) if numeric else None,
            'served_numeric_output_questions':len(snumeric),'served_all_literal_numbers_in_reference':float(np.mean([r['served_all_literal_numbers_in_reference'] for r in snumeric])) if snumeric else None,
            'raw_citation_id_precision':valid/citations if citations else None,'raw_citation_ids':citations,'raw_valid_citation_ids':valid,
            'citation_cohorts':citation_cohorts,
            'source_semantic_support':'not established by ID validity or token overlap; see separately labelled source review',
            'served_reason_counts':dict(Counter(r['support_reason'] for r in pos)),
            'truncated':sum(r['truncated'] for r in metrics),'repaired':sum(r['repaired'] for r in metrics),
            'generation_wall_seconds':{'p50':float(np.median(times)),'p95':float(np.quantile(times,.95)),
                                       'includes':'provider call and postprocessing; excludes retrieval; first request may include model loading'},
            'whole_gpu_observed_peak_mib':max(memory) if memory else None,'median_output_tokens_per_second':float(np.median(tps)) if tps else None,
            'answers_sha256':sha(run/'answers.jsonl'),'rows':metrics,
            'limitations':['Original questions answerable in publisher source; actual retrieved context may be insufficient.',
                           'Empty-context controls are synthetic and do not measure real-world refusal performance.',
                           'Literal numeric agreement scans entire publisher answer, not expert annotated target quantities.',
                           'Content token F1 is lexical agreement, not semantic or legal correctness.',
                           'Raw abstention is a wording heuristic and requires separate review for false positives.']}
    write(run/output_name,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--arm',required=True)
    p.add_argument('--output-name',default='summary.json');p.add_argument('--metric-version',type=int,choices=[2,3],default=3)
    a=p.parse_args();summarize(a.split,a.arm,a.output_name,a.metric_version)
