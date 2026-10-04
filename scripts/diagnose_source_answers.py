"""Report observable validator causes separately from unverified semantics."""
import json,sys,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline as P

def main():
    cases={r['id']:r for r in read(LOCAL/'sealed/dev.json')}
    retrieval={r['id']:r for r in [json.loads(l) for l in (LOCAL/'dev_baseline/retrieval.jsonl').read_text(encoding='utf8').splitlines()]}
    rows=[json.loads(l) for l in (LOCAL/'dev_baseline/generation/answers.jsonl').read_text(encoding='utf8').splitlines() if json.loads(l)['cohort']=='retrieved']
    broad={'coverage','cover','include','policy','insurance','limit','limits','sublimit','deductible','endorsement','reimbursement','provision','amount','liability','property','loss','use','auto','automobile','guide','document','pdf','apply','applies','actual','actually','declared','printed','stated','shown','date','benefit','benefits'}
    output=[]
    for row in rows:
        s=row['served'];case=cases[row['id']];pages=s['source_ranking']
        source=P._extract_answer_source(row['raw_answer'],pages)
        page=next((p for p in pages if p['source']==source),{})
        evidence=P._packed_source_evidence(row['context'],source or '')
        terms={t for t in P._support_terms(row['question'])-P._generic_terms() if len(t)>=4 and t not in broad}
        clean=re.sub(r'(?im)^\s*SOURCE:.*$','',row['raw_answer']).strip()
        m=re.search(r'(?im)^\s*SOURCE:\s*(.*)$',row['raw_answer'])
        identifiers=[t.strip(' .`*[]') for t in m.group(1).split(',')] if m else []
        known={p['source'] for p in pages}
        output.append({'id':row['id'],'publisher':case['publisher'],'question':row['question'],'raw_answer':row['raw_answer'],
                       'reference_answer':case['answer'],'cited_evidence':evidence,'cited_source':source,
                       'source_line_ids':identifiers,'unknown_source_ids':[x for x in identifiers if x not in known and x!='insufficient_evidence'],
                       'served_abstain':s['abstain'],'reason':s['citation_support_reason'],'raw_explicit_abstention':s['explicit_abstention'],
                       'gold_complete_in_top5':set(case['gold_answer_ids'])<=set(retrieval[row['id']]['order'][:5]),
                       'missing_specific_terms':sorted(terms-P._support_terms(evidence)),
                       'semantic_support_status':'not_adjudicated','model_error_status':'not_adjudicated'})
    write(LOCAL/'dev_answer_diagnosis.json',output)
    for r in output:
        if r['served_abstain']:
            print(json.dumps({k:r[k] for k in ['id','publisher','question','raw_answer','reason','gold_complete_in_top5','missing_specific_terms','unknown_source_ids']},ensure_ascii=True))
    print(json.dumps({'reasons':dict(Counter(r['reason'] for r in output)),
                      'unknown_citations':sum(bool(r['unknown_source_ids']) for r in output)},indent=2))
if __name__=='__main__':main()
