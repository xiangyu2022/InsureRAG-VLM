"""Publish compact counts, source URLs and hashes; never publisher answer text."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha
from scripts.prepare_source_faq import chunk_text

def main():
    pairs=read(LOCAL/'extracted_pairs.json');audit=read(LOCAL/'overlap_audit.json')
    excluded={r['id'] for r in audit['rows'] if r['excluded']}
    manifest=read(LOCAL/'sealed/manifest.json')
    sources=read(LOCAL/'extraction_summary.json')['sources']
    by_group={}
    for group in sorted({r['publisher'] for r in pairs}):
        group_pairs=[r for r in pairs if r['publisher']==group]
        count=len(group_pairs);dropped=sum(r['id'] in excluded for r in group_pairs)
        split=manifest['source_assignment'][group];selected=manifest['counts'][split][group]
        by_group[group]={'unique_original_qa':count,'preseal_overlap_exclusions':dropped,
            'eligible_after_preseal_screen':count-dropped,'deterministic_cap_exclusions':count-dropped-selected,
            'split':split,'evaluated_original_questions':selected}
    relevant={(r['source_url'],r['source_sha256']) for r in pairs}
    records=[]
    for p in (LOCAL/'acquisition').glob('*.json'):
        r=read(p)
        if (r.get('final_url'),r.get('sha256')) in relevant:
            records.append({k:r[k] for k in ['group','url','final_url','acquired_utc','sha256','bytes','status_code']})
    report={'raw_unique_original_qa':len(pairs),'preseal_overlap_exclusions':len(excluded),
            'eligible_original_qa':len(pairs)-len(excluded),'per_publisher_cap':30,
            'cap_exclusions':sum(r['deterministic_cap_exclusions'] for r in by_group.values()),
            'evaluated_original_questions':sum(r['evaluated_original_questions'] for r in by_group.values()),
            'by_publisher':by_group,'answer_chunk_occurrences_before_dedup':sum(len(chunk_text(r['answer'])) for r in pairs),
            'unique_new_answer_chunks':manifest['new_answer_chunks'],
            'corpus_scope':'All 218 original answers remain retrieval distractors, including questions excluded from evaluation. Deterministic answer splitting and publisher+normalized-text ID dedup yield 219 unique chunks.',
            'historical_answer_records':192231,'total_retrieval_answer_records':192231+manifest['new_answer_chunks'],
            'source_snapshots':sorted(records,key=lambda r:r['url']),
            'extracted_pairs_sha256':sha(LOCAL/'extracted_pairs.json'),'preseal_audit_sha256':sha(LOCAL/'overlap_audit.json'),
            'not_claimed':['A production corpus expansion','Training examples','Expert adjudicated evidence sufficiency','Independent publisher minimum achieved']}
    write(ROOT/'reports/source_holdout_v1/data_flow.json',report)
    registry=[{'group':r['group'],'url':r['url']} for r in records]
    write(ROOT/'reports/source_holdout_v1/acquisition_registry.json',registry)
    print(json.dumps({k:v for k,v in report.items() if k!='source_snapshots'},indent=2))
if __name__=='__main__':main()
