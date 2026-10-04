"""Export expanded audit scope and frozen-split overlap without historical text."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha

def main():
    inv=read(LOCAL/'history_v2/historical_inventory.json');old=read(LOCAL/'overlap_audit.json');new=read(LOCAL/'overlap_audit_v2.json')
    a={r['id'] for r in old['rows'] if r['excluded']};b={r['id'] for r in new['rows'] if r['excluded']}
    frozen={split:{r['id'] for r in read(LOCAL/'sealed'/(split+'.json'))} for split in ['dev','test']}
    domains=['ccpc.ie','hia.ie','privatehealth.gov.au','fsrao.ca','fsco.gov.on.ca','dfr.oregon.gov']
    report={'status':'Expanded corrective audit after selection lock; no resplitting, retuning or removal from fixed evaluation denominators.',
        'initial_preseal_scope':'Project data roots plus prior 26-case diagnostic; did not include all historical report JSON.',
        'initial_unique_files':142,'initial_unique_texts':313423,'initial_questions':43976,
        'expanded_scope':inv['scope'],'files_examined':len(inv['files']),'unique_file_contents':inv['unique_file_count'],
        'unique_normalized_strings':inv['text_count'],'question_key_strings':inv['question_count'],
        'by_root_file_counts':dict(Counter(r['path'].split('/')[0] for r in inv['files'])),
        'read_failures_in_included_roots':inv['failures'],'excluded_roots':inv['exclusions'],
        'known_access_gaps':'Ten historical demo session directories denied enumeration. Their containing demo_uploads tree was excluded from the corrective scan; no escalation, permission changes or bypass. Raw PDF bytes were not separately re-extracted for historical matching.',
        'publisher_url_domain_mentions':{d:sum(n for h,n in inv['hosts'].items() if h==d or h.endswith('.'+d)) for d in domains},
        'publisher_exposure_caveat':'No matching URL-domain mentions is a recorded screen, not proof against model pretraining, implicit publisher-name mentions, or inaccessible material.',
        'audit_method':new['method'],'threshold':new['threshold'],'candidate_original_pairs':new['candidate_pairs'],
        'initial_flagged':len(a),'expanded_flagged':len(b),'union_flagged':len(a|b),
        'newly_flagged_ids':sorted(b-a),'no_longer_flagged_ids':sorted(a-b),
        'idf_change_caveat':'Expanded corpus changes smoothed IDF; preserve the initial exclusions and report the union, never reinstate initially excluded questions.',
        'frozen_split_counts':{s:len(ids) for s,ids in frozen.items()},
        'frozen_split_flagged_by_either_audit':{s:sorted(ids&(a|b)) for s,ids in frozen.items()},
        'audit_runtime_seconds':new['seconds'],
        'hashes':{str(p.relative_to(LOCAL)).replace('\\','/'):sha(p) for p in [LOCAL/'overlap_audit.json',LOCAL/'overlap_audit_v2.json',
                 LOCAL/'history_v2/historical_inventory.json',LOCAL/'history_v2/historical_texts.jsonl']},
        'limits':'EXPLORATORY regardless of overlap result: heldout has only two publisher groups, below preregistered minimum. Full-answer relevance labels are mechanical, not expert semantic labels.'}
    write(ROOT/'reports/source_holdout_v1/history_audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
