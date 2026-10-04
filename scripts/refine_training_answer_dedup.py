"""Remove duplicate held-out positive answer TEXT, including distinct author answer IDs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha,verify_fixture
from scripts.prepare_reranker_training import normalize


def run(args):
    fixture=ROOT/'data/benchmarks/insuranceqa_v2';verify_fixture(fixture)
    original=json.loads((args.source/'manifest.lock.json').read_text(encoding='utf8'))
    for name,expected in original['files'].items():
        if sha(args.source/name)!=expected:raise ValueError('Original training fixture changed')
    answers={a['id']:normalize(a['text']) for a in read_jsonl(fixture/'answers.jsonl')}
    heldout=read_jsonl(fixture/'valid.jsonl')+read_jsonl(fixture/'test.jsonl')
    excluded_texts={answers[a] for c in heldout for a in c['gold_answer_ids']}
    groups=read_jsonl(args.source/'train_groups.jsonl');retained=[];excluded=[]
    for group in groups:
        matching=[a for a in group['positive_ids'] if answers[a] in excluded_texts]
        if matching:excluded.append({'id':group['id'],'positive_answer_ids':matching,'reason':'normalized_positive_answer_text_overlap_with_heldout'})
        else:retained.append(group)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'train_groups.jsonl').open('w',encoding='utf8') as handle:
        for row in retained:handle.write(json.dumps(row)+'\n')
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),
        'previous_retained':len(groups),'retained_train':len(retained),'additional_exclusions':excluded,
        'heldout_used_only_for_exclusion':True,'test_metrics_used_for_this_fix':False,
        'rule':'Reject whole train question if any normalized positive answer text matches any validation/test positive',
        'parent_decontamination_sha256':sha(args.source/'decontamination.json'),
        'negative_candidates':'Unlabeled corpus candidates may contain held-out answer texts; held-out relevance labels are not training targets.'},args.output/'decontamination.json')
    manifest={**original,'created_utc':datetime.now(timezone.utc).isoformat(),'version':'insuranceqa_hardneg_v2',
        'parent_manifest_sha256':sha(args.source/'manifest.lock.json'), 'refinement_code_sha256':sha(Path(__file__)),
        'training_questions':len(retained),'additional_text_duplicate_exclusions':len(excluded),
        'labeled_positive_pairs':sum(len(g['positive_ids']) for g in retained),
        'mined_negative_pairs':sum(len(g['negative_ids']) for g in retained),
        'negative_types':dict(Counter(t for g in retained for t in g['negative_sources'].values())),
        'domains':dict(Counter(g['domain'] for g in retained)),
        'files':{name:sha(args.output/name) for name in ['train_groups.jsonl','decontamination.json']}}
    write_json(manifest,args.output/'manifest.lock.json')
    print(json.dumps({k:manifest[k] for k in ['training_questions','additional_text_duplicate_exclusions','labeled_positive_pairs','mined_negative_pairs']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
