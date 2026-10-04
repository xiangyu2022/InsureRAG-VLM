"""Export all frozen arms and paired uncertainty without raw content."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha,verify_test_lock
from src.insurerag_vlm.evidence_evaluation import publisher_cluster_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True);a=p.parse_args()
    if a.split=='test':verify_test_lock('cross_quarter')
    arms=['baseline','cross_only','original_query','cross_quarter'] if a.split=='dev' else ['baseline','cross_quarter']
    scores={arm:read(LOCAL/(a.split+'_'+arm)/'summary.json') for arm in arms}
    generations={}
    for arm in ['baseline','cross_quarter']:
        folder=LOCAL/(a.split+'_'+arm)/'generation';path=folder/'summary_v2.json'
        if not path.exists():path=folder/'summary.json'
        s=read(path)
        if s.get('schema_version') not in {2,3}:raise ValueError('Use versioned citation-free content metrics')
        generations[arm]={k:v for k,v in s.items() if k!='rows'}
    if len({g['schema_version'] for g in generations.values()})!=1:raise ValueError('Do not compare arms with different metric versions')
    contrast=[publisher_cluster_delta(scores['baseline']['rows'],scores['cross_quarter']['rows'],metric)
              for metric in ['complete_after_packing','recall_at_10','hit_at_10','ndcg_at_10']]
    report={'split':a.split,'interpretation':'EXPLORATORY; preregistered minimum of 3 publisher groups in each split not met',
            'decision':read(ROOT/'reports/source_holdout_v1/selection.lock.json')['decision'],
            'retrieval':{arm:{k:v for k,v in s.items() if k!='rows'} for arm,s in scores.items()},
            'generation':generations,'paired_retrieval_contrasts':contrast,
            'selection_lock_sha256':sha(ROOT/'reports/source_holdout_v1/selection.lock.json'),
            'no_new_training':True,'no_default_promotion':True,
            'metric_caveats':['FAQ answer chunks are relevance proxies, not adjudicated sufficient evidence.',
                'Original FAQ questions may omit jurisdiction; exact source recall is not universal factual correctness.',
                'Both raw and served outputs retain the same original question denominators; errors are not discarded.',
                'Citation-ID correctness does not establish claim entailment. No blinded expert semantic review was performed.',
                'One generation per question/cohort/arm; fixed seed does not imply byte-for-byte deterministic GPU output.']}
    if a.split=='dev':
        report['independent_candidate_timing']=read(LOCAL/'dev_cross_quarter_timing/summary.json')['latency_seconds']
        report['replay_latency_caveat']='cross_only and original cross_quarter cached replay inherit baseline latency; use independent_candidate_timing for cross_quarter.'
    write(ROOT/'reports/source_holdout_v1'/(a.split+'_comparison.json'),report)
    print(json.dumps({'split':a.split,'primary_contrast':contrast[0],
        'served_coverage':{arm:g['served_abstention']['answerable_coverage'] for arm,g in generations.items()}},indent=2))
if __name__=='__main__':main()
