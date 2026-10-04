"""Freeze an exploratory comparison without promoting a failed candidate."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha

def main():
    target=ROOT/'reports/source_holdout_v1/selection.lock.json'
    if target.exists():raise ValueError('Selection already locked; do not overwrite')
    reports={arm:read(LOCAL/('dev_'+arm)/'summary.json') for arm in ['baseline','cross_only','original_query','cross_quarter']}
    generations={arm:read(LOCAL/('dev_'+arm)/'generation/summary.json') for arm in ['baseline','cross_quarter']}
    timing=read(LOCAL/'dev_cross_quarter_timing/summary.json')
    def rows(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
    replay=rows(LOCAL/'dev_cross_quarter/retrieval.jsonl');rerun=rows(LOCAL/'dev_cross_quarter_timing/retrieval.jsonl')
    if len(replay)!=47 or len(rerun)!=47 or any(a['id']!=b['id'] or a['order']!=b['order'] or a['context']!=b['context'] for a,b in zip(replay,rerun)):
        raise ValueError('Independent inference must reproduce replay ranking and context')
    b=reports['baseline'];c=reports['cross_quarter'];bg=generations['baseline'];cg=generations['cross_quarter']
    checks={
        'primary_gain_at_least_0_03':c['publisher_macro']['complete_after_packing']-b['publisher_macro']['complete_after_packing']>=.03,
        'recall_at_10_drop_at_most_0_01':c['publisher_macro']['recall_at_10']>=b['publisher_macro']['recall_at_10']-.01,
        'served_coverage_drop_at_most_0_02':cg['served_abstention']['answerable_coverage']>=bg['served_abstention']['answerable_coverage']-.02,
        'served_missing_context_refusal_recall_at_least_0_95':cg['served_abstention']['abstention_recall']>=.95,
        'raw_citation_id_precision_1_across_both_cohorts':cg['raw_citation_id_precision']==1,
        'raw_positive_citation_id_precision_1':cg['citation_cohorts']['retrieved']['raw_id_precision']==1,
        'served_citation_id_precision_1':cg['citation_cohorts']['retrieved']['served_id_precision']==1,
        'retrieval_p95_at_most_1_5x':timing['latency_seconds']['p95']<=1.5*b['latency_seconds']['p95'],
        'generation_p95_at_most_1_5x':cg['generation_wall_seconds']['p95']<=1.5*bg['generation_wall_seconds']['p95']}
    if all(checks.values()):raise ValueError('This rejection lock requires actual failed guardrails; review decision explicitly')
    code=['scripts/retrieve_source_holdout.py','scripts/generate_source_holdout.py','scripts/eval_insuranceqa_reranker.py',
          'scripts/eval_insuranceqa_scale.py','src/insurerag_vlm/hybrid_pipeline.py','src/insurerag_vlm/config.py',
          'src/insurerag_vlm/answer_safety.py','src/insurerag_vlm/vlm.py','src/insurerag_vlm/domain_reranker.py',
          'src/insurerag_vlm/query_adaptation.py']
    lock={'locked_utc':datetime.now(timezone.utc).isoformat(),'decision':'NO PROMOTION; retain existing default strategy',
          'frozen_test_comparators':['baseline','cross_quarter'],'candidate_count':3,'no_more_tuning':True,
          'test_purpose':'Exploratory transfer estimate for one frozen retrieval comparison, not validation of a deployable winner.',
          'test_status_at_lock':'No test question/label inspection or inference for strategy selection; source ingestion and overlap audit were mechanical.',
          'guardrail_interpretation':'Preregistration did not specify raw versus served scope for citation-ID validity. Conservatively require both; do not narrow scope after observing failures. Candidate also fails raw positive-only ID validity.',
          'development_guardrails':checks,'default_runtime_changes':False,'new_training':False,
          'development_primary':{arm:r['publisher_macro']['complete_after_packing'] for arm,r in reports.items()},
          'development_retrieval_timing_parity':{'questions':47,'all_rankings_equal':True,'all_contexts_equal':True,
             'independent_p95_seconds':timing['latency_seconds']['p95'],'retrieval_sha256':sha(LOCAL/'dev_cross_quarter_timing/retrieval.jsonl')},
          'data_status':read(LOCAL/'sealed/manifest.json')['interpretation'],
          'sealed_test_sha256':sha(LOCAL/'sealed/test.json'),
          'protocol_sha256':sha(ROOT/'reports/source_holdout_v1/preregistration.json'),
          'split_manifest_sha256':sha(ROOT/'reports/source_holdout_v1/split_manifest.json'),
          'candidate_registry_sha256':sha(ROOT/'reports/source_holdout_v1/candidate_registry.json'),
          'inference_files_sha256':{p:sha(ROOT/p) for p in code},
          'development_evidence_sha256':{str(p.relative_to(LOCAL)).replace('\\','/'):sha(p) for arm in reports for p in [LOCAL/('dev_'+arm)/'summary.json']}|
             {str(p.relative_to(LOCAL)).replace('\\','/'):sha(p) for arm in generations for p in [LOCAL/('dev_'+arm)/'generation/summary.json']}}
    write(target,lock);print(json.dumps(lock,indent=2))
if __name__=='__main__':main()
