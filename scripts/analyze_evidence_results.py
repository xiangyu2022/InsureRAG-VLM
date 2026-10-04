"""Post-selection failure cards and report/company-cluster sensitivity analyses."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from scripts.eval_evidence_reranker import RUN,PRIOR,PRIOR_ARM,specs,partitions,paired
from src.insurerag_vlm.evidence_ranking import summarize_evidence
from src.insurerag_vlm.reranker import blend_scores

def load(p):return json.loads(p.read_text(encoding='utf8'))


def main():
    lock=load(RUN/'selection.lock.json');summary=load(RUN/'test_evaluation/summary.json')
    assert summary['predictions_sha256']==sha(RUN/'test_evaluation/predictions.jsonl')
    predictions={(r['cohort'],r['arm'],r['id']):r for r in read_jsonl(RUN/'test_evaluation/predictions.jsonl')}
    previous={(r['cohort'],r['id']):r for r in read_jsonl(PRIOR/'test/predictions.jsonl') if r['arm']==PRIOR_ARM}
    parity={};stages={};cards=[];finqa_cases=None;fiqa_cases=None;literal_alias_sensitivity={}
    for co,qfile,afile,_ in specs('test'):
        cases=read_jsonl(ROOT/qfile);answers={a['id']:a for a in read_jsonl(ROOT/afile)}
        candidates={r['id']:r for r in read_jsonl(RUN/'test_candidates'/f'{co}.jsonl')}
        scored={r['id']:r for r in read_jsonl(RUN/'test_scores'/lock['config']['model']/f'{co}.jsonl')}
        if co!='finqa':
            changes=Counter()
            for c in cases:
                old=previous[co,c['id']];new=predictions[co,'previous_default',c['id']]
                changes['rank_list_changes']+=int(old['top_answer_ids']!=new['top_answer_ids'])
                for k in ['hit_at_1','hit_at_10','mrr_at_100','candidate_hit']:
                    changes[k+'_changes']+=int(old[k]!=new[k])
            assert not any(changes.values()),changes
            parity[co]={'questions':len(cases),**dict(changes)}
        else:finqa_cases=cases
        if co=='fiqa':fiqa_cases=cases
        # Secondary ID-label sensitivity only. Equal whitespace-normalized text
        # is not a semantic relevance judgment and never changes primary scores.
        if co in {'insuranceqa','fiqa','finqa'}:
            keys={a:' '.join(v['text'].split()) for a,v in answers.items()}
            literal_alias_sensitivity[co]={'n':len(cases),'normalization':'whitespace only; case and punctuation preserved','arms':{}}
            for arm in ['previous_default','previous_validation_best','selected','bge']:
                values={k:[] for k in [1,10]};changed={k:[] for k in [1,10]}
                for c in cases:
                    gold={keys[a] for a in c['gold_answer_ids']};r=predictions[co,arm,c['id']]
                    for k in [1,10]:
                        hit=float(any(keys[a] in gold for a in r['top_answer_ids'][:k]));values[k].append(hit)
                        assert hit>=r[f'hit_at_{k}']
                        if hit>r[f'hit_at_{k}']:changed[k].append(c['id'])
                literal_alias_sensitivity[co]['arms'][arm]={f'hit_at_{k}':{'literal_text_equivalent':float(np.mean(values[k])),
                                                                                         'additional_hits':len(changed[k]),'question_ids':changed[k]}
                                                           for k in [1,10]}
                if co=='finqa':
                    for k in [5,10]:
                        complete=[];extra=[]
                        for c in cases:
                            r=predictions[co,arm,c['id']];gold={keys[a] for a in c['gold_answer_ids']}
                            top=r['top_answer_ids'][:k]
                            assert all(answers[a]['source_group']==c['source_group'] for a in top)
                            hit=float(gold<={keys[a] for a in top});complete.append(hit)
                            assert hit>=r[f'all_evidence_at_{k}']
                            if hit>r[f'all_evidence_at_{k}']:extra.append(c['id'])
                        literal_alias_sensitivity[co]['arms'][arm][f'all_evidence_at_{k}']={
                            'literal_text_equivalent':float(np.mean(complete)),'additional_hits':len(extra),'question_ids':extra}
        for part,ix in partitions(co,cases).items():
            for arm in ['previous_default','previous_validation_best','selected']:
                counts=Counter()
                for i in ix:
                    r=predictions[co,arm,cases[i]['id']]
                    counts['top1_hit' if r['hit_at_1'] else 'top10_only' if r['hit_at_10'] else 'candidate_missing' if not r['candidate_hit'] else 'ranked_after_top10']+=1
                    if co=='finqa':
                        counts['all_evidence_top5' if r['all_evidence_at_5'] else 'candidate_missing_required_fact' if not r['candidate_all_evidence'] else 'required_fact_ranked_after_top5']+=1
                stages[f'{co}/{part}/{arm}']={'n':len(ix),**dict(counts)}
        for c in cases:
            old=predictions[co,'previous_default',c['id']];new=predictions[co,'selected',c['id']]
            metric='all_evidence_at_5' if co=='finqa' else 'hit_at_1'
            if old[metric] and new[metric]:continue
            raw=candidates[c['id']];scores=scored[c['id']]
            assert raw['candidate_ids']==scores['candidate_ids']
            blended=blend_scores(np.array(raw['dense']),np.array(raw['sparse']),np.array(scores['cross']),.2,lock['config']['cross_weight'])
            order=np.array(raw['candidate_ids'])[np.argsort(-blended,kind='stable')].tolist()
            assert order[:100]==new['top_answer_ids'];pos={a:i+1 for i,a in enumerate(order)}
            cards.append({'cohort':co,'id':c['id'],'question':c['question'],'source_group':c.get('source_group'),
                          'upstream_id':c.get('upstream_id'),'focus_metric':metric,
                          'transition':'gain' if new[metric] else 'loss' if old[metric] else 'persistent_miss',
                          'gold_answers':[{'id':a,'text':answers[a]['text'],'selected_rank':pos.get(a),
                                           **{k:answers[a][k] for k in ['source_page','evidence_type','upstream_evidence_key'] if k in answers[a]}}
                                          for a in c['gold_answer_ids']],
                          'previous_top5':[{'id':a,'text':answers[a]['text']} for a in old['top_answer_ids'][:5]],
                          'selected_top5':[{'id':a,'text':answers[a]['text']} for a in order[:5]],
                          'previous_metrics':{k:old[k] for k in ['hit_at_1','hit_at_10','all_evidence_at_5']},
                          'selected_metrics':{k:new[k] for k in ['hit_at_1','hit_at_10','all_evidence_at_5']},
                          'expert_adjudicated':False,'used_for_training_or_reselection':False})
    assert len(finqa_cases)==1147
    duplicate_groups=defaultdict(list)
    for c in finqa_cases:duplicate_groups[(c['source_group'],' '.join(c['question'].casefold().split()))].append(c)
    # Equal total weight per report/question group; average original labels within
    # duplicate groups instead of choosing whichever annotation gives a good score.
    duplicate_sensitivity={'original_test_rows':len(finqa_cases),'unique_report_question_groups':len(duplicate_groups),
                           'normalization':'casefold and whitespace only; report identifier included',
                           'groups':[{'source_group':k[0],'question':k[1],'ids':[c['id'] for c in cs],
                                      'identical_gold_ids':all(set(c['gold_answer_ids'])==set(cs[0]['gold_answer_ids']) for c in cs)}
                                     for k,cs in duplicate_groups.items() if len(cs)>1],
                           'primary_test_unchanged':True,'group_equally_weighted_metrics':{}}
    for arm in ['previous_default','previous_validation_best','selected','bge']:
        duplicate_sensitivity['group_equally_weighted_metrics'][arm]={
            metric:float(np.mean([np.mean([predictions['finqa',arm,c['id']][metric] for c in cs])
                                  for cs in duplicate_groups.values()]))
            for metric in ['hit_at_1','hit_at_10','all_evidence_at_5','all_evidence_at_10']}
    company_cases=[{**c,'source_group':c['source_group'].split('/')[0]} for c in finqa_cases]
    inherited=load(PRIOR/'data_verification.json')
    exposed={r['id'] for r in inherited['prior_reranker_near_question_flags'] if r['split']=='test'}
    unexposed=[c for c in fiqa_cases if c['id'] not in exposed]
    inherited_sensitivity={'excluded_ids':sorted(exposed),'n':len(unexposed),'primary_test_unchanged':True,
                           'prior_exposure_audit_sha256':sha(PRIOR/'data_verification.json'),
                           'summaries':{a:summarize_evidence([predictions['fiqa',a,c['id']] for c in unexposed])
                                        for a in ['previous_default','previous_validation_best','selected','bge']}}
    sensitivity={}
    for control in ['previous_default','previous_validation_best','previous_same_weight','bge']:
        result=paired([predictions['finqa','selected',c['id']] for c in finqa_cases],
                      [predictions['finqa',control,c['id']] for c in finqa_cases],company_cases,'finqa')
        result['cluster_unit']='company';sensitivity[control]=result
    strata={}
    for label,cases in [('single_fact',[c for c in finqa_cases if len(c['gold_answer_ids'])==1]),
                        ('multiple_facts',[c for c in finqa_cases if len(c['gold_answer_ids'])>1])]:
        strata[label]={'n':len(cases),'arms':{arm:summarize_evidence([predictions['finqa',arm,c['id']] for c in cases])
                                            for arm in ['previous_default','previous_validation_best','selected','bge']}}
    quality=load(RUN/'source_row_quality.json');newscope=read_jsonl(RUN/'test_candidates/finqa.jsonl')
    sizes=[r['scope_candidate_count'] for r in newscope]
    audit=load(RUN/'data_verification.json')
    keyword_ids=set(audit['insurance_keyword_slices']['test'])
    keyword_cases=[c for c in finqa_cases if c['id'] in keyword_ids]
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'historical_default_parity':parity,
            'failure_stages':stages,'failure_cards':len(cards),'finqa_company_cluster_sensitivity':sensitivity,
            'finqa_single_multiple_fact_strata':strata,
            'finqa_duplicate_question_sensitivity':duplicate_sensitivity,
            'fiqa_inherited_exposure_sensitivity':inherited_sensitivity,
            'finqa_candidate_scope_statistics':{'min':min(sizes),'median':float(np.median(sizes)),'max':max(sizes),
                                               'unique_reports':len({c['source_group'] for c in finqa_cases})},
            'finqa_insurance_keyword_slice':{'n':len(keyword_cases),'expert_insurance_classification':False,
                                             'arms':{a:summarize_evidence([predictions['finqa',a,c['id']] for c in keyword_cases]) for a in ['previous_default','selected','bge']}},
            'finqa_complete_at_5_theoretical_ceiling':sum(len(c['gold_answer_ids'])<=5 for c in finqa_cases)/len(finqa_cases),
            'source_row_quality':quality,'test_predictions_sha256':sha(RUN/'test_evaluation/predictions.jsonl'),
            'literal_text_alias_sensitivity_not_primary':literal_alias_sensitivity,
            'selection_unchanged_sha256':sha(RUN/'selection.lock.json'),'post_test_diagnostic_only':True,'code_sha256':sha(Path(__file__))}
    write_json(report,RUN/'failure_analysis.json');write_json({'n':len(cards),'expert_adjudicated':False,'cases':cards},RUN/'failure_cases.json')
    print(json.dumps({'failure_cards':len(cards),'historical_default_parity':parity,
                      'finqa_company_interval_all_evidence5':sensitivity['previous_default']['metrics']['all_evidence_at_5']}))


if __name__=='__main__':main()
