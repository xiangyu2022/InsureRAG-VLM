"""Record four concrete, post-training validation observations without relabelling."""
from datetime import datetime,timezone
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from scripts.eval_evidence_reranker import RUN,check_contract
from scripts.eval_insuranceqa_reranker import rank_row

CASES=[
    ('fiqa','fiqa_q_543','事件条件错配',
     '题目询问配偶去世后的独资企业处理。新首条文章讨论经营亏损与税务处理：保留了独资企业主题，遗漏了去世这一事件条件。这里只判断主题对应，不判断原文法律建议是否正确。'),
    ('fiqa','fiqa_q_530','更贴近问题意图',
     '题目询问现金购买交易股票为何不算费用。标注回答从第 5 升到第 1；它直接讨论资产交换与会计处理，旧首条主要讨论交易双方是否公平。'),
    ('finqa','finqa_q_9bb37e6fdf2804150cecdf84','多事实证据补齐',
     '计算期内贷款账面值变动需要期初、期末行。新模型将期初行从第 6 提至第 4，使两条原始标注行都进入前五。未执行数值计算，也未验证生成答案。'),
    ('finqa','finqa_q_5cc6e3adb6fc49d7d03f4088','表头完整性失分',
     '加拿大行与总计行仍为前两条，但作为第三条标注证据的表头从第 3 降到第 6。序列化数值行本身也包含列名和单位，因此严格标注 ID 完整性不等同于最小充分语义证据或算术正确性。')]


if __name__=='__main__':
    check_contract();model='seed_42_epoch_1';records=[];fingerprints={}
    for co in ['fiqa','finqa']:
        fixture=ROOT/'data/benchmarks'/('fiqa_v1' if co=='fiqa' else 'finqa_evidence_v1')
        cases={r['id']:r for r in read_jsonl(fixture/'valid.jsonl')};answers={r['id']:r for r in read_jsonl(fixture/'answers.jsonl')}
        raw={r['id']:r for r in read_jsonl(RUN/'valid_candidates'/f'{co}.jsonl')};scores={}
        for arm in ['previous',model]:
            folder=RUN/'valid_scores'/arm;completion=json.loads((folder/'completion.json').read_text(encoding='utf8'))
            assert sha(folder/f'{co}.jsonl')==completion['score_files_sha256'][f'{co}.jsonl']
            fingerprints[f'{co}/{arm}']=sha(folder/f'{co}.jsonl')
            scores[arm]={r['id']:r for r in read_jsonl(folder/f'{co}.jsonl')}
        for domain,ident,title,note in CASES:
            if domain!=co:continue
            q=cases[ident];r=raw[ident];orders={}
            for arm in scores:
                s=scores[arm][ident];assert s['candidate_ids']==r['candidate_ids']
                orders[arm]=rank_row({**r,'cross':s['cross']},{'pool':'union200','lexical_weight':.2,'cross_weight':.5})
            records.append({'cohort':co,'id':ident,'question':q['question'],'title_zh':title,'assistant_observation_zh':note,
                            'gold':[{'id':a,'text':answers[a]['text'],
                                     'ranks':{arm:o.index(a)+1 if a in o else None for arm,o in orders.items()}}
                                    for a in q['gold_answer_ids']],
                            'top5':{arm:[{'id':a,'text':answers[a]['text']} for a in o[:5]] for arm,o in orders.items()},
                            'expert_adjudicated':False,'labels_changed':False})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'split':'valid','checkpoint':model,'cross_weight':.5,
            'illustrative_cases_not_random_error_prevalence_sample':True,'used_for_retraining_or_selection_rule_changes':False,
            'source_score_files_sha256':fingerprints,'cases':records,'code_sha256':sha(Path(__file__))}
    write_json(result,RUN/'validation_case_review.json')
    print(json.dumps({'recorded_validation_cases':len(records),'checkpoint':model,'changed_frozen_protocol':False}))
