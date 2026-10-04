"""Save assistant-reviewed examples without altering labels, models or rankings."""
from datetime import datetime,timezone
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
RUN=ROOT/'reports/evidence_reranker_v1'
NOTES=[
 ('insuranceqa','iqa_v2_test_00035','gain','意图匹配改善',
  '问题询问失能保险的基础知识；旧首条偏向投保资格，新首条转向保障作用和等待期。这是与原始标注意图更接近，不构成对原文保险或税务说法的当前事实核验。'),
 ('insuranceqa','iqa_v2_test_00432','loss','受损对象与险种情境错配',
  '问题场景是汽车撞到行人。原标注指向人身伤害责任，新首条转向泛化车险报案及车辆损失文本；标注答案降至第 2。主题相近仍可能遗漏受损对象这一关键条件。'),
 ('insuranceqa','iqa_v2_test_00262','loss','原始标签可能不穷尽合理答案',
  '原标注答案由第 1 降至第 2，新首条也讨论移除受抚养人的保障及个体/团体计划条件。它属于确定的 ID 标签失分，但是否为真实业务退步需要专家复核；本轮不据此重标。'),
 ('finqa','finqa_q_847421011236c2e635bc78d5','gain','分子和分母行共同进入前五',
  '股票奖励占总收购价的问题，两个原始标注行在新模型下进入第 5 和第 3。未运行数值计算。原 FinQA 表格文本化可能把首行数据作为列名，文字表述仍有抽取格式局限。'),
 ('finqa','finqa_q_668b3f95d2c3b60851237141','loss','表头落后于数值行',
  '数值行升到第 1，但表头从第 1 降至第 8，严格完整证据@5 因而失分。列名和时间信息也出现在序列化数值行中；该指标不等价于最小充分证据或最终算术正确性。'),
 ('fiqa','fiqa_q_98','gain','命中标注不等于建议质量提升',
  '首位命中的原始标注答案带有反讽性质：以更大的初始资金回应目标收益问题。此次标签命中增加不能解释为可执行金融建议变得更好。'),
 ('fiqa','fiqa_q_1812','loss','决策意图错配',
  '原题涉及共有住房及共同房贷的分配安排，新首条转向把共有房产用于贷款抵押。仍围绕共有产权，但偏离分配责任的决策情境；原标注答案在第 4 和第 7。')]


if __name__=='__main__':
    path=RUN/'failure_cases.json';source=json.loads(path.read_text(encoding='utf8'))
    cards={(r['cohort'],r['id']):r for r in source['cases']}
    old={(r['cohort'],r['id']):r for r in read_jsonl(RUN/'test_evaluation/predictions.jsonl') if r['arm']=='previous_default'}
    result=[]
    for co,ident,transition,title,note in NOTES:
        c=cards[co,ident];assert c['transition']==transition
        order=old[co,ident]['top_answer_ids']
        result.append({**c,'review_title_zh':title,'assistant_observation_zh':note,
                       'gold_answers':[{**g,'previous_rank':order.index(g['id'])+1 if g['id'] in order else None} for g in c['gold_answers']]})
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'assistant','expert_review_completed':0,
                'illustrative_selection_not_a_random_sample':True,'used_for_retraining_or_reselection':False,
                'cases':result,'failure_cards_sha256':sha(path),'selection_lock_sha256':sha(RUN/'selection.lock.json'),
                'code_sha256':sha(Path(__file__))},RUN/'reviewed_test_cases.json')
    print(json.dumps({'reviewed_cases':len(result),'expert_review_completed':0,'test_labels_changed':False}))
