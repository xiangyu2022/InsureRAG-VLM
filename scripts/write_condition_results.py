"""Write measured Chinese results and a compact README entry from verified artifacts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(p):return json.loads(p.read_text(encoding='utf8'))
def pct(x):return f'{x*100:.2f}%'
def comparison(r):
    delta=r.get('hit10_difference',r.get('hit_at_10_difference'));ci=r.get('label_cluster_bootstrap_95',r.get('source_cluster_bootstrap_95ci'))
    return f'{100*delta:+.2f} 个百分点（95% 簇 bootstrap 区间 {100*ci[0]:+.2f} 至 {100*ci[1]:+.2f}；{r["wins"]} 胜 / {r["losses"]} 负）'
def run():
    run=ROOT/'reports/condition_v1';lock=load(run/'selection.lock.json');data=load(run/'test_evaluation/summary.json');verified=load(run/'verification.json')
    assert verified['status']=='verified';failure=load(run/'failure_followup.json');sweep=load(run/'validation_sweep.json');lengths=load(run/'length_audit.json')
    specs=[('historical_insuranceqa','all','保险 FAQ 历史测试 / 2,000'),('historical_government','all','政府问答历史测试 / 416'),
        ('historical_multidomain','government','上一轮政府测试 / 81'),('historical_multidomain','general','上一轮通用测试 / 600'),('fresh_government','all','本轮新政府测试 / 157')]
    table=['| 测试集 | BGE | 旧模型原设置 | 旧模型同设置 | 新选择结果 |','| --- | ---: | ---: | ---: | ---: |']
    for cohort,part,label in specs:
        m=data['summaries'][cohort][part];table.append('| '+label+' | '+' | '.join(pct(m[a]['hit_at_10']) for a in ['bge','previous_default','previous_matched','selected'])+' |')
    cfg=lock['config'];choice='新的训练权重已通过验证集选择' if lock['promoted_weights'] else '新增训练没有通过预定的晋升规则，因此保留上一轮权重'
    lines=['# InsureRAG 条件问答与难负例训练实测','',
        f'本轮已完成数据扩充、两次 GPU 训练、验证选择和固定测试。{choice}。最终选择 `{lock["selected_model"]}`，候选池 `{cfg["pool"]}`，BM25 权重 {cfg["lexical_weight"]}，重排权重 {cfg["cross_weight"]}。',
        '', '## 实际结果','', '下表均为 Hit@10：正确标注段落出现在前十条的比例。“同设置”使用与新结果相同的候选池和融合权重，便于区分训练收益和检索设置变化。','',*table,'',
        '![实测结果与配对置信区间](assets/condition_results.png)','',
        '公共 MiniLM 也以原设置及同设置参加全部对照；完整 Hit@1、Hit@5、Hit@10、Hit@100、MRR 和逐题排名见 `reports/condition_v1/test_evaluation/`。不同测试使用不同候选库，不能合成一个统一“准确率”。','']
    for cohort,part,label in specs:
        paired=data['paired_selected_minus_controls'][cohort][part]
        lines += [f'- **{label}**：相对旧模型同设置，{comparison(paired["previous_matched"])}；相对 BGE，{comparison(paired["bge"])}。']
    iq=data['summaries']['historical_insuranceqa']['all'];fresh=data['summaries']['fresh_government']['all']
    lines += ['',f'保险 FAQ 的 Hit@1 为 {pct(iq["selected"]["hit_at_1"])}，MRR@100 为 {iq["selected"]["mrr_at_100"]:.4f}。本轮新政府测试的 Hit@1 为 {pct(fresh["selected"]["hit_at_1"])}，MRR@100 为 {fresh["selected"]["mrr_at_100"]:.4f}。',
        '', '## 失败诊断如何改变了训练','',
        '上一轮保险 FAQ 的 533 道未命中题中，194 道的正确答案不在 BGE/BM25 联合候选内，34 道在候选池截为 100 条时丢失，305 道属于后续排序错误。另对 20 道 FAQ 退步题和 1 道政府退步题做了助手审阅：既有资格、主体、时间范围混淆，也有未被原标签列为正确的近似答案。后者没有擅自改成正例。',
        '', '本轮据此增加当前模型会排错的难负例、同来源的相邻政府答案，以及数值/时间/条件导向的原始人工问题。公共模型也认为相关的负例使用较小惩罚，并保留公共模型蒸馏约束。旧测试题只用于诊断方法方向，没有加入训练。',
        '', '| 本轮固定模型 | 前十命中 | 联合候选缺失 | 候选裁剪丢失 | 排序到十名后 |','| --- | ---: | ---: | ---: | ---: |']
    for cohort,v in failure['summaries'].items():
        d=v['failure_stages'];lines.append('| '+cohort+' | '+' | '.join(str(d.get(k,0)) for k in ['hit_at_10','absent_from_union','candidate_pool_cutoff','reranking_below_top10'])+' |')
    lines += ['', '完整失败卡片保留问题、原始正确答案、前三条检索结果、旧模型是否命中和失败阶段，见 `reports/condition_v1/failure_cases.json`。这些是本轮评分后的诊断，未用于重选模型。',
        '', '## 数据与实际训练','',
        '- 训练问题从 **19,941 增至 25,987**：11,729 道 InsuranceQA、259 道政府问答、13,999 道 SQuAD。新增 6,046 道中，46 道属于政府问答，6,000 道属于通用条件问答，不能全部称为保险数据。',
        '- 本轮新增独立政府测试 **157 道 / 28 个来源 URL**，搜索 **48,172 条答案**；合并 ACA 草案/最终版本后为 27 个来源家族，额外敏感性分析见 `failure_followup.json`。旧的 3,097 道测试现在只作历史回归检查。',
        '- 285,857 个难负例来自训练问题的检索候选；新留出来源及重复文本没有进入正负训练对。测试题未用来挖掘训练负例或选择超参数。',
        '- 政府问答逐条保留原文区间和 URL；6,000 道通用新增题逐条核对原始 SQuAD 问题与答案位置。格式抽查发现 23 条问答边界不可靠记录，已在训练和评估之前排除，保留前置快照与变更记录。',
        f'- 两个随机种子分别实际训练 1 个 epoch，合计 **{verified["total_optimizer_steps"]:,} 次优化更新、{verified["total_pair_exposures"]:,} 次问答对呈现**。重复呈现没有计成新增数据。两份权重均与初始模型不同，见参数差异与训练日志。',
        '', '| 随机种子 | 优化更新 | 问答对呈现 | 训练秒数 | 参数值改变数 |','| --- | ---: | ---: | ---: | ---: |']
    for t in verified['training_runs']:lines.append(f'| {t["seed"]} | {t["steps"]:,} | {t["pair_exposures"]:,} | {t["seconds"]:.1f} | {t["changed_values"]:,} |')
    lines += ['', '两种子及旧模型各比较 6 组事先登记的设置；按 0.6 FAQ、0.2 政府、0.2 通用的验证 Hit@10 加权选择，并施加各域回退限制。两个种子的测试成绩没有用于挑选更好的种子。完整验证扫描为 `validation_sweep.json`。',
        '', '## 仍然存在的限制','',
        f'新增训练政府题中 {lengths["counts"]["train_government"]["over_512"]}/46 道、新测试中 {lengths["counts"]["test_government"]["over_512"]}/157 道问答对超过 512 tokens。超长不等于关键条件一定丢失，但目前仍使用固定截断；不能靠扩大训练样本解决全部长上下文问题。',
        '', '政府测试使用机械抽取的公开问答，尚未经保险专家逐题审定；部分问题依赖章节背景，页眉/脚注可能残留。157 道题、28 个来源仍不足以证明跨保险产品与司法辖区的普遍有效性。原始政府资料保留历史日期，不代表现行法律结论。FAQ 标签不完整、历史集合已多次查看、公共预训练数据可能接触过文本等限制继续存在。',
        '', '本轮训练的是检索重排组件。BGE 向量模型、Qwen、图推理和 PDF 回答生成没有被这组指标验证；Hit@10 不能写成最终回答准确率。',
        '', '## 使用和复现','',
        '使用 `scripts/query_condition_reranker.py` 查询选定模型；支持 `condition`、`insuranceqa`、`hicric` 三个明确语料选项。数据、模型、代码、原始分数和选择顺序均有 SHA-256 记录。详情见 [复现说明](CONDITION_REPRODUCTION.md) 与 `reports/condition_v1/verification.json`。','']
    (ROOT/'docs/CONDITION_UPDATE_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    readme=ROOT/'README.md';old=readme.read_text(encoding='utf8');marker='## Condition-oriented hard-negative training'
    if marker in old:raise ValueError('README update already present')
    rows=['| Separate test population | BGE Hit@10 | Previous / matched | Selected |','| --- | ---: | ---: | ---: |']
    labels=['Historical FAQ / 2,000','Historical government / 416','Previous government / 81','Previous general / 600','Fresh government / 157']
    for (c,p,_),label in zip(specs,labels):
        m=data['summaries'][c][p];rows.append('| '+label+' | '+' | '.join(pct(m[a]['hit_at_10']) for a in ['bge','previous_matched','selected'])+' |')
    text='\n'.join([marker+' — October 1, 2026','',
        'The latest completed iteration expands training to **25,987 unique questions**',
        '(11,729 FAQ, 259 government, 13,999 general) and mines 285,857 training negatives.',
        f'Two independently seeded CUDA runs are complete. Validation selects **{lock["selected_model"]}**',
        f'with **{cfg["pool"]}**, lexical weight {cfg["lexical_weight"]}, cross weight {cfg["cross_weight"]}.',
        '',*rows,'',
        'Matched controls use the selected candidate/fusion configuration; the full report',
        'also retains original-config and public-reranker controls. The new 157-question',
        'government test spans 28 URLs and 48,172 answer candidates. Earlier tests are',
        'historical regression checks. Questions from held-out sources are excluded',
        'from positive and negative training pairs. Two pre-training format/code fixes',
        'are archived with their prior manifests; no test-based exclusions were made.','',
        'See [measured results and limitations](docs/CONDITION_UPDATE_ZH.md),',
        '[reproduction instructions](docs/CONDITION_REPRODUCTION.md),',
        '[raw evidence](reports/condition_v1/), and the',
        '[selected-model query CLI](scripts/query_condition_reranker.py).',
        'This is a reranker experiment, not a Qwen/GraphRAG answer-accuracy claim.','',
        ''])
    old=old.replace('## Mixed-domain retention training — October 1, 2026','## Previous mixed-domain retention training — October 1, 2026',1)
    old=old.replace('The latest research iteration expands supervised training','This earlier research iteration expands supervised training',1)
    insertion=old.index('## Previous mixed-domain retention training');readme.write_text(old[:insertion]+text+old[insertion:],encoding='utf8')

if __name__=='__main__':run()
