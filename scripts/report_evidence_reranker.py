"""Write result tables, model cards and figures exclusively from recorded artifacts."""
import csv,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha
RUN=ROOT/'reports/evidence_reranker_v1'
def load(p):return json.loads(p.read_text(encoding='utf8'))
COHORTS=[('insuranceqa','all','保险 FAQ 历史集','Insurance FAQ'),
         ('historical_government','all','政府历史集','Government historical'),
         ('multidomain','government','此前政府集','Government previous'),
         ('multidomain','general','此前通用集','General previous'),
         ('condition_government','all','条件政府集','Government condition'),
         ('fiqa','all','上一轮金融论坛集','FiQA previous'),
         ('finqa','all','新财报证据集','FinQA new reports')]


def main():
    summary=load(RUN/'test_evaluation/summary.json');lock=load(RUN/'selection.lock.json');verification=load(RUN/'verification.json')
    failures=load(RUN/'failure_analysis.json');audit=load(RUN/'data_verification.json');train=load(ROOT/'data/training/evidence_reranker_v1/manifest.lock.json')
    rows=['| 测试集 / 题数 | BGE Hit@10 | 上轮流程 Hit@10 | 旧模型验证最优 Hit@10 | 本轮选定 Hit@10 | 上轮 → 本轮 Hit@1 |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
    cirows=['| 测试集 | Hit@1 差值 / 百分点（95% 簇区间） | Hit@10 差值 / 百分点（95% 簇区间） | MRR 差值（95% 簇区间） |',
            '| --- | ---: | ---: | ---: |']
    def interval(d,scale=100):
        lo,hi=d['cluster_bootstrap_95ci'];return f'{d["difference"]*scale:+.2f}（{lo*scale:+.2f}, {hi*scale:+.2f}）' if scale==100 else f'{d["difference"]:+.4f}（{lo:+.4f}, {hi:+.4f}）'
    for co,part,zh,_ in COHORTS:
        arms=summary['summaries'][co][part];new=arms['selected'];old=arms['previous_default'];best=arms['previous_validation_best']
        rows.append(f'| {zh} / {new["n"]:,} | {arms["bge"]["hit_at_10"]:.2%} | {old["hit_at_10"]:.2%} | {best["hit_at_10"]:.2%} | {new["hit_at_10"]:.2%} | {old["hit_at_1"]:.2%} → {new["hit_at_1"]:.2%} |')
        d=summary['paired_selected_minus_controls'][co][part]['previous_validation_best']['metrics']
        cirows.append(f'| {zh} | {interval(d["hit_at_1"])} | {interval(d["hit_at_10"])} | {interval(d["mrr_at_100"],1)} |')
    fin=summary['summaries']['finqa']['all'];faq=summary['summaries']['insuranceqa']['all'];fd=summary['paired_selected_minus_controls']['finqa']['all']['previous_validation_best']['metrics']
    validation=load(RUN/'validation_sweep.json')
    vrows=['| 配置 | 验证综合值 | FAQ Hit@1 | FAQ Hit@10 | FiQA Hit@10 | 财报完整证据@5 | 通过约束 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in validation:
        m=r['domains'];mark=' **选定**' if r['config']==lock['config'] else ''
        vrows.append(f'| `{r["config"]["name"]}`{mark} | {r["utility"]:.5f} | {m["insuranceqa"]["hit_at_1"]:.2%} | {m["insuranceqa"]["hit_at_10"]:.2%} | {m["finance"]["hit_at_10"]:.2%} | {m["financial_report"]["all_evidence_at_5"]:.2%} | {"是" if r["eligible"] else "否"} |')
    aliases=failures['literal_text_alias_sensitivity_not_primary']['insuranceqa']['arms']
    fin_aliases=failures['literal_text_alias_sensitivity_not_primary']['finqa']['arms']
    strata=failures['finqa_single_multiple_fact_strata']
    duplicates=failures['finqa_duplicate_question_sensitivity']
    inherited=failures['fiqa_inherited_exposure_sensitivity']
    reviewed=load(RUN/'reviewed_test_cases.json')
    reviewrows=['| 测试案例 | 标注指标变化 | 逐例观察 |','| --- | --- | --- |']
    for c in reviewed['cases']:
        reviewrows.append(f'| `{c["id"]}` | {"改善" if c["transition"]=="gain" else "退步"} | **{c["review_title_zh"]}**：{c["assistant_observation_zh"]} |')
    faqdiff=summary['paired_selected_minus_controls']['insuranceqa']['all']['previous_validation_best']['metrics']
    faqstage=failures['failure_stages']['insuranceqa/all/selected'];finstage=failures['failure_stages']['finqa/all/selected']
    regressions=[]
    for co,part,zh,_ in COHORTS:
        arms=summary['summaries'][co][part]
        if arms['selected']['hit_at_1']<arms['previous_validation_best']['hit_at_1']:
            regressions.append(f'{zh} {arms["previous_validation_best"]["hit_at_1"]:.2%} → {arms["selected"]["hit_at_1"]:.2%}')
    regression_text='首位命中的点估计回退包括：'+'；'.join(regressions)+'。各任务需分别解读，不能概括为全面提升。' if regressions else '各集合首位命中的点估计均未下降；是否可靠仍需结合区间和测试独立性判断。'
    def uncertainty(d):
        lo,hi=d['cluster_bootstrap_95ci'];return '区间包含 0，尚不能认定稳定变化' if lo<=0<=hi else '名义区间未跨 0'
    finrows=['| 指标 / 1,147 题 | 公共 BGE | 上轮流程 | 旧模型验证最优 | 本轮选定 |','| --- | ---: | ---: | ---: | ---: |']
    for metric,label in [('hit_at_1','首条命中至少一条事实'),('all_evidence_at_5','前五包含全部标注事实'),('all_evidence_at_10','前十包含全部标注事实'),('evidence_recall_at_5','前五事实召回比例')]:
        finrows.append('| '+label+' | '+' | '.join(f'{fin[a][metric]:.2%}' for a in ['bge','previous_default','previous_validation_best','selected'])+' |')
    recordkeys=['hit_at_1','hit_at_5','hit_at_10','hit_at_100','mrr_at_100','ndcg_at_10','label_recall_at_10','candidate_hit','all_evidence_at_5','all_evidence_at_10','evidence_recall_at_5','candidate_all_evidence']
    with (RUN/'test_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['cohort','partition','arm','n',*recordkeys]);w.writeheader()
        for co,parts in summary['summaries'].items():
            for part,arms in parts.items():
                for arm,r in arms.items():w.writerow({'cohort':co,'partition':part,'arm':arm,'n':r['n'],**{k:r[k] for k in recordkeys}})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
    y=np.arange(len(COHORTS))
    for j,(arm,label,color) in enumerate([('previous_validation_best','Validation-tuned old model','#cf9e58'),('selected','Selected configuration','#197d83')]):
        vals=[summary['summaries'][c][p][arm]['hit_at_1']*100 for c,p,_,_ in COHORTS]
        bars=axes[0].barh(y+(j-.5)*.3,vals,height=.28,label=label,color=color);axes[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axes[0].set_yticks(y,[en for _,_,_,en in COHORTS]);axes[0].set_ylim(7.3,-.6);axes[0].set_xlim(0,105);axes[0].set_xlabel('Hit@1 (%)');axes[0].set_title('Head ranking across separate populations');axes[0].legend(loc='lower left',fontsize=8)
    labels=['All evidence @5','All evidence @10'];x=np.arange(2)
    for j,(arm,label,color) in enumerate([('bge','Public BGE','#8093a2'),('previous_validation_best','Old model, tuned','#cf9e58'),('selected','Selected','#197d83')]):
        vals=[fin[arm][m]*100 for m in ['all_evidence_at_5','all_evidence_at_10']]
        bars=axes[1].bar(x+(j-1)*.24,vals,width=.22,label=label,color=color);axes[1].bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    axes[1].set_xticks(x,labels);axes[1].set_ylim(0,100);axes[1].set_ylabel('Questions with every labelled fact retrieved (%)');axes[1].set_title('New FinQA evidence test: 1,147 questions / 278 reports');axes[1].legend(loc='upper left',fontsize=8)
    fig.suptitle('InsureRAG evidence reranking — frozen test results',fontsize=15)
    fig.supxlabel('Retrieval only. FinQA uses a supplied annual-report scope; earlier sets are historical regression checks.',fontsize=9)
    assets=ROOT/'docs/assets';fig.savefig(assets/'evidence_reranker_results.png',dpi=180);fig.savefig(assets/'evidence_reranker_results.svg');plt.close(fig)
    promoted=lock['new_training_promoted']
    conclusion='新训练通过了事先锁定的验证晋升规则，以下为一次固定选择后的测试。' if promoted else '新训练没有超过受约束的旧模型验证最优结果，保留旧模型作为最终配置；训练权重和全部负结果仍完整交付。'
    text=f'''# InsureRAG 首位排序与财报证据训练

{conclusion} 本轮选定 `{lock['config']['model']}`，重排融合权重 `{lock['config']['cross_weight']}`；旧模型也独立比较同一组三种权重，验证最优为 `{lock['best_previous_config']['cross_weight']}`。不能把调整融合参数的收益全部算成模型训练收益。

本轮增加 2,075 道实际进入梯度训练的原始财报问题，重新挖掘 367,338 个训练负例，完成两种子、各两轮训练。固定了上一轮查询编码器和 BGE 文档索引，训练重排器并在验证集比较三种融合权重；本次新旧最终融合权重均为 0.5，差异来自新重排权重。相对上一轮流程，候选深度与查询编码次数不变；相对单独 BGE，完整流程仍有两次查询编码及额外重排计算。

**固定测试的主要变化：** FAQ Hit@1 从 {faq['previous_validation_best']['hit_at_1']:.2%} 到 {faq['selected']['hit_at_1']:.2%}，{faqdiff['hit_at_1']['wins']} 题改善、{faqdiff['hit_at_1']['losses']} 题退步，差值及名义 95% 簇区间为 {interval(faqdiff['hit_at_1'])} 个百分点；这是重复使用的历史测试，不能当成全新独立验证。FAQ Hit@10 变化 {faqdiff['hit_at_10']['difference']*100:+.2f} 个百分点，{uncertainty(faqdiff['hit_at_10'])}。新财报完整证据@5 变化 {fd['all_evidence_at_5']['difference']*100:+.2f} 个百分点，{uncertainty(fd['all_evidence_at_5'])}。{regression_text}

## 测试结果

Hit@10 表示找到至少一条原始标注证据，并非生成答案正确率。公共 BGE 指 `BAAI/bge-small-en-v1.5`，不是所有 BGE 型号。

{chr(10).join(rows)}

![首位排序与完整证据检索](assets/evidence_reranker_results.png)

下表比较本轮选定配置与**旧模型的验证最优配置**。区间为 5,000 次配对簇 bootstrap，FAQ / FiQA 按共享标注答案，政府 / 通用按来源，FinQA 按年度报告。多指标、多集合比较未作多重检验校正；历史测试已经多次被查看。完整的同权重旧模型对照另存于 `test_metrics.csv` 和原始报告中。

{chr(10).join(cirows)}

## 新增测试：是否找齐回答需要的事实

新的 1,147 道 FinQA 问题来自 278 份年度报告、380 个原始页面和 100 家公司。每题给定公司 / 年度报告范围，检索该报告在数据集中发布的全部页面；不提供正确页面或正确行。这是本项目定义的报告范围证据检索任务，与 FinQA 官方单页输入、程序执行准确率不能直接横向比较。

原始测试包含 {len(duplicates['groups'])} 组同报告内重复问题，按忽略大小写及空白差异计为 {duplicates['unique_report_question_groups']:,} 个不同的“报告 + 问题”组合。全部原始记录保留；按问题组等权的次要敏感性结果中，前五完整证据命中率为旧模型验证最优 {duplicates['group_equally_weighted_metrics']['previous_validation_best']['all_evidence_at_5']:.2%}、本轮选定 {duplicates['group_equally_weighted_metrics']['selected']['all_evidence_at_5']:.2%}。不能把 1,147 条记录称为 1,147 个完全独立样本。

原题基于特定页面提问，扩展到整份报告的已发布页面后，可能出现指代不明确或其他页面存在相同事实的问题。主指标严格按原始证据 ID；若允许同一报告内完全相同的证据文本互相替代（只规范空白并去重），前五完整证据命中为旧模型验证最优 {fin_aliases['previous_validation_best']['all_evidence_at_5']['literal_text_equivalent']:.2%}、本轮选定 {fin_aliases['selected']['all_evidence_at_5']['literal_text_equivalent']:.2%}。这是标注敏感性检查，不是对语义等价或问题歧义的专家裁定。

{chr(10).join(finrows)}

相对旧模型验证最优，“前五找齐全部事实”的差值为 {interval(fd['all_evidence_at_5'])} 个百分点。按公司重采样的敏感性区间为 {interval(failures['finqa_company_cluster_sensitivity']['previous_validation_best']['metrics']['all_evidence_at_5'])} 个百分点；更粗的分组用于检查同公司不同年份之间的依赖。2 道题需要超过五条事实，因此前五完整命中的理论上限为 {failures['finqa_complete_at_5_theoretical_ceiling']:.2%}，同时报告前十指标。

候选报告的源行数最少 {failures['finqa_candidate_scope_statistics']['min']}、中位数 {failures['finqa_candidate_scope_statistics']['median']:.0f}、最多 {failures['finqa_candidate_scope_statistics']['max']}；各方法使用相同范围。多个事实题和单事实题分别统计，详见 `failure_analysis.json`。

其中 {strata['single_fact']['n']} 道只需一条标注事实，前五完整命中率由 {strata['single_fact']['arms']['previous_validation_best']['all_evidence_at_5']:.2%} 变为 {strata['single_fact']['arms']['selected']['all_evidence_at_5']:.2%}；{strata['multiple_facts']['n']} 道需多条事实，相应指标由 {strata['multiple_facts']['arms']['previous_validation_best']['all_evidence_at_5']:.2%} 变为 {strata['multiple_facts']['arms']['selected']['all_evidence_at_5']:.2%}。这些是事后分层描述，不用于再次选模型。

这些是财报分析题，并非新增 1,147 道保险业务题。仅 4 道新测试问题命中预先说明的保险关键词，规模不足以得出保险子域结论。{audit['actual_train_test_shared_companies']} 家公司在实际训练和测试的不同年份出现，所以这里只保证年度报告隔离，不宣称公司隔离。原始公开数据可能进入公共基础模型预训练，接触情况未知。

## 失败分析怎样影响了本轮设计

上轮 FAQ 的 38 道首位退步题中，24 道在固定重排器单独排序时标注答案仍为第一，14 道连重排器也排序错误。前者提示融合影响，但单纯加大重排权重在完整验证集上反而退步：FAQ Hit@1 在权重 0.5 / 0.75 / 1.0 时为 43.10% / 42.70% / 37.30%。因此没有根据局部案例直接手调一个测试最优权重。

重新挖掘训练候选后，5,914 / 11,629 道保险 FAQ、3,491 / 5,464 道 FiQA、856 / 2,075 道财报训练题存在未标注候选得分不低于最佳原始正例。它们不全是可靠错误：部分候选可能是漏标的合理回答。对非 FinQA 数据中的可疑负例降低权重，但不改成虚构正例；保留源标签和具体文本供审核。

训练目标在列表交叉熵外加入关注靠前名次的成对损失，并用旧重排器的 KL 约束减轻遗忘。名次权重使用交换正负例后的倒数排名差，停止梯度。每次呈现一个按轮次循环的原始正例和五个不同负例，已知其他正例及相同文本不会进入负例池。新增数据、挖掘方式与损失同时改变，因此结果属于组合方案，不能单独归因于某个因素或宣称新方法 SOTA。

本轮保留了 {failures['failure_cards']:,} 张失败和排名变化卡片：FAQ 等集合检查首位命中，FinQA 检查前五是否找齐事实。卡片含问题、原始标注、完整候选内正例名次、前五结果和旧 / 新对照。未删坏例、未改测试标签，案例由助手诊断，专家复核完成数为 0。

FAQ 仍有 {faqstage['candidate_missing']} 题的标注答案完全不在固定候选池中，另有 {faqstage['ranked_after_top10']} 题虽被召回却排在前十之后；单独重排无法补救前者。FinQA 的 {finstage['candidate_missing_required_fact']} 题候选中缺少至少一条原始标注事实，另有 {finstage['required_fact_ranked_after_top5']} 题候选中事实齐全但前五未找齐。多事实题完整证据命中只有 {strata['multiple_facts']['arms']['selected']['all_evidence_at_5']:.2%}，仍是明显短板。

以下是助手逐例检查的具体得失，不是随机抽样的错误率估计，也未进行专家业务裁定。原文和完整排名见[测试案例复核](../reports/evidence_reranker_v1/reviewed_test_cases.json)。

{chr(10).join(reviewrows)}

第一份 checkpoint（种子 42、第 1 轮）的[四个验证案例复核](../reports/evidence_reranker_v1/validation_case_review.json)进一步揭示两类限制：FiQA `fiqa_q_543` 的“配偶去世 / 独资企业”问题被排到第一的“经营亏损 / 税务”文章抢占，属于主题相关但事件条件不符；FinQA `finqa_q_5cc6e3adb6fc49d7d03f4088` 的两个数值行仍在前两位，表头却从第 3 变为第 6，造成严格完整证据指标失分。序列化行已包含列名与单位，原始标注完整性不完全等于最小充分语义证据。相应改进方向是有明确条件对照的监督，以及表头 / 单位 / 数值行关联的上下文组织；本轮没有根据这些案例再改训练或测试标签，也不据此声称已验证 GraphRAG 收益。

另外检查了“不同答案 ID、相同文本”的影响，只合并空白差异、保留大小写和标点。本轮 FAQ 首位命中按相同文本等价计算为 {aliases['selected']['hit_at_1']['literal_text_equivalent']:.2%}，相对原始 ID 指标多计 {aliases['selected']['hit_at_1']['additional_hits']} 题；旧流程多计 {aliases['previous_default']['hit_at_1']['additional_hits']} 题。这仅是标注敏感性检查，**不替换上面的主指标**，也不把语义相近的未标注答案自动当成正确答案。

## 数据、来源和实际训练

项目级唯一训练问题从 31,351 增至 **33,426**。新重排器使用 11,629 保险 FAQ、259 政府、13,999 通用、5,464 金融论坛和 2,075 财报问题。5,464 道 FiQA 以前训练过查询编码器，本轮首次用于该重排器，不能再次计成项目新增题。相对旧重排器，移除 100 道旧近重复题、加入 7,539 道题，净增 7,439。

原始 FinQA 为 6,251 / 883 / 1,147 训练、验证、测试问题。其官方划分不共享页面，但共享部分年度报告。排除训练中的留出年度报告、验证中的测试年度报告后，得到 2,351 / 496 / 1,147 题。另对训练去除 169 道留出文本重叠、51 道近重复、28 道重复问题，再排除 28 道缺少五个不同负例的问题，得到 2,075 道新训练题。原始一个 `text_-1` 索引异常训练记录在来源审查时排除。**测试 1,147 题全部保留。**

数据来自 [FinQA 官方仓库](https://github.com/czyssrs/FinQA)，固定提交 `0f16e2867befa6840783e58be38c9efb9229d742`，保留原 README、MIT 许可和逐文件哈希。支持事实由原数据标注；其背景见 [FinQA 论文](https://aclanthology.org/2021.emnlp-main.300/)。表格行全部从原始字段统一转换，不读取 `gold_inds` 来生成正例文本，也不使用上游已给出的检索结果或 `model_input`。原仓库曾修正过正负表格格式导致的标签泄漏，本项目加入了相应的不变性测试。

共有 86,421 条原始财报源行，其中 5,486 条没有字母数字，属于抽取噪声；没有把这些源行包装成高质量人工样本。总源索引为 192,231 条，各评估只搜索其声明的语料或报告范围。财报训练 / 验证 / 测试的标注问答对均未超过 512 tokens，但这不代表所有负例或其他语料不存在截断问题。

两次真实 CUDA 训练共 **{verification['total_optimizer_steps']:,} 次优化更新、{verification['total_query_presentations']:,} 次问题呈现、{verification['total_pair_presentations']:,} 次问答对呈现**，保存四份 epoch 权重。重复呈现不计为新样本。每个运行两轮，micro-batch 8 个问题、累积 4、学习率 3e-6，22,713,601 个参数。模型文件、参数实际变化、日志和步数均已独立核验。

两次训练循环合计 {sum(r['seconds'] for r in verification['training_runs'])/60:.1f} 分钟；不含数据准备、挖掘和评估。单次训练峰值已分配 GPU 显存最高 {max(r['peak_cuda_allocated_bytes'] for r in verification['training_runs'])/1024**3:.2f} GiB，不含缓存、驱动和其他程序占用。本机部分时间存在其他 GPU 负载，耗时不作为独占硬件基准。完整运行环境见 `environment.json`；记录随机种子不代表跨 GPU 环境保证逐位一致。

## 选择与复现边界

主规则在第一步梯度之前锁定：四份新权重及旧模型各比较三种融合权重，共 15 个配置。选择指标综合旧域 MRR 和新财报完整证据指标；FAQ 首位命中不能低于旧默认验证结果，并约束各域 Hit@10 与完整证据回退。先找出满足约束的旧模型最佳设置，新训练必须再超过它才能晋升。选定后只评分该新模型的测试，不在测试上挑种子或轮次。

全部验证候选如下，包含未通过约束的配置；本表不是测试结果：

{chr(10).join(vrows)}

此前 3,902 道历史题的旧流程逐题排名、Hit@1、Hit@10、MRR 和候选命中均完全复现；本轮增加 1,147 道财报题，总计分开报告 5,049 题，不能合并成一个“保险准确率”。新测试现已被查看，下轮应视作回归集，并继续引入独立来源。

FiQA 历史测试题 `fiqa_q_3446` 与旧重排器曾经用过的一道寿险训练题近似。该旧题已从当前梯度数据中排除，但继承权重仍保留历史接触，不能称为完全未见。主结果保留全部 648 题；排除这一题的 {inherited['n']} 题敏感性结果中，Hit@10 为旧模型验证最优 {inherited['summaries']['previous_validation_best']['hit_at_10']:.2%}、本轮选定 {inherited['summaries']['selected']['hit_at_10']:.2%}。FiQA 缺少可靠作者 / 原帖分组，使用共享标签聚类也不能消除全部依赖。

训练前的随机负例生成效率问题和去重后词法 ID 不匹配问题均保留前置代码、日志及相同向量哈希，未产生训练权重或测试分数。完整源证据检查、软件测试、真实查询入口和交付验证见 `data_verification.json`、`verification.json` 与 `delivery_checks.json`。

这仍是 research prototype：没有新增 Qwen 训练、GraphRAG 或 HNSW 效果结论，没有测程序计算、最终回答正确率或生产性能。使用、依赖与复现顺序见 [复现说明](EVIDENCE_RERANKER_REPRODUCTION.md)。本轮更新在本地，未推送 GitHub。

复核入口：[完整测试指标](../reports/evidence_reranker_v1/test_metrics.csv)、[失败和排名变化卡片](../reports/evidence_reranker_v1/failure_cases.json)、[验证选择记录](../reports/evidence_reranker_v1/selection.lock.json)、[真实训练核验](../reports/evidence_reranker_v1/verification.json)、[最终交付检查](../reports/evidence_reranker_v1/delivery_checks.json)。
'''
    (ROOT/'docs/EVIDENCE_RERANKER_UPDATE_ZH.md').write_text(text,encoding='utf8')
    for run in verification['training_runs']:
        parent=ROOT/f'../models/insurerag-evidence-reranker-v1-seed-{run["seed"]}'
        source=ROOT/'../models/ms-marco-MiniLM-L6-v2/README.md'
        if source.exists():shutil.copyfile(source,parent/'BASE_MODEL_CARD.md')
        card=[f'# InsureRAG evidence reranker v1 — seed {run["seed"]}', '',
              'Research fine-tune initialized from the condition-listwise seed123/epoch1 MiniLM reranker.',
              'Architecture: 22,713,601 parameters, single relevance logit, 512-token pair limit.',
              f'33,426 unique training questions; {run["optimizer_steps"]:,} actual optimizer updates and {run["pair_presentations"]:,} pair presentations across two epochs.',
              'Original BGE document encoder and prior 50/50 query adaptation remain fixed.',
              'The trained loss combines listwise CE, head-weighted pairwise softplus and prior-reranker KL retention.', '',
              '| Epoch | Weights SHA-256 | Changed parameter values |','| --- | --- | ---: |']
        card += [f'| {e["epoch"]} | `{e["weights_sha256"]}` | {e["changed_parameter_values"]:,} |' for e in run['epochs']]
        card += ['',f'Validation-selected final model/configuration: `{lock["config"]}`. Training promoted: {promoted}.',
                 'Use the query CLI and exact frozen document/query configuration described in InsureRAG-VLM/docs/EVIDENCE_RERANKER_REPRODUCTION.md.',
                 'Evidence selection only; no numerical-program execution, generated-answer accuracy or current insurance/legal correctness is established.',
                 'FinQA raw source/license and original supporting-fact annotations are preserved. Preserve prior InsuranceQA, FiQA, SQuAD and government source terms; no new unrestricted downstream license is asserted.',
                 'Historical validation/test reuse and unknown public-base-model pretraining exposure are disclosed in the experiment report.']
        (parent/'README.md').write_text('\n'.join(card)+'\n',encoding='utf8')
    marker='## Evidence reranker and financial-report evidence — October 3, 2026'
    readme=ROOT/'README.md';oldtext=readme.read_text(encoding='utf8');nextmarker='## Previous query-side adaptation and matched data ablation — October 2, 2026'
    oldtext=oldtext.replace('## Query-side adaptation and matched data ablation — October 2, 2026',nextmarker)
    newsection=f'''{marker}

The reranker now trains on **33,426 unique questions**, adding 2,075 original
financial-report questions to the project's training data and mining **367,338
negative pairs**. Two seeds × two epochs produced four checkpoints. The query
encoder, document vectors and candidate budget are fixed. Old-model fusion
weights are independently tuned before claiming a training gain.

Validation selected `{lock['config']['model']}`, cross weight {lock['config']['cross_weight']}.
New training promoted: **{str(promoted).lower()}**. Historical FAQ Hit@1 is
{faq['previous_default']['hit_at_1']:.2%} → {faq['selected']['hit_at_1']:.2%}; Hit@10 is
{faq['previous_default']['hit_at_10']:.2%} → {faq['selected']['hit_at_10']:.2%}.
These are reused historical tests; see all gains, regressions and clustered
intervals in the report.

The new **1,147-question / 278-annual-report FinQA evidence test** measures
retrieval of supporting facts within a supplied report scope. Complete evidence
in the first five results is {fin['previous_default']['all_evidence_at_5']:.2%} →
{fin['selected']['all_evidence_at_5']:.2%}; the old model tuned on validation scores
{fin['previous_validation_best']['all_evidence_at_5']:.2%}. This is not FinQA program
execution accuracy, a new insurance-only benchmark, or evidence of unseen-company
generalization. Public pretraining exposure is unknown.

See [measured results and failure analysis](docs/EVIDENCE_RERANKER_UPDATE_ZH.md),
[reproduction](docs/EVIDENCE_RERANKER_REPRODUCTION.md),
[raw evidence](reports/evidence_reranker_v1/), and the
[selected-model retrieval CLI](scripts/query_evidence_reranker.py).
This CLI is the entry point for these trained retrieval results; the interactive
document-QA application is a separate Qwen serving workflow. Generation quality
is not measured by this experiment. One inherited FiQA near-question exposure is
retained and disclosed with a 647-row sensitivity check.

'''
    if marker in oldtext:
        before=oldtext.split(marker,1)[0];after=nextmarker+oldtext.split(nextmarker,1)[1]
        oldtext=before+newsection+after
    else:oldtext=oldtext.replace(nextmarker,newsection+nextmarker,1)
    readme.write_text(oldtext,encoding='utf8')
    print(json.dumps({'report':'docs/EVIDENCE_RERANKER_UPDATE_ZH.md','selected':lock['config'],'training_promoted':promoted}))


if __name__=='__main__':main()
