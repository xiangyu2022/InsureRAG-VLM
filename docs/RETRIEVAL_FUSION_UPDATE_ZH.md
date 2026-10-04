# 检索融合与语义重排：2026-10-01

在同一份 InsuranceQA V2 历史测试集的全部 2,000 道题、同一批 27,413 个候选答案上，新方案 Hit@10 为 **69.55%**，高于 BGE 的 **67.50%**；Hit@1 从 **34.45%** 提升至 **37.35%**。这是 FAQ 检索实验，不能转换成 Qwen 问答准确率、GraphRAG 收益或保险责任判断准确率。

## 原 RRF 为什么更差

原实验对 BM25 和 BGE 使用等权 RRF、k=60、每路 top 100。BM25 的 Hit@10 为 48.25%，BGE 为 67.50%，两路能力并不相同，但排名贡献相同。

只在 BGE 排第一的答案得分为 1/61≈0.01639；两路都排第 30 的答案得分为 2/90≈0.02222。因此，同词匹配较多、两路排名中等的答案可能超过很强的单路语义命中。RRF 同时丢弃了原始相似度间距。

逐题对照发现：旧 RRF 相对 BGE 救回 **98** 题，损失 **189** 题，净损失 **91** 题，解释了 67.50%→62.95% 的全部下降。在验证集上也出现同方向结果：救回 111、损失 214。诊断记录见 [historical_diagnosis.json](../reports/insuranceqa_v2/fusion_dev_v1/historical_diagnosis.json)。

例如验证题 `How Much Does Medicare Pay For A Ct Scan?`：标准答案之一在 BGE 排第 6、BM25 不在前 100，经 RRF 后落到第 25；另一个标准答案分别排第 13、23，融合后排第 11。由此从“前十有正确答案”变成失败。答案标签只用于事后诊断，不用于构造候选。

## 修改后的方法

1. BGE 和 BM25 各取前 100，合并候选，最多 200 个；BM25 只纳入正分结果。
2. 在同一候选集合内分别对两路原始分数做 min-max 归一化，用 `0.8 × dense + 0.2 × lexical` 选出 100 个候选。
3. 使用公开的 [MS MARCO MiniLM-L6-v2 cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) 对问题与候选文本联合打分。模型约 2,270 万参数，锁定 revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`。本项目没有对它进行 InsuranceQA 微调。
4. 在选中的 100 个候选内重新归一化三路分数，最终为 `0.5 × (0.8 × dense + 0.2 × lexical) + 0.5 × cross_encoder`。

这些分数是排序分数，不是答案正确概率。最终方法包括模型重排，不能把收益表述为“原来的 RRF 已经优于 BGE”。两阶段检索的机制可参考 [Sentence Transformers 官方说明](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)。

## 选择过程与完整结果

先用全部 2,000 道验证题比较 30 种简单融合配置，再比较 24 种候选范围、融合权重及 cross-encoder 权重配置。主要选择指标为 Hit@10，平局时依次比较 MRR@100、Hit@1。最终配置在新方案测试推理前写入 [selection.lock.json](../reports/insuranceqa_v2/rerank_selection_v2/selection.lock.json)。没有测试后改参数，也没有给候选列表追加标准答案。

| 同一历史测试集，n=2,000 | Hit@1 | Hit@10 | MRR@100 |
| --- | ---: | ---: | ---: |
| BM25 | 23.00% | 48.25% | 0.3144 |
| 原等权 RRF | 31.40% | 62.95% | 0.4184 |
| BGE | 34.45% | 67.50% | 0.4541 |
| 归一化融合，不使用 cross-encoder | 35.45% | 68.25% | 0.4654 |
| BGE＋相同 cross-encoder，100 候选 | 36.05% | 69.50% | 0.4729 |
| 仅 cross-encoder 排联合候选 | 37.00% | 67.85% | 0.4756 |
| **验证集选定的融合＋cross-encoder，100 候选** | **37.35%** | **69.55%** | **0.4839** |

新方案相对 BGE：Hit@10 增加 **2.05 个百分点**，净增加 **41** 道命中（80 改善、39 退步）；Hit@1 增加 **2.90 个百分点**。共享标准答案的题目按连通组进行配对 bootstrap，5,000 次、seed=42，Hit@10 差值的 95% 区间为 **+1.00 至 +3.11 个百分点**。

相对“BGE＋相同重排器”，新方案 Hit@10 只增加 **0.05 个百分点**，即净增加 1 题（28 改善、27 退步），差值区间 **−0.65 至 +0.80 个百分点**。因此主要收益来自重排，不能声称混合检索在 Hit@10 上稳定优于更简单的 BGE＋重排。Hit@1 的点估计高 1.30 个百分点。

排除之前独立标记的 476 条词面近重复题后，剩余 1,524 题：BGE Hit@10 **68.77%**，新方案 **70.80%**。排除 8 条精确重复题后分别为 **67.52%** 与 **69.53%**。这些是预有审计切片，未用于本轮选参，也不是人工语义去重。

分领域结果未隐藏：汽车保险从 161/249 降到 160/249，退休计划从 73/95 降到 72/95；其他领域持平或提高。完整领域、命中与召回指标、预测排名均保存在 [summary.json](../reports/insuranceqa_v2/rerank_test_evaluation_v1/summary.json) 与相邻 `predictions.jsonl`。

## 实际运行与开销

验证集实际计算 348,855 个问题—候选对，测试集计算 348,026 对。为复用分数完成各个预设对照，记录了整个候选并集的分数；最终方案只在筛选的 100 候选内排序。RTX 4070 Laptop GPU、float16、最大 512 tokens；测试评分约 150 秒，其中 cross-encoder 约 125 秒。这是离线批处理，排除了预先生成 BGE embedding 及模型加载时间，不能当作在线延迟。

CPU、CUDA 命令行均已用自由输入 `How does a deductible differ from a copayment?` 跑通，返回相同的 3 条可检查原始 FAQ 文本，没有调用生成模型。包含加载与建立稀疏索引耗时分别约 27.4 秒、7.5 秒；这是功能 smoke，不是新的准确率样本。

原始 BGE、BM25 的全部验证/测试前 100 排名经过逐行一致性检查。初次评分使用批量矩阵乘法，浮点近似造成 85 道验证题的近似同分排序交换，前十候选集合均未改变；一致性门禁仍在选参前拒绝该次结果。现有结果使用与基线完全相同的矩阵—向量计算。被拒绝的尝试及说明保留在 `rerank_valid_v1`、`rerank_selection_v1`，不作为最终成绩。

## 复现

依赖沿用 `requirements-scale.txt`。模型与 embedding 缓存不随源码包分发；先安装适合本机的 PyTorch。原 BGE checkpoint、CLS pooling、512-token 上限和 query instruction 均沿用原冻结实验。

```powershell
python scripts/acquire_minilm_reranker.py --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --output models/ms-marco-MiniLM-L6-v2
# 如无原始 embedding 缓存，先生成新的基线目录。
python scripts/eval_insuranceqa_scale.py --model models/bge-small-en-v1.5 --device cuda --output reports/insuranceqa_v2/my_baseline
python scripts/score_insuranceqa_reranker.py --baseline reports/insuranceqa_v2/my_baseline --split valid --model models/ms-marco-MiniLM-L6-v2 --device cuda --output reports/insuranceqa_v2/my_valid_scores
python scripts/eval_insuranceqa_reranker.py --baseline reports/insuranceqa_v2/my_baseline --scores reports/insuranceqa_v2/my_valid_scores --select --output reports/insuranceqa_v2/my_selection
python scripts/score_insuranceqa_reranker.py --baseline reports/insuranceqa_v2/my_baseline --split test --model models/ms-marco-MiniLM-L6-v2 --device cuda --selection reports/insuranceqa_v2/my_selection/selection.lock.json --output reports/insuranceqa_v2/my_test_scores
python scripts/eval_insuranceqa_reranker.py --baseline reports/insuranceqa_v2/my_baseline --scores reports/insuranceqa_v2/my_test_scores --selection reports/insuranceqa_v2/my_selection/selection.lock.json --output reports/insuranceqa_v2/my_test_results
python scripts/query_insuranceqa.py --question "How does a deductible differ from a copayment?" --embedding-model models/bge-small-en-v1.5 --reranker-model models/ms-marco-MiniLM-L6-v2 --baseline reports/insuranceqa_v2/my_baseline --selection reports/insuranceqa_v2/my_selection/selection.lock.json --scoring-run reports/insuranceqa_v2/my_valid_scores --device cuda
python -m pytest -q
```

各命令使用新输出目录，避免覆盖旧记录。支持 CPU，但耗时不同。可复用当前电脑已有的 `../models/...` 和 `retrieval_frozen` embedding 缓存。评测代码、模型文件、数据集、评分与参数选择均记录 SHA-256。

随包冻结结果可用 `python scripts/verify_retrieval_fusion.py` 验证，无需模型或 embedding 缓存。它核对数据、代码、验证/测试模型一致性、候选集合、所有题目覆盖，并从原始排名独立重算 Hit@1、Hit@10、MRR@100。完整测试通过 257 项、58 项子测试；不安装可选模型/评测依赖的环境通过 252 项、跳过 5 项可选测试、58 项子测试通过。

## 结论范围

历史测试集此前已被查看；本轮新增方案只用验证集选择，并在测试前冻结，属于历史测试回归，不是全新盲测。领域、近重复与预训练暴露仍限制泛化结论。模型接受答案文本联合编码会增加算力开销。

本次新增的是可运行的 FAQ 检索入口及重排模块；PDF 文档 QA、GraphRAG 与 Qwen 的历史结果没有随之更新。之前发现的金额归属校验、错误版本配对、词典路由和图上下文挤出问题仍需分别修复。数据继续遵守 InsuranceQA 作者的 research-only 使用条款。
