# Qwen3.5 4B / 9B：接入已训练检索器后的本机开发诊断

日期：2026-10-04。基线：`31caa1d51ca724dddf20524cbfbed780390cce66`。

**结论：本轮不将 9B 替换为默认生成模型。** 在这台 RTX 4070 Laptop 8 GiB 上，9B Q4_K_M 的文本推理可以全部驻留 GPU，但速度更慢、显存余量更小；改进提示后，两者在小样本数值参考一致性和实际返回答案数量上持平。源值校验的算术诊断中，4B 的参考一致结果更多。这个结论只支持本机、当前任务配置下的选择，不代表 4B 普遍强于 9B。

应用仍默认 `local-extractive`；已有显式 Qwen3.5-4B 配置不变。本次不训练、不合并旧 PR、不修改权重、索引或发布资产。

## 实际使用的链路与模型

- 使用 main 的已选定检索配置：原始 BGE 文档索引、BGE 查询适配两遍编码并以 0.5 混合、dense top100 与正分 BM25 top100 的并集、seed123/epoch2 MiniLM 重排器，cross weight 0.5。输出 top5，不注入 gold。
- 启动前执行继承的代码／数据合约、固定编码器文件清单、索引和重排器指纹校验。语料索引 192,231 条；重排权重 SHA256 为 `697d3a725ad9d684c1b978365bda314114e8dbec2a27c2841a07f028ed2e81b8`。
- 从项目原有本地产物复制必要资产到独立工作区，没有使用 local-hashing 生成检索结果。打包／后处理对象使用 local-hashing 配置仅为避免初始化另一个检索后端；实际输入完全来自已校验的训练后检索结果。
- 使用 Ollama 0.34.2 的本机独立服务，每次仅驻留一个生成模型。此次为文本 RAG，没有评估图像能力。
- 官方模型：[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)、[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)。量化包来自 Ollama 模型库：[4B Q4_K_M](https://ollama.com/library/qwen3.5:4b)、[9B Q4_K_M](https://ollama.com/library/qwen3.5:9b)。不声称这些 GGUF 是 Qwen 官方发布的量化权重。

| 配置 | 4B | 9B |
|---|---|---|
| 包大小 | 3,389,983,735 bytes | 6,594,474,711 bytes |
| Ollama manifest digest | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| 文本运行时驻留量（Ollama 报告） | 3,128,038,521 bytes | 5,490,081,790 bytes |
| 改进提示后的解码速度中位数 | 70.53 token/s | 42.55 token/s |
| 改进提示后的热请求中位数 / P95 | 1.49 s / 2.43 s | 2.66 s / 5.04 s |

所有请求使用 context 4096、temperature 0、seed 42、thinking=false。自然语言对比均为最多 384 输出 token；算术结构化诊断均为 768。所有请求后的 Ollama 快照均显示全部 GPU 驻留，未观察到 CPU 卸载。9B 在全部诊断中的整卡使用快照最高为 7,620 MiB，包含桌面等其他进程；这些是请求后快照，不是连续测量的显存峰值，也不是生产延迟承诺。包大小包含的内容与文本运行时加载内容不同，因此不能以 6.6 GB 包大小直接推断显存需求。

## 样本与指标边界

使用固定哈希排序选择 20 条公开问题：FinQA dev 8、InsuranceQA valid 4、政府问答既有 test 4、FiQA valid 4。另有 6 条**合成**负对照：2 条删除全部上下文、2 条替换为另一份年报的证据、2 条向公开 FAQ 询问个人保单金额。全部样本均作为已暴露的开发诊断；既有 test 名称不代表本次 fresh holdout。未做独立专家审定或统计显著性检验。

FinQA 只向检索器／生成器传入问题和明确的公司／年度范围，不提供 gold 页、答案、执行程序或支持事实标签。原始答案只用于离线参考比较。财报范围包括同一公司／年度的全部已发布证据行；某些问题在失去原题页面上下文后存在业务口径歧义。

| 当前 top5 检索诊断 | 问题数 | 至少一个 gold 命中 | 全部 gold 命中 | 候选并集中至少一个 gold |
|---|---:|---:|---:|---:|
| FinQA | 8 | 8 | 6 | 8 |
| InsuranceQA | 4 | 1 | 0 | 3 |
| 政府问答 | 4 | 4 | 4 | 4 |
| FiQA | 4 | 2 | 1 | 2 |

未命中原始标签不等于所有检索内容都无关；FAQ 可能有多个语义相关回答。反过来，命中 gold 也不等于模型已答对。完整多 gold macro recall 与逐题记录见 `summary.json`；这里没有用 Hit 替代 Recall，也没有把 top5 结果冒充 @10。

## 两轮自然语言回答

第一轮保持 main 的提示与打包方法；第二轮仅在独立研究入口增加真实来源页／报告信息、限定简洁回答，并明确允许依据已给值进行算术。没有放宽主应用的保单证据校验。

| 26 条诊断 | main 提示 4B | main 提示 9B | 改进提示 4B | 改进提示 9B |
|---|---:|---:|---:|---:|
| 8 条 FinQA 主答案数值参考一致 | 1 | 3 | 3 | 3 |
| 20 条公开问题最终非拒答 | 1 | 2 | 3 | 3 |
| 6 条合成负对照最终拒答 | 6 | 6 | 6 | 6 |
| 输出截断 | 1 | 1 | 0 | 0 |

“数值参考一致”是人工从原始输出抄录主答案后计算的开发指标：百分数误差不超过 0.1 个百分点、小数计数误差不超过 0.01、整数精确匹配；缺失或互相矛盾的主答案计失败。它不是官方 FinQA 程序执行准确率，不等于完整答案正确率。具体判断和争议记录在 `numeric_review.json`，例如 13.44% 可通过粗粒度参考阈值，但并非所列操作数的精确计算。

原始答案、最终答案、引用、修复标志、拒答原因、截断标志和运行元数据分别保存。main 的单引用保单校验会拒绝一些多行财报计算和普通问答；也确实拦截了错误数值。因此“最终非拒答数”只是服务行为，不被当作准确率。改进提示两组的非拒答案均没有后处理修复；baseline 9B 有 1 次修复，详见原始记录。

改进提示后，20 条公开问题的两组输出均包含已知来源 ID，未发现未知 corpus ID；这只检查 ID 是否属于给定上下文，**不证明陈述被来源支持**。例如 FiQA 非 ATM 提现问题，两模型对缺失证据仍给出相反结论；不能据此宣布真实业务质量合格。

## 有来源校验的算术诊断

自然语言输出出现了明确的算术错误和口径错误，因此增加了独立实验：让模型选择操作、逐项数值、来源 ID 和原文短引，再由有界 Decimal 运算验证。只支持 identity、sum、mean、difference、ratio、percent_ratio、percent_change，不执行模型代码。操作数必须出现在引用的真实片段中，且报告范围一致；未知来源、虚构短引、零分母、错误参数数量和截断均拒绝。

第一版输出 schema 允许任意字符串，模型多次输出 `{339.9}` 一类非十进制值，按原规则拒绝并保留结果。第二版增加十进制字符串约束后，同一批 8 条财报问题得到：

| 第二版算术诊断 | 4B | 9B |
|---|---:|---:|
| 操作数来源与算术验证通过 | 6/8 | 6/8 |
| 计算结果与参考一致 | 6/8 | 4/8 |
| 验证通过但参考不一致 | 0 | 2 |
| 4 条合成缺证据对照主动拒答 | 4/4 | 4/4 |

9B 的两条偏差分别是：合计证券数仅选取一个分量；平均收入混合不同业务分部。运算本身正确，业务口径不正确。这正是输出中始终保留 `semantics_verified: false` 的原因。该诊断**没有接管应用答案，也不能把 6/8 写成上线问答准确率或模型训练提升**；它在已见开发样本上使用了不同输出格式和预算，与自然语言轮次不是单变量对照。

## 验证、复现与后续边界

最终完整离线测试：339 passed、58 subtests passed；覆盖未知引用、个人保单边界、截断、打包、算术输入来源、错误范围、数值格式、零分母及 Unicode 读取。测试没有下载模型。初次沙箱测试的两个临时目录权限错误在获准的正常执行环境重跑后消失，没有修改测试断言来绕过失败。

回放检查发现初版 runner 在 Windows 上隐式使用 CP936 读取 UTF-8 JSON，导致 `fiqa_q_1753` 的一个不换行空格变成汉字。现已显式指定 UTF-8，并对同一批 26 条题完整重跑四个自然语言实验臂；本报告表格来自 `utf8_*` 轮次。8 条 FinQA 的 32 个自然语言原始答案与此前逐字一致；全部 12 条算术诊断输入在两种解码下相同，因此算术诊断未受该问题影响。旧输出仅在本地保留，不混入最终自然语言统计。最终 104 条输出的提示及服务后处理离线回放完全一致，没有模型调用。

另外，实时 CLI 暴露了一个后处理缺陷：“什么是免赔额”的正确概念解释被金额修复逻辑改成证据中的金额示例。现在仅对完整、无个人保单限定的定义句型免除金额要求；个人金额和混合提问保留原检查。修复前后使用相同检索结果且模型原始答案逐字一致，修复后保留概念解释、不再标记金额修复。

先按 `docs/EVIDENCE_RERANKER_REPRODUCTION.md` 恢复公开检索资产和模型，再启动包含准确 digest 的本地 Ollama 实例。安装 `requirements-scale.txt` 与 `requirements-dev.txt` 后，可执行：

```powershell
python scripts/answer_evidence_reranker.py --question "What is an insurance deductible?" --corpus insuranceqa --device cuda --model ollama:qwen3.5:4b --base-url http://localhost:11434 --expected-digest 2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd --prompt-mode research --output answer.json
python scripts/summarize_qwen35_grounding.py --root reports/qwen35_main_comparison
python -m pytest -q
```

`compare_qwen35_grounding.py prepare --finqa-dev <original FinQA dataset/dev.json> --device cuda --output <new-dir>` 可重新准备固定样本。`run --data <new-dir> --model <tag> --expected-digest <digest> --base-url <local-url> --prompt-mode main|research --output <new-run>` 执行自然语言对比。`diagnose_qwen35_arithmetic.py` 使用相同 data/model/digest/base-url/output 参数运行独立算术诊断。输出目录必须不存在，防止覆盖历史记录。

完整复现时将 `<new-dir>` 设为 `reports/qwen35_main_comparison/dev`，并按以下名称保存输出：`utf8_baseline_4b`、`utf8_baseline_9b` 使用 `--prompt-mode main`；`utf8_revised_4b`、`utf8_revised_9b` 使用 `--prompt-mode research`；最终算术实验输出为 `arithmetic_v2_4b`、`arithmetic_v2_9b`。这些目录均放在 `reports/qwen35_main_comparison/` 下。从 `published/numeric_review.json` 复制审阅清单到其父目录，再运行汇总脚本。复现若产生不同原始主答案，应重新审阅，不能机械沿用原清单；零温度与 seed 并不保证跨硬件逐字相同。第一版拒绝输出的算术记录是可选历史审计材料，汇总不会要求重新生成它们。

发布的轻量证据位于 `reports/qwen35_main_comparison/published/`：统计、逐题数值审阅、ID 清单、模型配置、验证结果，以及本地完整证据文件的大小和哈希。完整模型输出、输入语料片段、权重、索引、日志均保留在本地，不在 PR 中。`export_qwen35_summary.py --root reports/qwen35_main_comparison` 可从本地复现结果导出同样的精简审阅包。上面的汇总／回放命令需要先重建本地输入和实验输出；克隆后的轻量包本身不含这些大文件。

数据使用范围核对：InsuranceQA 上游声明仅供研究使用；FiQA 数据卡声明 CC-BY-SA-4.0；FinQA 上游代码许可证为 MIT，年报内容仍保留其原有权利；政府来源亦保留原出处与日期。本 PR 只新增派生统计、来源 ID／哈希及自写合成单元测试，不重新发布这些语料或改变其许可。出处见现有各 benchmark 目录的 `provenance.json`、`manifest.lock.json`、`UPSTREAM_README.md`／`UPSTREAM_LICENSE`。

初始自然语言集成提交 `a1ad313`；来源提示与第一版算术提交 `a2ade4f`；最终 schema、UTF-8 修正和定义问题修复由当前提交提供。

后续有价值的方向是：保留表格业务标题与单位、区分年际平均与跨分部平均、在多引用语义校验设计完成后独立验证财报服务路径，以及补充未参与调试的新问题。当前证据不足以支持长时间微调、替换默认模型或生产效果声明。
