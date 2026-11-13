# V2 最终配置、完整结果与错误分析

本报告冻结 Amazon Reviews 2023 `Musical_Instruments` 的 V2 主线。所有正式数字均来自
真实数据全量 Test，`synthetic_data_used=false`；参数只在 Validation 上选择，Test
不参与选参。三个独立训练种子为 2026、2027、2028，区间为双侧 95% Student-t 区间。

V2 核心 Test 证据快照为 `V2.3-pr-retraining-final`。V2.4 增强线先以
`v2.4-enhancement-final` 冻结 Validation 决策，随后按冻结配置执行一次性 Test 确认，并以
`v2.4-post-freeze-test-final` 收口。两个新方案均未通过升级门槛，因此不改写 V2.3 的
默认 Test 结论。V2.3 同时冻结主排序、长尾、跨品类、
Preference Reasoning 分层解释、冻结 checkpoint 细粒度干预，以及三个细粒度替代结构的
三种子独立重训结果。此前 `V2.2-pr-inference-freeze` 保留为重训前快照；最终机制结论以
V2.3 的公平重训结果为准，V2.4 只补充负结果、数据准备状态与停止边界。

## 最终决策

- 默认线上方案：`token_loss_010`，即行为强度 V2 + Semantic-ID token inverse-frequency
  loss `alpha=0.10` + Beam-20/Top-10，不使用 MMR 或候选重排。
- 实验方案：温和 BLaIR+频率重排。它保留 99.45% 的整体 Recall@10 并显著扩大目录覆盖，
  但 Observed-Tail Recall@10 增量的三种子区间跨 0，不能作为稳定命中提升结论。
- 长尾增强模式：激进软配额，仅在业务明确接受整体 Recall/NDCG 损失时启用。它的尾部
  Recall@10 和目录覆盖最高，但不是默认准确率方案。
- Preference Reasoning：保留完整模型作为冻结默认配置，但不把它解释为结构显著优于
  简化版本。当前证据支持“近期事件对长历史用户有稳定预测贡献”；learned gate、event
  attention 和 Fusion MLP 在分别独立重训后均无稳定 Recall@10 优势，不能宣称每个推理
  组件均带来不可替代的增益。
- V2.4 增强线：冻结后 Test 确认动态遗忘 Recall@10 只保留原 V2 的 96.08%，配对区间
  仍跨 0；learned hybrid 只保留 56.69%，且配对区间稳定为负，正式否决。UniSRec 因本地
  资源限制主动停止且无正式结果。故原 V2 `token_loss_010 + Beam-20/Top-10` 仍是最终
  默认方案。
- V2.5 增强线：baseline-preserving bounded residual 在三种子 Validation 上保留
  99.961% 的总体 Recall@10，但 Observed-Tail Recall 与覆盖增量区间均跨 0，正式不升级；
  V2.5 未读取 Test，不改变原 V2 默认结论。
- V2.6 增强线：全量 Train 普查得到 3,310 条真实 Observed-Tail hard positives，但
  seed 2026 的有界残差在 Validation 没有新增任何尾部 Top-10 命中，阶段 2 失败并按协议
  停止；五种子与 Test 均未运行。原 V2 默认结论不变。
- V2.7 增强线：无硬上限 listwise scorer 在保守 scale 下只净增 1 个尾部命中且 CI 跨 0；
  较强 scale 虽有稳定尾部增益，却使总体 Recall 保留率降至 45.0% 或 3.5%。没有方案通过
  联合门禁，多种子与 Test 均未运行，原 V2 默认结论不变。

冻结文件：

- 默认训练与解码：[v2_final_default.yaml](../configs/v2_final_default.yaml)
- 三种部署模式：[v2_final_modes.yaml](../configs/v2_final_modes.yaml)

## 数据与模型配置

| 项目 | 冻结值 |
|---|---|
| 数据集 | McAuley-Lab/Amazon-Reviews-2023，`Musical_Instruments` 去重 5-core |
| 切分 | leave-last-out；339,519 / 57,439 / 57,439 |
| 商品目录 | 24,587 个真实商品 |
| 商品向量 | `hyp1231/blair-roberta-base`，CLS pooling，L2 normalized，24,587 × 768 |
| RQ-VAE | 3 层 × 256 codes，K-means 初始化，EMA=0.99，50 epochs |
| 生成器 | 2 层 encoder + 2 层 decoder，dim=128，4 heads，10 epochs |
| 用户行为 | 真实负向/中性/正向评分、`rating/5` 强度、真实时间差 log-day bucket |
| 默认长尾训练 | Semantic-ID token loss `alpha=0.10`；sampling/item loss alpha=0 |
| 默认解码 | Catalog Trie constrained Beam-20，20 candidates，Top-10，MMR=0 |
| 流行度分层 | Head=训练出现商品 Top 20%；Torso=中间 30%；Observed Tail=Bottom 50%；Cold-start=训练频次 0 |

## 三种子完整 Test 表格

除延迟外，表中数值均为百分比，格式为 `均值 [95% CI 下界, 上界]`。Student-t 区间未
裁剪到 `[0, 100%]`，因此极低概率指标的下界可能略小于 0。

### 准确率

| 方案 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| 原 V2 | 2.4890 [2.3874, 2.5907] | 1.6159 [1.5670, 1.6648] | 4.0002 [3.9341, 4.0663] | **2.1012 [2.0757, 2.1267]** |
| **token-loss 0.10（默认）** | **2.5186 [2.4589, 2.5783]** | **1.6166 [1.5314, 1.7019]** | **4.0019 [3.6935, 4.3103]** | 2.0927 [2.0539, 2.1315] |
| BLaIR+频率温和方案 | 2.5157 [2.4482, 2.5833] | 1.5968 [1.5368, 1.6568] | 3.9799 [3.7565, 4.2032] | 2.0664 [2.0231, 2.1098] |
| 激进软配额 | 2.4884 [2.3959, 2.5809] | 1.5785 [1.5222, 1.6348] | 3.8209 [3.7008, 3.9409] | 2.0085 [1.9818, 2.0352] |

### 长尾、覆盖与效率

| 方案 | Tail R@5 | Tail R@10 | 总目录覆盖 | Tail 目录覆盖 | Tail 推荐占比 | ms/样本 |
|---|---:|---:|---:|---:|---:|---:|
| 原 V2 | 0.0000 [0.0000, 0.0000] | 0.0138 [0.0019, 0.0257] | 2.4945 [2.2520, 2.7371] | 0.3289 [0.2003, 0.4576] | 0.4456 [-0.1314, 1.0225] | 24.8414 [21.8946, 27.7881] |
| **token-loss 0.10（默认）** | 0.0055 [-0.0064, 0.0174] | 0.0166 [0.0166, 0.0166] | 2.6166 [2.3960, 2.8371] | 0.3806 [0.2272, 0.5340] | 0.4258 [0.0934, 0.7583] | 24.9968 [22.0060, 27.9876] |
| BLaIR+频率温和方案 | 0.0110 [-0.0008, 0.0229] | 0.0387 [0.0072, 0.0701] | 8.4842 [4.4819, 12.4864] | 11.4718 [3.6236, 19.3199] | 1.2967 [-0.1431, 2.7364] | 25.8352 [25.6491, 26.0213] |
| **激进软配额（长尾模式）** | **0.0110 [-0.0008, 0.0229]** | **0.1270 [0.0795, 0.1745]** | **15.0310 [12.4355, 17.6265]** | **24.5610 [19.5541, 29.5678]** | **6.2557 [1.6445, 10.8669]** | 24.2138 [22.4918, 25.9359] |

绝对延迟来自不同正式运行，受机器负载影响，不应据此声称激进方案更快。同一次配对运行
内，温和方案相对其基线增加 `0.915 ms/样本`，激进方案增加 `0.863 ms/样本`。

## 验收结果

温和方案锁定参数为 BLaIR weight `0.25`、frequency alpha `0.10`、`tail_20` soft quota、
strength `0.01`。

- 整体 Recall@10：4.0019% → 3.9799%，保留率 99.45%，通过 97% 门槛；三个种子分别
  保留 98.45%、99.47%、100.49%。
- Observed-Tail Recall@10：0.0166% → 0.0387%，均值相对增加 133.3%，但配对增量
  95% CI 为 `[-0.0093, +0.0535]` 个百分点，跨 0，未通过预注册门槛。
- NDCG@10：2.0927% → 2.0664%，配对增量区间为
  `[-0.0501, -0.0024]` 个百分点，显示小幅稳定下降。
- 总目录覆盖增加 5.87 个百分点，Observed-Tail 目录覆盖增加 11.09 个百分点；两者的
  三种子增量区间下界均大于 0。

因此正确表述是：“温和重排稳定扩大目录与尾部目录覆盖，同时基本保留整体 Recall；尾部
命中均值提高，但三种子证据不足以确认稳定提升。”

## 为什么候选召回没有转化为 Top-10 命中

错误分析单位是 `seed-example pair`。三个种子各包含 12,072 个 Observed-Tail Test 目标，
合计 36,216 个；用户标识在案例文件中只保留 SHA-256 前缀。

### 两级瓶颈漏斗

| 阶段 | 包含目标数 | 相对全部尾部目标 |
|---|---:|---:|
| Observed-Tail 目标 | 36,216 | 100.0000% |
| Generator Beam-20 候选 | 15 | 0.0414% |
| BLaIR Tail@50 候选 | 1,860 | 5.1360% |
| 两路候选并集 | 1,867 | 5.1552% |
| 最终 Top-10 | 14 | 0.0387% |

第一层仍是候选缺失：34,349 个目标（94.84%）不在两路候选并集中。第二层是条件排序失败：
并集已包含的 1,867 个目标中，1,853 个未进入 Top-10，候选到 Top-10 的转化率只有
`14 / 1,867 = 0.75%`。

### 融合结构偏向生成候选

温和方案分数为：

```text
0.75 × generator_relevance
+ 0.25 × BLaIR_relevance
- 0.10 × normalized_log_frequency
+ soft_quota_bonus(strength=0.01)
```

BLaIR-only 候选的 `generator_relevance=0`。三种子共有 1,852 个 BLaIR-only 目标进入候选，
只有 7 个进入 Top-10，转化率 0.38%；generator-only 为 3/7，两个来源都命中为 4/8。
后两组样本很小，不能估计稳定比例，但清楚显示“是否得到生成器分数”决定了融合排序的量级。
0.25 的语义权重和 0.01 的软配额是为保护整体 Recall 而设，通常不足以把 BLaIR-only
商品推过高分生成候选。

这一问题甚至发生在强语义候选上：BLaIR 排名 1–5 的 423 个尾部目标只有 7 个进入
Top-10，转化率 1.65%；排名 11–20 的 387 个目标没有一个进入 Top-10。命中目标的 BLaIR
排名中位数为 2，候选存在但排序失败目标的中位数为 16，说明 BLaIR 排名有信号，但当前
融合函数没有充分利用。

### 稀疏命中导致统计区间很宽

token-loss 基线每个种子都只有 2 个尾部 Top-10 命中；温和方案为 5、3、6 个，即增量
分别为 3、1、4。三种子合计没有丢失基线尾部命中，新增 8 个，但 McNemar 的每种子
discordant pairs 太少，p 值为 0.25、1.0、0.125。`n=3` 的 Student-t 临界值为 4.303，
因此即使三个种子方向都非负，增量区间仍会跨 0。

### 不是简单的“历史太短”或“尾部内部频次更低”

候选缺失、候选存在但排序失败、最终命中三组的目标训练频次中位数分别为 4、4、4.5；
历史长度中位数分别为 6、5、5。差异不足以支持单一的历史长度或尾部内部频次解释，且最终
命中只有 14 个，不能作强因果归因。主要证据仍指向候选覆盖上限和融合分数量级不匹配。

### 真实案例

- 排序失败：`B08L9RLVNH`（Epiphone Les Paul Starter Pack）是 BLaIR rank 1、相似度
  0.9421 的真实尾部目标，但不在 generator 候选，最终未进入 Top-10。
- 排序失败：`B000I1YLG0`（Denon DL-110 Cartridge）是 BLaIR rank 1、相似度 0.9708；
  历史中包含多个唱头商品，仍因缺少 generator relevance 未进入 Top-10。
- 新增命中：`B004KCLIFA`（Audio-Technica DJ Headphones）是 BLaIR rank 1，历史为高度
  一致的专业耳机，最终从基线未命中提升到 rank 9。
- 新增命中：`B00WR8M5MO`（Lava Cable Pedal-Board Kit）是 BLaIR rank 2，历史中包含同
  系列 Lava Cable，最终从基线未命中提升到 rank 9。

完整匿名案例及前五个 BLaIR 竞争商品见
[`error_cases.jsonl`](../artifacts/v2_final_report/error_cases.jsonl)。

## Preference Reasoning 证据与边界

三组实验使用同一真实 Amazon Reviews 2023 / Musical Instruments 数据、Beam-20/Top-10
口径和 seed 2026/2027/2028，但回答的问题不同：

1. `full vs mean-pool` 是分别独立训练的粗粒度结构消融；
2. 历史长度分层与事件遮蔽是冻结模型内的预测敏感性分析；
3. fixed gate、no event attention 和 no fusion MLP 是同一 full checkpoint 内的细粒度机制
   干预，用于判断当前已训练模型是否依赖相应计算路径；
4. 同三个变体各自从随机初始化独立训练 10 epoch，才用于最终的公平架构容量比较。

### 独立重训的粗粒度消融

| Split | 指标 | Full | Mean-pool | Full−Mean-pool，配对 95% CI |
|---|---|---:|---:|---:|
| Validation | Recall@5 | 0.028215 | 0.026091 | +0.002124 [+0.000221, +0.004027] |
| Validation | Recall@10 | 0.044436 | 0.042503 | +0.001932 [-0.000209, +0.004074] |
| Test | Recall@5 | 0.025186 | 0.023753 | +0.001433 [-0.000299, +0.003166] |
| Test | Recall@10 | 0.040019 | 0.038418 | +0.001602 [-0.000884, +0.004087] |

Validation 的 Recall@5/NDCG@5 增量区间下界大于 0，但四项 Test 增量区间均跨 0。因此
只能表述为“完整推理在 Test 均值上较高”，不能宣称对 mean-pool 有稳定测试优势。

### 历史长度分层与遮蔽

Test 中 history≤10、11–20、>20 的样本分别为 47,552、7,466、2,421。短历史中长期和
短期窗口包含同一批事件，long/short cosine=1 是结构决定的，不能作为兴趣迁移证据。

| History | Full R@10 | Mean-pool R@10 | ΔR@10，95% CI |
|---|---:|---:|---:|
| ≤10 | 0.043104 | 0.041218 | +0.001886 [-0.001194, +0.004965] |
| 11–20 | 0.026788 | 0.026431 | +0.000357 [-0.001980, +0.002694] |
| >20 | 0.020240 | 0.020377 | -0.000138 [-0.002720, +0.002445] |
| >10 | 0.025185 | 0.024949 | +0.000236 [-0.002152, +0.002624] |

history>10 时移除最近 10 条事件使 Recall@10 稳定下降 0.005597，95% CI
`[0.004307, 0.006886]`，说明近期上下文具有稳定预测贡献。history>20 时只保留最近 10 条
反而比完整历史高 0.002341，95% CI `[0.001748, 0.002933]`；较早事件在超长序列上平均
造成干扰，后续应优化动态窗口或遗忘机制，而不是无条件增加历史。

### 冻结 checkpoint 的细粒度机制干预

正值表示完整模型优于固定或移除该路径后的模型：

| 组件 | 全量 ΔR@10，95% CI | history>10 ΔR@10，95% CI | history>20 ΔR@10，95% CI |
|---|---:|---:|---:|
| Learned gate vs fixed 0.5 | -0.000012 [-0.000102, 0.000078] | -0.000067 [-0.000451, 0.000316] | -0.000138 [-0.000730, 0.000455] |
| Event attention | +0.000128 [-0.000154, 0.000409] | +0.000405 [-0.000260, 0.001069] | -0.000138 [-0.000730, 0.000455] |
| Fusion MLP | +0.021716 [+0.015935, +0.027497] | +0.010283 [+0.008331, +0.012235] | +0.006196 [+0.003481, +0.008911] |

learned gate 与 event attention 的 Recall@10 区间均跨 0；当前证据不支持稳定独立增益。
移除 Fusion MLP 会造成大幅稳定下降，说明 full checkpoint 强烈依赖该非线性融合路径；
但这是模型内必要性而非公平的架构优越性结论。下面的独立重训结果表明替代结构可以通过
重新优化恢复绝大部分性能。Gate、attention、遮蔽差异都只是模型内部诊断，不构成用户
偏好的因果识别。

### 细粒度结构三种子独立重训

三个变体均不加载 full 权重，固定相同 BLaIR、RQ-VAE、token-loss 0.10、优化器、10 epoch
和 Beam-20/Top-10。训练期间只评估 Validation；Test 在结构和协议冻结后统一评估一次。
下表差值为 Full−变体：

| 模式 | Test Recall@10，95% CI | ΔR@10，配对 95% CI | history>10 ΔR@10，95% CI | 判定 |
|---|---:|---:|---:|---|
| Fixed 0.5 gate | 0.039955 [0.036948, 0.042963] | +0.000064 [-0.000190, +0.000317] | -0.000236 [-0.000759, +0.000287] | 无稳定差异 |
| No event attention | 0.039723 [0.036640, 0.042806] | +0.000296 [-0.000357, +0.000949] | -0.000270 [-0.001654, +0.001114] | 无稳定差异 |
| No fusion MLP | 0.039218 [0.038449, 0.039988] | +0.000801 [-0.001523, +0.003125] | +0.000506 [-0.003196, +0.004207] | 无稳定差异 |

Full Test Recall@10 为 0.040019 `[0.036935, 0.043103]`。三个替代结构的配对区间均跨 0，
长历史分层也全部跨 0。逐种子 McNemar 与 5,000 次用户样本 bootstrap 总体同样不支持
稳定组件优势；no-fusion 的 seed 2026 单独显著，但 seed 2027/2028 不复现，不能据此作
跨种子结论。

关键修正是：冻结 checkpoint 删除 Fusion MLP 时 R@10 下降 0.021716，但独立重训后差值
仅为 0.000801 且区间跨 0。因此此前的大幅下降主要表示既有模型对训练时计算路径的依赖，
不能作为 Fusion MLP 结构不可替代的证据。完整模型仍可作为冻结默认实现，但论文结论必须
写成“显式长期/短期建模提供可解释的预测敏感性，细粒度组件的独立增益尚未得到三种子
稳定支持”。

完整报告：

- [粗粒度独立重训消融](../artifacts/preference_reasoning_ablation_v1/REPORT.md)
- [修正版历史分层与遮蔽解释](../artifacts/preference_reasoning_interpretability_v2/REPORT.md)
- [冻结 checkpoint 细粒度机制干预](../artifacts/preference_reasoning_mechanism_ablation_v1/REPORT.md)
- [细粒度结构独立重训](../artifacts/preference_reasoning_retraining_ablation_v1/REPORT.md)

## V2.4 增强线最终结论（Validation 决策 + post-freeze Test）

V2.4 固定种子 `2026/2027/2028`。先只用 Validation 决定两个新方案均不升级；配置、
checkpoint 和 scorer 冻结后，再按用户要求执行一次性 Test 确认。Test 不参与训练、选参、
重选或结论翻转。

### 动态遗忘未稳定提升

| 指标 | Frozen V2 | Learned exponential | 配对增量 95% Student-t CI |
|---|---:|---:|---:|
| Recall@5 | 0.028215 | 0.027351 | [-0.001449, -0.000281] |
| NDCG@5 | 0.018065 | 0.017948 | [-0.000945, +0.000712] |
| Recall@10 | 0.044436 | 0.043043 | [-0.005261, +0.002476] |
| NDCG@10 | 0.023274 | 0.022996 | [-0.000985, +0.000427] |
| Observed-Tail Recall@10 | 0.000288 | 0.000227 | [-0.000321, +0.000200] |

Recall@10 保留率约 96.87%，增量区间跨 0；Recall@5 还出现稳定下降。动态遗忘不升级。

### Learned hybrid scorer 正式否决

该 scorer 在 Validation 用户哈希 70% fit / 30% selection 上拟合，候选为 Beam-20 与
BLaIR Observed-Tail@50 的并集。其三种子 selection Recall@10 为 0.025070，而生成基线为
0.043916；只保留 **57.1%**，配对增量 95% CI 为
`[-0.022677, -0.015015]`。Observed-Tail Recall@10 从 0.000104 降为 0，配对区间仍跨 0。
它同时未通过 97% 总体保留门槛和尾部稳定增量门槛，正式否决，只保留为候选漏斗诊断。

### 冻结后一次性 Test 确认

完整 57,439 条 Test、Beam-20/Top-10 的三种子均值如下；区间为相对原 V2 默认方案的
配对 Student-t 95% CI。

| 指标 | 原 V2 默认 | 动态遗忘 | Learned hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.025186 | 0.024049 | 0.011380 |
| NDCG@5 | 0.016166 | 0.015624 | 0.006684 |
| Recall@10 | 0.040019 | 0.038452 | 0.022685 |
| NDCG@10 | 0.020927 | 0.020247 | 0.010280 |
| Observed-Tail Recall@10 | 0.000166 | 0.000110 | 0.000055 |
| 总目录覆盖 | 0.026166 | 0.025311 | 0.037540 |
| Observed-Tail 目录覆盖 | 0.003806 | 0.003126 | 0.003887 |

- 动态遗忘 Recall@10 保留率 96.08%，配对增量 -0.001567，95% CI
  `[-0.005463, +0.002329]`：仍无稳定提升，且低于 97% 保留门槛。
- learned hybrid Recall@10 保留率 56.69%，配对增量 -0.017334，95% CI
  `[-0.022460, -0.012208]`：稳定显著下降；虽然总目录覆盖增加 0.011375，尾部命中与尾部
  覆盖均无稳定增益。
- 逐种子 McNemar 与 5,000 次用户 bootstrap 方向一致：hybrid 的总体 Top-10 在三个种子
  上均显著下降；动态遗忘 seed 2026/2028 下降、seed 2027 无显著差异。

原 V2 延迟来自此前冻结运行，V2.4 来自本次三种子并行运行，绝对延迟不作跨运行机制归因；
同一次 V2.4 运行内，hybrid 相对动态遗忘增加约 3.284 ms/example。

### 现代基线与外部数据边界

- UniSRec 已按官方 commit `05aa5cba2809112c32808f70d16abc61c05c6538` 完成适配，但
  CPU 训练在本地资源下耗时不可接受，用户主动停止；2026/2027/2028 均无完整正式结果，
  不做任何优劣推断。
- All Beauty、Toys and Games、Sports and Outdoors 的 Amazon Reviews 2023 真实数据、
  leave-last-out 审计和 BLaIR 商品向量均已准备完成，所有审计
  `synthetic_data_used=false`。三个目录分别含 356、162,035、156,235 个商品。
- 外部 V2/adaptive 三种子训练未执行，不能把“数据准备完成”表述为“外部验证完成”。原始
  数据 SHA-256、BLaIR SHA-256、行数和停止状态详见
  [V2.4 协议](V2_4_ENHANCEMENT_PROTOCOL.md)。

因此冻结后的科学边界是：V2.4 的一次性 Test 确认没有产生总体或长尾优势，外部品类模型
优势与 UniSRec 对比仍无结果；原 V2 默认配置保持不变。

## V2.5 三种子冻结确认（Validation-only）

V2.5 使用零初始化、有界残差保护原 Beam 排序，冻结配置为
`b0.5_tail8_preserve4_scale1`。seed 固定为 `2026/2027/2028`，三个种子独立构建 Train
候选、拟合 scorer 并在完整 Validation 评测；Test 全程不可用。

| 指标 | 三种子均值 | Student-t 95% CI | 门禁 |
|---|---:|---:|---|
| Recall@10 增量 | -0.000017 | [-0.000017416, -0.000017405] | 通过非劣保护 |
| Observed-Tail Recall@10 增量 | +0.000364 | [-0.000831, +0.001558] | 未通过 |
| Observed-Tail 目录覆盖增量 | +0.069075 | [-0.164446, +0.302596] | 未通过 |
| 新增延迟，ms/example | +0.800 | [+0.355, +1.245] | 通过 |

seed 2026/2027/2028 的尾部 Recall@10 增量分别为 `+0.000909/+0.000182/0`；只有 seed
2026 通过单种子临时门禁。正确结论是“有界残差基本保持总体准确率，但尾部增益未跨种子
稳定复现”，不得把 seed 2026 解释为稳定优势。完整结果见
`artifacts/v2_5_multiseed/summary/{report.json,REPORT.md}`，冻结协议见
`docs/V2_5_IMPROVEMENT_PROTOCOL.md`。

### V2.6 冻结负结果与 V2.7 边界诊断

V2.6 对全部 339,519 条 Train 样本完成普查，3,310 条 hard positives 覆盖 2,838 个用户和
2,611 个商品，数据门禁通过。seed 2026 的 scale `0.25/0.50/0.75` 没有改变尾部命中；
scale `1.0` 的总体 Recall@10 从 `0.046101` 降至 `0.046049`，Observed-Tail Recall@10
保持 `0.000182`，尾部新增/损失均为 0，尾部覆盖增量 `0.004404` 也低于门槛。因此
`selected_validation_variant=null`，没有启动五种子或读取 Test。

随后冻结的 V2.7 Validation 诊断确认，Beam-20/BLaIR-50/union 的尾部 oracle 分别为
`0.000364/0.063534/0.063716`。697 个 BLaIR-only 尾部目标中有 251 个 BLaIR rank≤10。
这些目标跨越 Top-10 平均需要约 `0.526316` 的 pair lift；V2.6 实际中位 lift 为
`0.511432`，最终中位边界差仍为 `-0.014976`，转入 Top-10 数量为 0。这说明候选存在，
旧模型却被有界残差的量级卡住。该诊断通过的是工程可行性门禁，不是效果门禁。

固定 V2.7 listwise scorer 随后完成全部 Train/seed 2026 Validation 门禁。scale 0.25 的
总体 Recall@10 为 `0.046031`（保留率 99.85%），尾部只净增 1 个命中，bootstrap CI
`[-0.000182, 0.000454]`；scale 0.50 的尾部净增 60 且 CI 下界大于 0，但总体 Recall@10
降到 `0.020752`（保留率 45.02%）；scale 0.75/1.00 的总体保留率仅 3.47%，推荐结果几乎
全部成为尾部商品。故 `selected_validation_variant=null`。新模型证明无硬上限确实能跨越
边界，但当前特征无法区分正确尾部候选与语义相似负例，不能形成准确率—长尾兼得的方案。

冻结协议见 `docs/V2_6_GAIN_CONFIRMATION_PROTOCOL.md` 与
`docs/V2_7_LISTWISE_JOINT_SCORER_PROTOCOL.md`；结构化诊断见
`artifacts/v2_7_top10_boundary/seed_2026/report.json`，正式门禁见
`artifacts/v2_7_listwise_joint/seed_2026/report.json`。原 V2 默认方案不变。

## 可复现实验说明

建议 Python 3.12；正式环境使用 `/opt/anaconda3/bin/python`、PyTorch 2.2.2、CPU。

```bash
cd /path/to/Reasoning-Enhanced-Generative-Recommendation-for-E-commerce
pip install -e '.[embeddings,dev]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. pytest -q -p no:cacheprovider

# 验证真实数据、SHA-256、行数与商品引用
python scripts/audit_amazon2023.py \
  --processed-dir data/amazon2023/processed/Musical_Instruments/last_out_v2

# 从冻结配置独立训练三个默认模型；每个 seed 使用独立输出目录
for seed in 2026 2027 2028; do
  python scripts/train_amazon2023.py \
    --config configs/v2_final_default.yaml \
    --seed "$seed" \
    --output-dir "artifacts/v2_final_reproduction/seed_${seed}"
done

# 在同一 Beam-20 harness 下重评原 V2 与 token-loss
python scripts/run_v2_long_tail_paired_reevaluation.py \
  --config configs/v2_amazon2023_top10.yaml \
  --output-root artifacts/v2_long_tail_paired_reevaluation

# Validation-only 温和选参、锁参和三种子正式 Test
python scripts/run_candidate_reranking_mild_experiments.py \
  --output-dir artifacts/candidate_reranking_mild_v1 \
  --bootstrap-samples 5000

# 激进软配额的历史正式实验
python scripts/run_candidate_reranking_experiments.py \
  --output-dir artifacts/candidate_reranking_v1 \
  --pilot-examples 10000 \
  --bootstrap-samples 5000

# 重算错误漏斗、BLaIR rank 和匿名真实案例
python scripts/analyze_v2_final_errors.py \
  --config configs/v2_amazon2023_top10.yaml \
  --mild-root artifacts/candidate_reranking_mild_v1 \
  --catalog artifacts/amazon2023_musical_instruments_long_tail_training_v1/formal/token_loss_010/seed_2026/catalog.json \
  --output-dir artifacts/v2_final_report

# 修正版历史分层、gate/cosine 与事件遮蔽
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python scripts/analyze_preference_reasoning_interpretability.py

# 冻结 checkpoint 的细粒度机制干预（可断点续跑）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python scripts/run_preference_reasoning_mechanism_ablation.py \
  --batch-size 128 --workers 4 --torch-threads-per-worker 2

# fixed gate / no event attention / no fusion MLP 的三种子独立重训。
# 训练只看 Validation；配置冻结后由同一 runner 统一执行 Test 和统计汇总。
caffeinate -i env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python scripts/run_preference_reasoning_retraining_ablation.py \
  --train-workers 3 --train-threads-per-worker 2 \
  --eval-workers 4 --torch-threads-per-worker 2

# V2.4 动态遗忘：三种子、Validation-only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_dynamic_forgetting.py \
  --workers 3 --threads-per-worker 2

# V2.4 learned hybrid：Validation 用户哈希 70/30，Test 不参与
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_hybrid_scorer_multiseed.py --workers 3

# Beauty/Toys/Sports 真实数据审计、BLaIR 编码与导出
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/prepare_v2_4_external_categories.py --minimum-free-gib 4

# 冻结后的三种子一次性 Test 确认；已有报告会断点复用
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_test_confirmation.py \
  --workers 3 --threads-per-worker 2 --bootstrap-samples 5000

# V2.7 Validation-only 候选 oracle 与 Top-10 边界诊断；不读取 Test
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /opt/anaconda3/bin/python scripts/analyze_v2_7_top10_boundary.py \
  --artifact-dir artifacts/amazon2023_musical_instruments_long_tail_training_v1/formal/token_loss_010/seed_2026 \
  --validation-cache artifacts/v2_5_residual/full_validation_seed_2026/candidate_cache.pt \
  --cache-report artifacts/v2_5_residual/full_validation_seed_2026/report.json \
  --v2-6-report artifacts/v2_6_full_hard_positive/seed_2026/report.json \
  --output-dir artifacts/v2_7_top10_boundary/seed_2026

# V2.7 固定 listwise scorer；全部门禁失败时自动停止，不启动多种子
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /opt/anaconda3/bin/python scripts/train_v2_7_listwise_joint.py \
  --artifact-dir artifacts/amazon2023_musical_instruments_long_tail_training_v1/formal/token_loss_010/seed_2026 \
  --census-dir artifacts/v2_6_hard_positive_census/full_seed_2026 \
  --validation-cache artifacts/v2_5_residual/full_validation_seed_2026/candidate_cache.pt \
  --protocol docs/V2_7_LISTWISE_JOINT_SCORER_PROTOCOL.md \
  --output-dir artifacts/v2_7_listwise_joint/seed_2026
```

错误分析会校验 indicator 与原始 Test 的目标、顺序和用户哈希，并重算 BLaIR Tail@50；当前
620 个单种子检索命中与正式报告完全一致。所有输入/输出 SHA-256 记录在
[`provenance.json`](../artifacts/v2_final_report/provenance.json)。

## 正式产物索引

- 四方案 Pareto：[v2_pareto_summary.json](../artifacts/candidate_reranking_mild_v1/v2_pareto_summary.json)
- 温和方案验收：[acceptance.json](../artifacts/candidate_reranking_mild_v1/acceptance.json)
- Student-t 汇总：[test_summary.json](../artifacts/candidate_reranking_mild_v1/test_summary.json)
- McNemar 与用户 bootstrap：[paired_tests.json](../artifacts/candidate_reranking_mild_v1/paired_tests.json)
- 错误漏斗：[error_analysis.json](../artifacts/v2_final_report/error_analysis.json)
- 真实匿名案例：[error_cases.jsonl](../artifacts/v2_final_report/error_cases.jsonl)
- 来源哈希：[provenance.json](../artifacts/v2_final_report/provenance.json)
- V2.2 重训前证据冻结清单：[v2_evidence_freeze.json](../artifacts/v2_final_report/v2_evidence_freeze.json)
- V2.3 最终证据冻结清单：[v2_evidence_freeze_v2_3.json](../artifacts/v2_final_report/v2_evidence_freeze_v2_3.json)
- V2.4 增强线冻结清单：[v2_evidence_freeze_v2_4.json](../artifacts/v2_final_report/v2_evidence_freeze_v2_4.json)
- V2.4 Test 确认冻结清单：[v2_evidence_freeze_v2_4_test.json](../artifacts/v2_final_report/v2_evidence_freeze_v2_4_test.json)
- V2.4 动态遗忘：[summary.json](../artifacts/v2_4_adaptive_forgetting/summary.json)
- V2.4 learned hybrid：[summary.json](../artifacts/v2_4_hybrid_scorer_multiseed/summary.json)
- V2.4 外部数据准备状态：[state.json](../artifacts/v2_4_external_preparation/state.json)
- UniSRec 主动停止状态：[state.json](../artifacts/unisrec_amazon2023_musical_multiseed/state.json)
- 外部模型验证停止状态：[state.json](../artifacts/v2_4_external_validation/state.json)
- V2.4 Test 确认：[REPORT.md](../artifacts/v2_4_test_confirmation/REPORT.md)
- V2.4 Test 三种子统计：[summary.json](../artifacts/v2_4_test_confirmation/summary.json)
- V2.4 Test McNemar/bootstrap：[paired_tests.json](../artifacts/v2_4_test_confirmation/paired_tests.json)
- V2.6 全量 Train 普查：[report.json](../artifacts/v2_6_hard_positive_census/full_seed_2026/report.json)
- V2.6 seed 2026 严格门禁：[report.json](../artifacts/v2_6_full_hard_positive/seed_2026/report.json)
- V2.7 Top-10 边界诊断：[REPORT.md](../artifacts/v2_7_top10_boundary/seed_2026/REPORT.md)
- V2.7 seed 2026 listwise 严格门禁：[REPORT.md](../artifacts/v2_7_listwise_joint/seed_2026/REPORT.md)
- Preference Reasoning 粗粒度消融：[summary.json](../artifacts/preference_reasoning_ablation_v1/summary.json)
- 历史分层与解释实验：[summary.json](../artifacts/preference_reasoning_interpretability_v2/summary.json)
- 细粒度机制干预：[summary.json](../artifacts/preference_reasoning_mechanism_ablation_v1/summary.json)
- 细粒度结构独立重训：[summary.json](../artifacts/preference_reasoning_retraining_ablation_v1/summary.json)
