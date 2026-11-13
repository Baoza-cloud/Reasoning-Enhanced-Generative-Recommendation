# 实验台账

所有正式结果只接受通过 `data_audit.json` 的真实数据；smoke、hashing 向量和合成数据不得
进入正式对比表。

## 数据与协议

- 数据集：McAuley Lab Amazon Reviews 2023，`Musical_Instruments`，官方去重 5-core；
- 切分：官方 leave-last-out；
- 商品数：24,587；训练/验证/测试样本：339,519 / 57,439 / 57,439；
- 商品向量：`hyp1231/blair-roberta-base`，CLS pooling，L2 归一化，24,587 × 768；
- 历史 V1/V2 单种子评测使用 Catalog Trie constrained Beam-10；V2.1 正式多种子评测使用
  Beam-20 并返回 Top-10，指标均为单目标 Recall@5/10、NDCG@5/10；
- 默认不移除历史商品，所有模型保持同一协议；
- V1 数据审计 SHA-256：
  `9291e399add36fc89a388265a90831f58b1d827155b59b1b4c34ed28315929ad`。

## V1 正式结果

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Popularity | Valid | 0.018663 | 0.011836 | 0.030415 | 0.015594 |
| First-order Markov | Valid | 0.023660 | 0.017071 | 0.034280 | 0.020482 |
| RQ-VAE + Preference Reasoning + Beam | Valid | **0.023834** | 0.016016 | **0.035429** | 0.019778 |
| Popularity | Test | 0.015321 | 0.009716 | 0.026271 | 0.013232 |
| First-order Markov | Test | **0.021623** | **0.015661** | 0.031094 | **0.018713** |
| RQ-VAE + Preference Reasoning + Beam | Test | 0.021310 | 0.013852 | **0.032434** | 0.017462 |

结论：生成模型在 Valid Recall@5/10 和 Test Recall@10 上最好，但 Test Recall@5 与 NDCG
仍略低于一阶 Markov。当前优势是候选覆盖和 Top-10 召回，短板是头部排序，不能声称全面
超过传统基线。V2 的目标是用真实评分强度和真实时间间隔改善这个短板。

RQ-VAE 质量：三层使用率 0.90625 / 0.984375 / 1.0；困惑度 209.867 / 240.661 /
250.851；唯一 code tuple 比例 0.901777。K-means 初始化、EMA 更新和质量门禁均已启用。

产物：

- `artifacts/amazon2023_musical_instruments_blair_kmeans_ema_v1/report.json`
- `artifacts/amazon2023_musical_instruments_blair_kmeans_ema_v1/baseline_report.json`

## V2 行为强度

V2 不生成行为。它从 511,836 条官方 benchmark 行中按 `(user_id, parent_asin)` 回填历史
事件自己的评分与时间戳：3,353,960 个历史事件全部匹配，缺失 0，未来时间穿越 0。

- 行为类型：1–2 星负向、3 星中性、4–5 星正向；
- 连续强度：`rating / 5.0`；
- 时间桶：`floor(log2(1 + elapsed_days))`；
- 模型：连续强度投影、用户上下文条件化衰减率、强度感知长期/短期偏好加权；
- Semantic ID：复用 V1 已验收的 RQ-VAE，保证 V1/V2 对比不混入码本变化。

全量训练配置为 10 epoch，完整 valid/test 均使用 Catalog Trie constrained Beam-10；主实验
已完成：

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| V1 RQ-VAE + Preference Reasoning | Valid | 0.023834 | 0.016016 | 0.035429 | 0.019778 |
| V2 + observed behavior intensity | Valid | **0.025314** | **0.016647** | **0.038859** | **0.021046** |
| V1 RQ-VAE + Preference Reasoning | Test | 0.021310 | 0.013852 | 0.032434 | 0.017462 |
| V2 + observed behavior intensity | Test | **0.023451** | **0.015343** | **0.035237** | **0.019154** |

相对 V1，V2 在 Test Recall@5、NDCG@5、Recall@10、NDCG@10 上分别提升 10.05%、
10.77%、8.64%、9.69%；Valid 对应提升 6.21%、3.95%、9.68%、6.41%。验证/测试 loss
分别为 2.182982 / 2.231635，较 V1 微升约 0.15% / 0.14%，说明序列 token loss 与最终
Top-K 排序指标并不完全同向。主实验表明 V2 整体方案相对 V1 改善了排序；下述无强度
消融进一步表明，不能把全部增益单独归因于连续强度模块。

正式报告通过真实数据审计，覆盖训练/验证/测试 339,519 / 57,439 / 57,439 条样本；
`synthetic_data_used=false`。产物：

- `artifacts/amazon2023_musical_instruments_blair_v2_intensity/report.json`
- report SHA-256：`9584035dad6505870b55bf24c117532e03e284f7112e8022b116e41c96ce7c51`
- generator SHA-256：`bdc97701a75e640ac9eb34197d0a4dda94567ad5c6470fbc5f00156f8a6771f2`

连续强度的主消融使用
`configs/v2_amazon2023_ablation_no_intensity.yaml`：保留真实 rating 行为类型和真实时间桶，
仅关闭连续强度投影、个性化衰减和强度感知加权。10 epoch 全量消融已经完成：

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| V2 no-intensity ablation | Valid | **0.026445** | **0.017271** | **0.039851** | **0.021603** |
| V2 observed intensity | Valid | 0.025314 | 0.016647 | 0.038859 | 0.021046 |
| V2 no-intensity ablation | Test | 0.023016 | 0.014717 | **0.036473** | 0.019062 |
| V2 observed intensity | Test | **0.023451** | **0.015343** | 0.035237 | **0.019154** |

相对无强度消融，强度模块在 Test Recall@5、NDCG@5、NDCG@10 上分别提升 1.89%、
4.26%、0.48%，但 Test Recall@10 下降 3.39%，且四项 Valid 指标均下降 2.49%–4.28%。
验证/测试 loss 也分别上升 0.92% / 0.82%。因此当前证据只支持“改善测试集头部排序”，
不支持“行为强度全面提升推荐效果”；V2 相对 V1 的增益还包含真实行为类型与时间信息。

消融报告通过真实数据审计，`synthetic_data_used=false`。产物：

- `artifacts/amazon2023_musical_instruments_blair_v2_no_intensity/report.json`
- report SHA-256：`0270d49000d2e1041e17de4b755683fdc175f41a734e479e610ace27ac6cab0f`
- generator SHA-256：`b283b0e02653ca228973bcf90d96135e03aae614383df2dfe560d955b7b70561`

## SASRec 全量基线

SASRec 在完全相同的 Amazon leave-last-out 数据、目录商品集合和不移除历史商品协议上训练
10 epoch；评测直接在全部 24,587 个商品 logits 上取 Top-K，不使用采样候选：

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| V2 observed intensity + Beam-10 | Valid | 0.025314 | 0.016647 | 0.038859 | 0.021046 |
| SASRec | Valid | **0.028065** | **0.017820** | **0.045857** | **0.023520** |
| V2 observed intensity + Beam-10 | Test | 0.023451 | 0.015343 | 0.035237 | 0.019154 |
| SASRec | Test | **0.024304** | **0.015497** | **0.039677** | **0.020409** |

SASRec 在四项 Test 指标上较 V2 分别高 3.51%、0.99%、11.19%、6.15%，说明当前生成式
模型尚未超过强深度序列基线，主要差距仍在 Top-10 覆盖。SASRec 报告已通过真实数据审计，
`synthetic_data_used=false`，样本数为 339,519 / 57,439 / 57,439；其全商品 softmax loss
与生成模型的 Semantic-ID token loss 定义不同，不作横向比较。

- `artifacts/amazon2023_musical_instruments_sasrec/report.json`
- data-audit SHA-256：`9291e399add36fc89a388265a90831f58b1d827155b59b1b4c34ed28315929ad`
- report SHA-256：`0c0c4c5e1d276da0f2223ad177743918b5857af1006832e72f2369999e4cb3a8`
- model SHA-256：`8bcb4875bf3b0728701a2911912f11fc6aa88b4c2ef1016bae7f3874da7f567f`

## V2.1 Beam-20 三随机种子

为检验 Top-10 覆盖和随机种子稳定性，V2 在相同真实数据、BLaIR/RQ-VAE 和全局交叉熵
训练口径上，将解码扩展为 Catalog Trie constrained Beam-20、返回前 10 个真实商品。
seed 2026 复用原 epoch-10 检查点完整重评，seed 2027/2028 分别独立训练 10 epoch；所有
57,439 条验证和测试样本均参与评测。区间为三个独立随机种子的双侧 95% Student-t 区间，
下表记为“均值 ± 区间半宽”。

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| V2.1 global-CE + Beam-20 | Valid | 0.027403 ± 0.002290 | 0.017980 ± 0.001513 | 0.043739 ± 0.002540 | 0.023221 ± 0.001641 |
| SASRec full-catalog | Valid | **0.028668 ± 0.001962** | **0.018122 ± 0.000948** | **0.046560 ± 0.002413** | **0.023858 ± 0.001134** |
| V2.1 global-CE + Beam-20 | Test | **0.024890 ± 0.001016** | **0.016159 ± 0.000489** | **0.040002 ± 0.000661** | **0.021012 ± 0.000255** |
| SASRec full-catalog | Test | 0.024763 ± 0.003246 | 0.015891 ± 0.002561 | 0.039955 ± 0.003171 | 0.020767 ± 0.002509 |

生成模型四项 Test 均值均略高于 SASRec，SASRec 四项 Valid 均值均较高；所有对应 95%
区间均明显重叠，因此当前只能表述为“测试均值持平且略高”，不能声称相对 SASRec 有统计
显著优势。生成模型跨种子方差明显更小，但样本量只有三个种子，也不作更强稳定性推断。

正式产物：

- `artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed/summary.json`
  （SHA-256：`5ed8ecfb284d7a3ae16d91b6099e532c19092d2941c54d4abd98267cbcc95a64`）；
- `artifacts/amazon2023_musical_instruments_sasrec_multiseed/summary.json`
  （SHA-256：`51d88233ca387708ec52b2da85a5e926b76c167bdcbc9a5625e1ef67337cbc73`）。

### 行为强度三种子消融

同协议无强度消融仅关闭连续强度投影、个性化衰减和强度感知加权，其余数据、RQ-VAE、
训练轮数和 Beam-20/Top-10 评测口径保持不变。三个相同种子均完成独立训练与全量评测：

| 模型 | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| V2.1 no-intensity | Valid | **0.028511 ± 0.000264** | **0.018406 ± 0.000426** | **0.044691 ± 0.001313** | **0.023611 ± 0.000850** |
| V2.1 observed intensity | Valid | 0.027403 ± 0.002290 | 0.017980 ± 0.001513 | 0.043739 ± 0.002540 | 0.023221 ± 0.001641 |
| V2.1 no-intensity | Test | 0.024815 ± 0.002070 | 0.016056 ± 0.001769 | 0.039921 ± 0.002988 | 0.020908 ± 0.001632 |
| V2.1 observed intensity | Test | **0.024890 ± 0.001016** | **0.016159 ± 0.000489** | **0.040002 ± 0.000661** | **0.021012 ± 0.000255** |

按相同随机种子计算“强度版本减无强度版本”的配对差值：Test Recall@5、NDCG@5、
Recall@10、NDCG@10 的平均相对变化分别为 +0.30%、+0.64%、+0.20%、+0.50%；对应
差值的 95% Student-t 区间分别为 `[-0.001118, 0.001269]`、
`[-0.001222, 0.001428]`、`[-0.003299, 0.003461]`、
`[-0.001310, 0.001518]`，全部跨 0。Valid 四项均值则下降 1.65%–3.89%。因此三种子证据
支持“行为强度在测试均值上方向一致地小幅改善”，但不支持统计显著或跨切分稳定增益。

- `artifacts/amazon2023_musical_instruments_blair_v2_top10_no_intensity_multiseed/summary.json`
- summary SHA-256：`76f8d60d1eeeed58c05271e5f32c7dcff0bb371b0adb84195bc12e9e03eae59a`

## Beam-1/10/20 覆盖、长尾与效率

使用 V2.1 seed 2026 的 epoch-10 模型，在全部 57,439 条 Test 样本上统一重评。Beam-1
返回 1 个商品，仅报告 Recall/NDCG@1 并作为贪心延迟基线；Beam-10/20 均返回 Top-10。
计时覆盖数据组批、host-to-device 传输和 Catalog Trie 约束解码，设备为 8 逻辑核 CPU。

长尾定义只使用训练目标频次，避免测试泄漏：按目录频次排序，前 20%（4,918 个商品）为
head，后 80%（19,669 个商品）为 tail。训练目标中 tail 占 35.69%，Test 目标中 tail 占
41.04%。

| Beam | 返回数 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 唯一商品 | 目录覆盖率 | 长尾目录覆盖率 | 长尾曝光占比 | ms/样本 | 样本/秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | — | — | — | — | 236 | 0.9599% | 0.2135% | 4.0251% | **2.506** | **399.04** |
| 10 | 10 | 0.023451 | 0.015343 | 0.035237 | 0.019154 | **805** | **3.2741%** | **0.9558%** | **2.5408%** | 12.409 | 80.58 |
| 20 | 10 | **0.024478** | **0.016034** | **0.039712** | **0.020909** | 634 | 2.5786% | 0.5593% | 0.9981% | 24.191 | 41.34 |

Beam-1 的 Recall@1/NDCG@1 均为 0.004805。Beam-20 相对 Beam-10 将 Recall@5、
NDCG@5、Recall@10、NDCG@10 分别提高 4.38%、4.50%、12.70%、9.16%，代价是耗时
增加 94.94%。同时唯一推荐商品从 805 降至 634，目录覆盖率下降 21.24%，长尾目录覆盖率
下降 41.49%，长尾曝光占比下降 60.72%。因此 Beam-20 是当前准确率配置，Beam-10 是更
合适的覆盖/吞吐折中；扩大 beam 并不会自动增加最终 Top-10 的商品多样性。

长尾命中仍是主要短板：Beam-20 对 head 目标的 Recall@5/10 为 0.041514 / 0.067232，
对 tail 目标仅为 0 / 0.000170；即使 Beam-10 的 tail Recall@5/10 也只有 0.000042 /
0.000467。后续应优先尝试流行度校准、长尾感知训练或多样性约束，而不是继续单纯增大 beam。

- `artifacts/amazon2023_musical_instruments_blair_v2_beam_coverage_efficiency/report.json`
- report SHA-256：`a1e27449caff21528bc0a108ce549749784b0e3389e336f81d752c2c37628dd8`
- `synthetic_data_used=false`，所有结果均来自通过审计的 Amazon Reviews 2023 真实数据。

## 长尾校准与错误分析

在 V2.1 seed 2026 的 Beam-20 候选池上，用训练目标频次实施推理期流行度校准：
`score - alpha * log1p(train_frequency) / log1p(max_train_frequency)`。仅在全部 57,439 条
Validation 上选择 `alpha`，并要求 Recall@10 至少保留未校准版本的 97%；Test 不参与
选参。最终 `alpha=0.75`，完整 57,439 条 Test 结果如下：

| Test 配置 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 唯一商品 | 目录覆盖率 | 长尾目录覆盖率 | 长尾曝光占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 Beam-20 Top-10 | **0.024478** | **0.016034** | **0.039712** | **0.020909** | 634 | 2.5786% | 0.5593% | 0.9981% |
| 训练频次校准 `alpha=0.75` | 0.023851 | 0.015615 | 0.038441 | 0.020293 | **657** | **2.6721%** | **0.5999%** | **1.5051%** |

校准使唯一商品数增加 3.63%、长尾目录覆盖率增加 7.27%、长尾曝光占比增加 50.78%，但
Recall@5/10 分别下降 2.56% / 3.20%，NDCG@5/10 分别下降 2.61% / 2.95%。长尾目标
Recall@10 从 `0.000170` 增至 `0.000212`，实际只从 4 次命中增加到 5 次；Recall@5 仍为
0。因此该方法只作为覆盖率可调旋钮，不进入默认准确率配置。

错误分析进一步显示，Beam-20 候选池 Recall@20 为 0.058392，而最终 Recall@10 为
0.039712；校准产生 57 个 recovered、130 个 lost，净损失 73 次命中。测试目标训练频次
为 0 或 1–5 的 10,289 个样本在 Top-10 中合计只命中 1 次，主要瓶颈位于候选生成和训练
分布，而非 Top-10 内的简单重排。后续修复应转向长尾感知采样/损失、Semantic-ID token
频率校正和多样化候选生成。

- `artifacts/amazon2023_musical_instruments_long_tail_full/report.json`
  （SHA-256：`99823ac7c37c5247dd5f13993125894e34a69edefebb8440678a4c911d8adc01`）
- `artifacts/amazon2023_musical_instruments_long_tail_full/error_analysis.json`
  （SHA-256：`57cfd878c450adb89be7226ed362895e363b2b6347a9327b98173de79d8b8e58`）
- 两份报告均为 `synthetic_data_used=false` 的真实 Test 全量结果；错误案例中的用户 ID 已哈希。

## Industrial & Scientific 三种子外部验证

使用同一份通过审计的 Amazon Reviews 2023 `Industrial_and_Scientific` leave-last-out
切分（259,992 Train、50,985 Validation、50,985 Test，25,848 个商品），固定 BLaIR 商品
向量、RQ-VAE 与 Semantic-ID catalog，分别独立训练 V2.1 生成器和 SASRec 的
seed 2026/2027/2028。V2.1 使用 10 epoch、Beam-20/return-10 全目录约束解码；SASRec
使用相同 10 epoch 和完整商品目录。全部报告均为真实数据，`synthetic_data_used=false`。

三种子 Test 均值及各模型独立的 95% Student-t 区间：

| 模型 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| V2.1 Beam-20 | 0.017698 [0.013577, 0.021819] | 0.011356 [0.008783, 0.013929] | 0.028093 [0.026205, 0.029982] | 0.014705 [0.012854, 0.016556] |
| SASRec | 0.016482 [0.014797, 0.018167] | 0.010864 [0.009660, 0.012068] | 0.025890 [0.024738, 0.027042] | 0.013883 [0.012821, 0.014945] |

更直接的同 seed 配对分析如下；差值定义为 V2.1 减 SASRec：

| Test 指标 | 平均绝对差值 | 配对 95% CI | 平均相对差值 | 相对差值 95% CI |
|---|---:|---:|---:|---:|
| Recall@5 | +0.001216 | [-0.004571, 0.007003] | +7.78% | [-28.97%, 44.53%] |
| NDCG@5 | +0.000492 | [-0.003279, 0.004263] | +4.96% | [-31.20%, 41.12%] |
| Recall@10 | +0.002203 | [-0.000836, 0.005242] | +8.57% | [-3.66%, 20.80%] |
| NDCG@10 | +0.000822 | [-0.002091, 0.003734] | +6.10% | [-15.72%, 27.92%] |

四项 Test 均值方向为正，但所有配对区间均跨 0；Validation 四项平均差值则均为负且区间
同样跨 0。因此不能沿用 seed 2026 单次实验的“提升 14%–25%”表述，也不能宣称 V2.1
统计显著或稳定优于 SASRec。严格结论是：V2.1 在该外部品类的 Test 均值高 4.96%–8.57%，
但三个种子的证据不足以排除随机初始化造成的差异。

- `artifacts/amazon2023_industrial_scientific_blair_v2_multiseed/summary.json`
  （SHA-256：`3999fb6a25a96408eab887bb28b3de5470992ac2644341d4081a62da22e5c0c2`）
- `artifacts/amazon2023_industrial_scientific_sasrec_multiseed/summary.json`
  （SHA-256：`ea63a61f48026321c9cfe7c03f7f3e228f82b2527aab8e8253e3c9411c84af67`）
- `artifacts/amazon2023_industrial_scientific_multiseed_comparison/paired_summary.json`
  （SHA-256：`0d1bae8a48a9e6a864e1f3bfe6d4157bca9f5bf2c2688dc59d602a2a5c455a72`）

## V2 最终 Pareto 与候选漏斗

三种子正式结果已经冻结：默认线上方案为 `token_loss_010`；温和 BLaIR+频率方案稳定扩大
目录覆盖但尾部 Recall@10 增量区间跨 0；激进软配额具有稳定尾部命中增益，但整体
Recall/NDCG 下降，因此只作为可选长尾增强模式。

真实错误漏斗显示，36,216 个 Observed-Tail seed-example pairs 中，Beam-20 生成器只覆盖
15 个，BLaIR Tail@50 覆盖 1,860 个，候选并集覆盖 1,867 个，最终 Top-10 仅命中 14 个。
候选召回为 5.155%，候选到 Top-10 的条件转化率仅 0.750%。主要原因是 BLaIR-only 候选
的 generator relevance 为 0，而温和配置只给 0.25 语义权重和 0.01 软配额奖励；这是
在 97% 整体 Recall 保留约束下形成的准确率—长尾折中，不应解释成检索器本身无效。

完整配置、四方案指标区间、统计边界、复现命令和匿名真实案例统一见
[docs/V2_FINAL_REPORT.md](docs/V2_FINAL_REPORT.md)。结构化产物为：

- `artifacts/v2_final_report/error_analysis.json`
- `artifacts/v2_final_report/error_cases.jsonl`
- `artifacts/v2_final_report/provenance.json`

## Preference Reasoning 分层解释与细粒度机制消融

粗粒度 `full vs mean_pool` 为三个种子分别独立训练，固定真实数据、BLaIR、RQ-VAE、
token-loss 0.10、10 epoch 和 Beam-20/Top-10。Test Recall@10 为 0.040019 对 0.038418，
配对增量 +0.001602，95% CI `[-0.000884, +0.004087]`；四项 Test 增量区间均跨 0，不能
声称完整 Preference Reasoning 对 mean-pool 有稳定测试优势。

修正版解释实验按 history≤10、11–20、>20 分层。history≤10 的 long/short cosine=1
由窗口结构决定，不作为兴趣迁移证据。history>10 移除最近 10 条事件使 Recall@10 稳定
下降 0.005597，95% CI `[0.004307, 0.006886]`；history>20 只保留最近 10 条比完整历史
高 0.002341，95% CI `[0.001748, 0.002933]`。这说明近期上下文有稳定预测贡献，而超长
历史中的早期事件平均可能造成干扰。

在同一批 full checkpoint 上的细粒度干预结果如下。差值均为 full 减干预模式：

| 组件 | 全量 ΔRecall@10，95% CI | history>10 ΔRecall@10，95% CI |
|---|---:|---:|
| Learned gate vs fixed 0.5 | -0.000012 [-0.000102, 0.000078] | -0.000067 [-0.000451, 0.000316] |
| Event attention | +0.000128 [-0.000154, 0.000409] | +0.000405 [-0.000260, 0.001069] |
| Fusion MLP | +0.021716 [+0.015935, +0.027497] | +0.010283 [+0.008331, +0.012235] |

learned gate 与 event attention 尚无稳定独立增益；移除 Fusion MLP 会使当前 checkpoint
显著退化。该结果回答“当前模型是否依赖组件”，不回答“独立重训后哪个架构容量更强”。
三个替代结构随后完成 seed 2026/2027/2028 的独立 10-epoch 训练和完整 Beam-20 Test：

| 独立重训模式 | Test Recall@10 | Full−模式 ΔR@10，配对 95% CI | history>10 ΔR@10，95% CI |
|---|---:|---:|---:|
| Fixed 0.5 gate | 0.039955 | +0.000064 [-0.000190, +0.000317] | -0.000236 [-0.000759, +0.000287] |
| No event attention | 0.039723 | +0.000296 [-0.000357, +0.000949] | -0.000270 [-0.001654, +0.001114] |
| No fusion MLP | 0.039218 | +0.000801 [-0.001523, +0.003125] | +0.000506 [-0.003196, +0.004207] |

三个区间均跨 0，细粒度组件没有稳定独立 Recall@10 优势。尤其是 no-fusion 在冻结 full
checkpoint 内下降 0.021716，独立重训后只下降 0.000801 且区间跨 0，说明冻结干预的大幅
退化主要是计算路径失配，不能解释为 Fusion MLP 架构不可替代。no-fusion seed 2026 的
McNemar/用户 bootstrap 单独显著，但 2027/2028 不复现，正式结论仍以三种子区间为准。
所有遮蔽和权重分析均为预测敏感性，不作因果解释。

- `artifacts/preference_reasoning_ablation_v1/{summary.json,REPORT.md}`
- `artifacts/preference_reasoning_interpretability_v2/{summary.json,REPORT.md}`
- `artifacts/preference_reasoning_mechanism_ablation_v1/{summary.json,REPORT.md}`
- `artifacts/preference_reasoning_retraining_ablation_v1/{summary.json,REPORT.md}`

## V2.4 增强线（已结束并冻结）

V2.4 固定 `2026/2027/2028` 三个种子，先只在 Musical Instruments Validation 上验收新
方案。动态遗忘 Recall@10 为 0.043043，frozen V2 为 0.044436，配对增量
95% Student-t CI `[-0.005261, +0.002476]`，未稳定提升。learned hybrid scorer 的
Recall@10 为 0.025070，生成基线为 0.043916，只保留 57.1%，配对增量 CI
`[-0.022677, -0.015015]`，正式否决。

冻结配置后完成一次性完整 Test 确认：原 V2、动态遗忘、learned hybrid 的 Recall@10
分别为 0.040019、0.038452、0.022685。动态遗忘保留率 96.08%，相对原 V2 配对增量
95% CI `[-0.005463, +0.002329]`；hybrid 保留率 56.69%，配对增量 CI
`[-0.022460, -0.012208]`。逐种子 McNemar 与 5,000 次用户 bootstrap 同样确认 hybrid
总体 Top-10 稳定下降。Test 不参与训练、选参或重选，原冻结决策不变。

UniSRec 已完成官方实现与 Amazon 2023/BLaIR 适配，但 CPU 单 worker 训练在本地资源下
耗时不可接受，用户主动停止；三个种子均无完整正式结果。All Beauty、Toys and Games、
Sports and Outdoors 的真实数据审计和 BLaIR 编码已经完成，`synthetic_data_used=false`；
外部 V2/adaptive 训练未执行，因此不报告跨品类效果。原 V2
`token_loss_010 + Beam-20/Top-10` 仍为最终默认方案。

- `artifacts/v2_4_adaptive_forgetting/summary.json`
- `artifacts/v2_4_hybrid_scorer_multiseed/summary.json`
- `artifacts/v2_4_external_preparation/state.json`
- `artifacts/unisrec_amazon2023_musical_multiseed/state.json`
- `artifacts/v2_4_external_validation/state.json`
- `artifacts/v2_4_test_confirmation/{summary.json,paired_tests.json,REPORT.md}`
- 完整配置、复现命令和数据 SHA-256：
  [docs/V2_4_ENHANCEMENT_PROTOCOL.md](docs/V2_4_ENHANCEMENT_PROTOCOL.md)

## V3 Causal Uplift

Amazon Reviews 2023 只有已发生的评论行为，没有候选级曝光/处理分配、合格对照组、固定
归因窗口转化结果和可审计 propensity。因而该数据可以用于 V1/V2 排序，但不能单独识别
增量转化。

项目已经实现 causal-readiness 门禁、overlap 校验、AIPW/DR pseudo-outcome 和 uplift
reranker；正式 V3 训练会保持阻断，直到接入真实的曝光/对照日志。审计结果：

- `artifacts/causal_readiness_amazon2023.json`
- `status=not_identifiable`
- 禁止把“出现评论”当 treatment，禁止把未评论商品标成 non-conversion。

辅助验证已接入官方 `criteo/criteo-uplift`。本地全量审计通过 13,979,592 行，含
11,882,655 treatment、2,096,937 control 和 40,774 conversion；源文件 SHA-256 为
`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`。全量 80/10/10
确定性切分和 outcome/effect 各 5 epoch 已完成：

| Split | AUUC | Qini coefficient | Uplift@10% | Observed ATE |
|---|---:|---:|---:|---:|
| Valid | -0.002476 | -2044.602 | 0.000761 | 0.001133 |
| Test | -0.002395 | -2018.507 | 0.001361 | 0.001133 |

训练损失稳定收敛且指标均为有限值，但 AUUC/Qini 在 Valid 与 Test 均为负，当前 effect
模型没有形成可靠的全局 uplift 排序能力，不能进入 Amazon 重排主线。Test Uplift@10%
虽为正且略高于观测 ATE，也不足以推翻积分排序指标的失败结论。Criteo 官方数据做过非均匀
下采样，Observed ATE 不应解释为原始总体绝对增量；该结果仅作为真实随机实验数据上的辅助
基准，Amazon 生产结论仍需真实曝光/对照日志。

- `artifacts/criteo_uplift_dr_full/report.json`
- report SHA-256：`88af39d9d1f35ac060e49eae4b1199f4731c3ac8ab4025f4134dea5fe80a2a93`
- outcome/effect model SHA-256：
  `938d6d4ccb7b9894202639da4fdfed3dcc68780c6ca84078988b1b6204c17b9f` /
  `8dcb5c6576fa718d303f8be0e79b53b6692ae3862c7d6041bbdc2d1482150d12`

### V3-lite 强基线复评

旧神经 DR 的负结果之后，补充 S/T/X/DR learner、三折 OOF nuisance cross-fitting、
response-only 强基线、三种子 Student-t CI 和 Test 分层 Poisson bootstrap。正式样本为从
全部 13,979,592 行中 uniform reservoir 抽取的 1,000,000 行真实记录；Validation/Test
分别为 99,995/100,002 行。Validation 只选出 `dr_learner` 与
`response_rank + 0.5 * uplift_rank`，写入 `selection.json` 后才运行 Test。

| Test 指标 | 冻结策略 | Bootstrap 95% CI |
|---|---:|---:|
| Qini/row | 0.0004235 | [0.0001197, 0.0007030] |
| Policy incremental@10% | 0.0012437 | [0.0005855, 0.0018863] |
| 每千样本增量（Criteo 下采样尺度） | 1.2437 | [0.5855, 1.8863] |

冻结策略相对 random targeting 的 Qini/row 差值区间为
`[0.0001873, 0.0008476]`，但相对 response-only 的 Qini/row 差值区间为
`[-0.0000247, 0.0000286]`，policy incremental@10% 差值区间为
`[-0.0000121, 0.0001644]`。后两者跨 0，因此结论冻结为：targeting 排序信号有效，
但没有充分证据证明 uplift 模块相对一般响应率模型带来额外稳定收益。

- `artifacts/v3_lite_formal/{selection.json,report.json,REPORT.md,official_test_scores.npz}`
- [docs/V3_FINAL_REPORT.md](docs/V3_FINAL_REPORT.md)
