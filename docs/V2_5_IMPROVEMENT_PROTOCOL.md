# V2.5 Baseline-Preserving Joint Scorer 协议

状态：**已结束并冻结（2026-11-11，未升级）**。V2.4 Test 与原 V2 默认结论保持冻结；
V2.5 全程只使用 Train 与 Validation，未读取 Test。三种子冻结确认未通过预注册的尾部
命中与尾部覆盖稳定性门槛，因此原 V2 继续作为默认方案。

## 三种子冻结结论

冻结变体为 `b0.5_tail8_preserve4_scale1`，seed 固定为 `2026/2027/2028`：

- 总体 Recall@10 平均保留率为 `99.961%`，增量 Student-t 95% CI 为
  `[-0.000017416, -0.000017405]`；
- Observed-Tail Recall@10 平均增量为 `+0.000364`，95% CI
  `[-0.000831, +0.001558]`，下界未大于 0；
- Observed-Tail 目录覆盖平均增量为 `+0.069075`，95% CI
  `[-0.164446, +0.302596]`，下界未大于 0；
- 新增排序延迟均值为 `0.800 ms/example`，95% CI
  `[0.355, 1.245]`；
- seed 2026 单种子门禁通过，但 seed 2027/2028 均未通过，2028 的尾部 Recall@10
  增量为 0。

因此 V2.5 只支持“有界残差能够基本保持总体 Recall”的结论，不支持“稳定提高长尾命中或
覆盖”的结论。完整结果见
`artifacts/v2_5_multiseed/summary/{report.json,REPORT.md}`。后续 V2.6 不得通过重新解释
V2.5 Test、追加有利种子或修改尾部分组来翻转这一冻结结论。

## 失败归因

V2.4 learned hybrid 将 BLaIR-only 候选的 generator relevance 设为 0，再让线性 scorer
重写整个排序。Test Recall@10 只保留原 V2 的 56.69%。这不能解释为语义检索无效：候选
oracle 显示 BLaIR 能召回大量 Observed-Tail 目标，但旧 scorer 没有为这些候选计算与 Beam
候选同口径的生成似然。

## 阶段门禁

1. **全候选生成评分**：对 `Beam-20 ∪ BLaIR-Tail-50` 中每个商品 teacher-force 完整
   Semantic ID，累计全词表 token log-probability，与 Beam 的评分定义一致。
2. **零初始化残差**：新 scorer 从原 V2 排序开始，只能在有证据时做有界修正；不得完全
   重写生成分数。
3. **Train hard negatives**：使用 Train 的滚动 next-item 样本，负例来自 Beam false
   positives、BLaIR 近邻和高频混淆商品；Validation 只选参。
4. **不确定性门控**：仅在 Beam entropy 高、Top-1/Top-10 margin 小或语义候选证据强时
   开启残差。
5. **安全窗口混合**：以最近 10 条为不可覆盖的主体，20/50 条历史只能提供有界残差。

预注册升级门槛：

- Validation Recall@10 平均保留率至少 99%；
- 相对原 V2 的三种子 Recall@10 CI 下界不低于 `-0.0004`；
- NDCG@10 不得稳定下降；
- Observed-Tail Recall@10 增量 CI 下界大于 0；
- Observed-Tail 目录覆盖增量 CI 下界大于 0；
- 同次运行新增排序开销不超过 5 ms/example。

## 第一阶段诊断

先运行 seed 2026、5,000 条 Validation reservoir pilot。通过代码与评分一致性门禁后才运行
三种子全量 Validation。诊断同时报告：

- 原 Beam、teacher-score union 和轻量有界语义残差的 Recall/NDCG；
- BLaIR-only 正例在 union 内的 teacher rank、semantic rank 和 pairwise AUC；
- generator / BLaIR / union candidate oracle；
- 候选缓存与所有输入 SHA-256。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/analyze_v2_5_teacher_rescoring.py \
  --artifact-dir artifacts/amazon2023_musical_instruments_long_tail_training_v1/formal/token_loss_010/seed_2026 \
  --seed 2026 --max-examples 5000 --selection-strategy reservoir \
  --output-dir artifacts/v2_5_teacher_rescoring/pilot_seed_2026
```

第一阶段不训练任何参数，也不访问 Test。

## seed 2026 诊断结果

5,000 条 Validation reservoir pilot 已完成：

- Beam-20 candidate oracle Recall 为 `0.0656`；BLaIR Observed-Tail-50 为
  `0.0110`，union 为 `0.0764`，证明语义检索提供了真实增量候选；
- 54 个 BLaIR-only 正例的 teacher-score 平均 rank 为 `45.26`，Top-20 rate
  为 `0`，pairwise AUC 为 `0.358 [0.300, 0.415]`；
- 同一批正例的 BLaIR semantic rank 均值为 `22.80`，Top-10 rate 为
  `29.63%`，pairwise AUC 为 `0.683 [0.624, 0.743]`；
- 固定权重 `0.01–0.10` 未产生 Observed-Tail Top-10 命中，权重 `0.10`
  已把总体 Recall@10 从 `0.0450` 降至 `0.0446`；
- union teacher scoring 用时 `254.99 s / 5,000`，即约 `51.0 ms/example`，
  不满足线上新增排序开销不超过 5 ms/example 的门槛。

结论：候选召回门禁通过，纯 teacher 重排与固定残差门禁失败。teacher score
只保留为离线失败诊断，不作为最终线上特征。第二阶段使用原 Beam score 作为
不可覆盖主干，以 Train-only hard pairs 学习零初始化、有界语义残差和不确定性
gate；Validation 只用于选取已经训练完成的配置。
